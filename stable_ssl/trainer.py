# -*- coding: utf-8 -*-
"""Main function for training a SSL model."""

# Author: Randall Balestriero <randallbalestriero@gmail.com>
#         Hugues Van Assel <vanasselhugues@gmail.com>
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import uuid
import copy
import logging
import torchvision
import warnings
import yaml
import time
import dataclasses

from tqdm import tqdm
from pathlib import Path

# import tables

import torch
import torch.optim as optim

from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    ConstantLR,
)
from torch.distributed.optim import ZeroRedundancyOptimizer

from stable_ssl.utils import (
    BreakAllEpochs,
    BreakEpoch,
    NanError,
    BreakStep,
    seed_everything,
    setup_distributed,
    FullGatherLayer,
    LARS,
    LinearWarmupCosineAnnealing,
)
from stable_ssl.config import TrainerConfig


class Trainer(torch.nn.Module):
    r"""Base class for training a Self-Supervised Learning (SSL) model.

    Parameters:
    -----------
    config : TrainerConfig
        Parameters for Trainer organized in the following groups :
        'optim', 'architecture', 'hardware', 'log'.
        For details, see the `TrainerConfig` class in `config.py`.
    """

    def __init__(self, config: TrainerConfig):
        super().__init__()

        self.config = copy.deepcopy(config)

        if self.config.log.add_version:
            self.folder = (Path(self.config.log.folder) / str(uuid.uuid4())).absolute()
        else:
            self.folder = Path(self.config.log.folder).absolute()

        self.folder.mkdir(parents=True, exist_ok=True)

        print(f"[stable-SSL] Logging in {self.folder}")
        print(f"[stable-SSL] \t=> Dumping hyper-parameters...")
        with open(self.folder / "hparams.yaml", "w+") as f:
            yaml.dump(dataclasses.asdict(config), f, indent=2)

    def __call__(self):

        logging.basicConfig(level=self.config.log.log_level)
        seed_everything(self.config.hardware.seed)

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.config.hardware.float16)

        try:
            self.config.hardware = setup_distributed(self.config.hardware)
            self._device = f"cuda:{self.config.hardware.gpu}"
        except RuntimeError as e:
            print(e)
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self.config.hardware.gpu = 0
            self.config.hardware.world_size = 1
        torch.cuda.set_device(self.config.hardware.gpu)

        if not self.config.log.eval_only:
            logging.info("Creating train_loader dataset...")
            self.train_loader = self.initialize_train_loader()
            assert hasattr(self, "train_loader")
            logging.info(f"\t=> Found training set of length {len(self.train_loader)}")
        else:
            logging.info(f"\t=> No training set loaded since eval_only")

        logging.info("Creating val_loader dataset...")
        try:
            self.val_loader = self.initialize_val_loader()
            logging.info(f"\t=> Found validation set of length {len(self.val_loader)}")
        except NotImplementedError:
            logging.info(
                f"\t=> Found no implementation of initialize_val_loader... skipping"
            )
            self.val_loader = None

        logging.info("Calling initialize_modules() method...")
        self.initialize_modules()

        for name, module in self.named_children():
            if self.config.model.memory_format == "channels_last":
                module.to(memory_format=torch.channels_last)
            if self.config.model.sync_batchnorm:
                module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)
            module.to(self.this_device)
            has_parameters = False
            if sum(p.numel() for p in module.parameters() if p.requires_grad) > 0:
                has_parameters = True
            if self.config.hardware.world_size > 1 and has_parameters:
                module = torch.nn.parallel.DistributedDataParallel(
                    module, device_ids=[self.config.hardware.gpu]
                )
            setattr(self, name, module)

            trainable = sum(
                param.numel() for param in module.parameters() if param.requires_grad
            )
            logging.info(
                f"\t=> Found module '{name}' with\n\t\t\t- {trainable} trainable parameters"
            )

        if not self.config.log.eval_only:
            logging.info("Calling initialize_optimizer() method...")
            self.optimizer = self.initialize_optimizer()
            logging.info("Calling initialize_scheduler() method...")
            try:
                self.scheduler = self.initialize_scheduler()
            except NotImplementedError:
                logging.info("No scheduler given...")
        else:
            logging.info(
                f"Not calling initialize_optimizer() method... since eval_only"
            )

        logging.info("Calling load_checkpoint() method...")
        self.load_checkpoint()
        self.start_time = time.time()
        self.execute()

    def execute(self) -> None:
        """Routine that is executed after the class is initialized.

        This will commonly consist of training + evaluation. Can be customized by the user to fit the use-cases. This is just a boilerplate version that provides minimal things.

        Args:
            eval_only (bool): whether to only eval the model, or also train
        """
        if self.config.log.eval_only:
            self.eval_epoch()
        else:
            try:
                self.before_train_all_epochs()
                self._train_all_epochs()
                # after training, we always eval the model even if done per epoch
                self.eval_epoch()
            except BreakAllEpochs:
                self.cleanup()

    def _train_all_epochs(self):
        while self.epoch < self.config.optim.epochs:
            if hasattr(self, "train_sampler"):
                self.train_sampler.set_epoch(self.epoch)
            try:
                self._train_epoch()
            except BreakEpoch:
                print("Train epoch cut by user...")
                print("Going to the next one...")
            except NanError:
                print("Nan error...")
                return
            except Exception as e:
                raise (e)

            if self.config.log.eval_each_epoch:
                self.eval_epoch()
            self.epoch = self.epoch + 1

            freq = self.config.log.checkpoint_frequency
            if self.epoch % freq == 0:
                print("checkpointing everything to restart if needed...")
                self.save_checkpoint("tmp_checkpoint.ckpt", model_only=False)

        # at the end of training, we (optionally) save the final model
        if self.config.log.save_final_model:
            self.save_checkpoint(
                f"{self.config.log.final_model_name}.ckpt", model_only=True
            )
        # and remove any temporary checkpoint
        (self.folder / "tmp_checkpoint.ckpt").unlink(missing_ok=True)

    def _train_epoch(self):

        # hierarchically set up all modules in train mode
        self.before_train_epoch()

        # we do not ensure that the model is still in train mode to not
        # override any user desired behavior, simply speak out
        if not self.training:
            logging.warn(
                "starting training epoch but model is no longer in\
                    train mode after call to before_train_epoch()"
            )

        if self.config.optim.max_steps < 0:
            max_steps = len(self.train_loader)
        elif 0 < self.config.optim.max_steps < 1:
            logging.info(
                f"\t=> Training on {self.config.optim.max_steps*100}% of "
                "the training dataset"
            )
            max_steps = int(self.config.optim.max_steps * len(self.train_loader))
        else:
            max_steps = self.config.optim.max_steps

        for step, ((view1, view2), _) in enumerate(
            tqdm(self.train_loader, total=max_steps, desc=f"Training: {self.epoch=}")
        ):

            # set up the data to have easy access throughout the methods
            self.step = step
            view1 = view1.to(self.this_device, non_blocking=True)
            view2 = view2.to(self.this_device, non_blocking=True)
            self.data = [view1, view2]

            try:
                # call any user specified pre-step function
                self.before_train_step()

                # perform the gradient step
                self.train_step()

                # call any user specified post-step function
                self.after_train_step()
            except BreakStep:
                logging.warn("train_step has been interrupted by user...")

            # we cut early in case the user specifies to only use
            # X% of the training dataset
            if step >= max_steps:
                break

        # call any user specified post-epoch function
        self.after_train_epoch()

        # be sure to clean up to avoid silent bugs
        self.data = None

    def eval_epoch(self) -> dict:

        if self.val_loader is None:
            logging.info("No val_loader hence skipping eval epoch")
            return

        # set-up model in eval mode + reset metrics
        self.before_eval_epoch()

        # TO DO : add config to control this eval optimizer
        self.optimizer_classifier = optim.SGD(
            self.classifier.parameters(), lr=self.config.optim.lr
        )

        # we do not ensure that the model is still in eval mode to not
        # override any user desired behavior
        if self.training:
            warnings.warn(
                "starting eval epoch but model is not in\
                    eval mode after call to before_eval_epoch()"
            )

        try:
            max_steps = len(self.val_loader)
            with torch.no_grad():
                for step, (x, y) in tqdm(
                    enumerate(self.val_loader),
                    total=max_steps,
                    desc=f"Eval: {self.epoch=}",
                ):
                    self.step = step

                    x = x.to(self.this_device, non_blocking=True)
                    y = y.to(self.this_device, non_blocking=True)
                    self.data = [x, y]

                    # call any user specified pre-step function
                    self.before_eval_step()

                    # call the eval step
                    with torch.amp.autocast(
                        "cuda", enabled=self.config.hardware.float16
                    ):
                        self.eval_step()

                    # call any user specified post-step function
                    self.after_eval_step()

        except BreakEpoch:
            print("Eval epoch cut by user...")
        except Exception as e:
            raise (e)

        # be sure to clean up to avoid silent bugs
        self.data = None

        # call any user specified post-epoch function
        self.after_eval_epoch()

    def train_step(self):
        with torch.amp.autocast("cuda", enabled=self.config.hardware.float16):
            loss = self.compute_loss()
        if np.isnan(loss.item()):
            raise NanError
        self.scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        self.scaler.unscale_(self.optimizer)
        if self.config.optim.grad_max_norm is not None:
            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.config.optim.grad_max_norm
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def checkpoint(self):
        # the checkpoint method is called asynchroneously when the slurm manager
        # sends a preemption signal, with the same arguments as the __call__ method
        # "self" is your callable, at its current state.
        # "self" therefore holds the current version of the model:
        print("Requeuing...")
        config = copy.deepcopy(self.config)
        config.log.add_version = False
        config.log.folder = self.folder.absolute().as_posix()
        model = type(self)(config)
        return submitit.helpers.DelayedSubmission(model)

    def load_checkpoint(self):
        """load a model and optionnally a scheduler/optimizer/epoch from a given path to checkpoint

        Args:
            load_from (str): path to checkpoint
        """
        load_from = Path(self.config.log.load_from)
        if load_from.is_file():
            logging.info(f"\t=> file {load_from} exists\n\t=> loading it...")
            checkpoint = load_from
        elif (self.folder / "tmp_checkpoint.ckpt").is_file():
            logging.info(
                f"\t=> folder {self.folder} contains `tmp_checkpoint.ckpt` file\n\t=> loading it..."
            )
            checkpoint = self.folder / "tmp_checkpoint.ckpt"
        else:
            logging.info(f"\t=> no checkpoint at `{load_from}`")
            logging.info(
                f"\t=> no checkpoint at `{self.folder / 'tmp_checkpoint.ckpt'}`"
            )
            logging.info("f\t=> training from scratch...")
            self.epoch = 0
            return

        ckpt = torch.load(checkpoint, map_location="cpu")

        for name, model in self.named_children():
            if name not in ckpt:
                logging.info(f"\t\t=> {name} not in ckpt, skipping...")
                continue
            model.load_state_dict(ckpt[name])
            logging.info(f"\t\t=> {name} successfully loaded...")
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            logging.info(f"\t\t=> optimizer successfully loaded...")
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
            logging.info(f"\t\t=> scheduler successfully loaded...")
        if "epoch" in ckpt:
            self.epoch = ckpt["epoch"]
            logging.info(f"\t\t=> training will start from epoch {ckpt['epoch']}")
        else:
            self.epoch = 0

    def initialize_train_loader(self):
        raise NotImplementedError

    def initialize_val_loader(self):
        raise NotImplementedError

    def initialize_modules(self):
        raise NotImplementedError

    def initialize_optimizer(self):
        if self.config.optim.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config.optim.lr,
                weight_decay=self.config.optim.weight_decay,
                eps=1e-8,
                betas=self.config.optim.betas,
            )
        elif self.config.optim.optimizer == "RMSprop":
            optimizer = torch.optim.RMSprop(
                self.parameters(),
                lr=self.config.optim.lr,
                weight_decay=self.config.optim.weight_decay,
                momentum=self.config.optim.momentum,
            )
        elif self.config.optim.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.config.optim.lr,
                weight_decay=self.config.optim.weight_decay,
                momentum=self.config.optim.momentum,
            )
        elif self.config.optim.optimizer == "LARS":
            optimizer = LARS(
                self.parameters(),
                lr=self.config.optim.lr,
                weight_decay=self.config.optim.weight_decay,
                momentum=self.config.optim.momentum,
                epsilon=1e-8,
            )
        return optimizer

    def initialize_scheduler(self):
        min_lr = self.config.optim.lr * 0.005
        peak_step = 5 * len(self.train_loader)
        total_steps = self.config.optim.epochs * len(self.train_loader)
        return LinearWarmupCosineAnnealing(
            self.optimizer, end_lr=min_lr, peak_step=peak_step, total_steps=total_steps
        )

    def save_checkpoint(self, name, model_only):
        if self.config.hardware.world_size > 1:
            if torch.distributed.get_rank() != 0:
                return
        saving_name = self.folder / name
        state = {}
        for subname, model in self.named_children():
            state[subname] = model.state_dict()
        if model_only:
            torch.save(state, saving_name)
            return
        if hasattr(self, "optimizer"):
            state["optimizer"] = self.optimizer.state_dict()
        if hasattr(self, "scheduler"):
            state["scheduler"] = self.scheduler.state_dict()
        state["epoch"] = self.epoch

        torch.save(state, saving_name)

    def generate_logging_default_bucket(self):
        cur_time = time.time()
        rel_time = cur_time - self.start_time
        if self.config.hardware.world_size > 1:
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        bucket = {
            "rank": rank,
            "timestamp": cur_time,
            "relative_time": rel_time,
            "training": self.training,
            "epoch": self.epoch,
            "step": self.step,
        }
        return bucket

    def cleanup(self):
        if self.config.hardware.world_size > 1:
            print("Cleaning distributed processes...")
            torch.distributed.destroy_process_group()
        else:
            print("Not using distributed... nothing to clean")

    def gather(self, x):
        return FullGatherLayer.apply(x)

    @property
    def rank(self):
        if self.config.hardware.world_size > 1:
            return torch.distributed.get_rank()
        return 0

    @property
    def epoch(self):
        if not hasattr(self, "_epoch"):
            return None
        return self._epoch

    @epoch.setter
    def epoch(self, value):
        self._epoch = value

    @property
    def step(self):
        if not hasattr(self, "_step"):
            return None
        return self._step

    @step.setter
    def step(self, value):
        self._step = value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def this_device(self):
        return self._device

    def before_train_all_epochs(self):
        return

    def before_train_epoch(self):
        self.train()

    def before_train_step(self):
        self.optimizer.zero_grad(set_to_none=True)
        return

    def after_train_step(self):
        return

    def after_train_epoch(self):
        return

    def before_eval_epoch(self):
        self.eval()

    def before_eval_step(self):
        return

    def after_eval_step(self):
        return

    def after_eval_epoch(self):
        return

    def eval_step(self):
        raise NotImplementedError

    def compute_loss(self):
        raise NotImplementedError
