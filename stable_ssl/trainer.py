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
import warnings
import yaml
import time
import dataclasses
from pathlib import Path
from tqdm import tqdm
import submitit

import wandb

import torch
import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils.data import RandomSampler

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
    to_device,
)
from stable_ssl.utils.eval import AverageMeter, accuracy
from stable_ssl.config import TrainerConfig
from .sampler import (
    PositivePairSampler,
    ValSampler,
)

DATASETS = {
    "CIFAR10": torchvision.datasets.CIFAR10,
    "CIFAR100": torchvision.datasets.CIFAR100,
}


class Trainer(torch.nn.Module):
    r"""Base class for training a model.

    Parameters:
    -----------
    config : TrainerConfig
        Parameters for Trainer organized in groups.
        For details, see the `TrainerConfig` class in `config.py`.
    """

    def __init__(self, config: TrainerConfig):
        super().__init__()
        self.config = copy.deepcopy(config)

    def __call__(self):

        self.folder = Path(self.config.log.folder).absolute()
        self.folder.mkdir(parents=True, exist_ok=True)

        if self.config.log.project is not None:
            print(
                f"[stable-SSL] \t=> Initializating wandb for logging in {self.folder}."
            )
            wandb.init(
                entity=self.config.log.entity,
                project=self.config.log.project,
                config=dataclasses.asdict(self.config),
                name=self.config.log.run_name if self.config.log.run_name else None,
                dir=str(self.folder),
                resume="allow",
            )
        else:
            print(f"[stable-SSL] \t=> Dumping config file in {self.folder}.")
            with open(self.folder / "hparams.yaml", "w+") as f:
                yaml.dump(dataclasses.asdict(self.config), f, indent=2)

        logging.basicConfig(level=self.config.log.log_level)
        seed_everything(self.config.hardware.seed)

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.config.hardware.float16)
        self._set_device()

        if not self.config.log.eval_only:
            logging.info("[stable-SSL] Creating train_loader dataset.")
            self.train_loader = self.initialize_train_loader()
            assert hasattr(self, "train_loader")
            logging.info(
                "[stable-SSL] \t=> Found training set of length "
                f"{len(self.train_loader)}."
            )
        else:
            logging.info("[stable-SSL] \t=> No training set loaded since eval_only.")

        logging.info("[stable-SSL] Creating val_loader dataset.")
        try:
            self.val_loader = self.initialize_val_loader()
            logging.info(
                "[stable-SSL] \t=> Found validation set of length "
                f"{len(self.val_loader)}."
            )
        except NotImplementedError:
            logging.info(
                "[stable-SSL] \t=> Found no implementation of initialize_val_loader. "
                "Skipping for now."
            )
            self.val_loader = None

        logging.info("[stable-SSL] Calling initialize_modules() method.")
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
                f"[stable-SSL] \t=> Found module '{name}' with\n\t\t\t- {trainable} "
                "trainable parameters."
            )

        if not self.config.log.eval_only:
            logging.info("[stable-SSL] Calling initialize_optimizer() method.")
            self.optimizer = self.initialize_optimizer()
            logging.info("[stable-SSL] Calling initialize_scheduler() method.")
            try:
                self.scheduler = self.initialize_scheduler()
            except NotImplementedError:
                logging.info("[stable-SSL] No scheduler given...")
        else:
            logging.info(
                "[stable-SSL] Mode is eval_only, skipping optimizer and "
                "scheduler initializations."
            )

        logging.info("[stable-SSL] Calling load_checkpoint() method.")
        self.load_checkpoint()
        self.start_time = time.time()
        self.execute()

    def execute(self):
        """Routine that is executed after the class is initialized.

        This will commonly consist of training + evaluation.
        Can be customized by the user to fit the use-cases.
        This is just a boilerplate version that provides minimal things.
        """
        if self.config.log.eval_only:
            self.eval_epoch()
        else:
            try:
                self.before_train_all_epochs()
                self._train_all_epochs()
                self.eval_epoch()  # always eval the model after training
            except BreakAllEpochs:
                self.cleanup()
                wandb.finish()

    def _train_all_epochs(self):
        while self.epoch < self.config.optim.epochs:

            if hasattr(self, "train_sampler"):
                self.train_sampler.set_epoch(self.epoch)

            try:
                self._train_epoch()
            except BreakEpoch:
                print("[stable-SSL] Train epoch cut by user. Going to the next one.")
            except NanError:
                print("[stable-SSL] Nan error.")
                return
            except Exception as e:
                raise (e)

            if self.config.log.eval_each_epoch:
                self.eval_epoch()
            self.epoch = self.epoch + 1

            freq = self.config.log.checkpoint_frequency
            if self.epoch % freq == 0:
                print("[stable-SSL] Checkpointing everything to restart if needed.")
                self.save_checkpoint("tmp_checkpoint.ckpt", model_only=False)

        # at the end of training, we (optionally) save the final model
        if self.config.log.save_final_model:
            self.save_checkpoint(
                f"{self.config.log.final_model_name}.ckpt", model_only=True
            )
        # and remove any temporary checkpoint
        (self.folder / "tmp_checkpoint.ckpt").unlink(missing_ok=True)

        wandb.finish()

    def _train_epoch(self):

        # hierarchically set up all modules in train mode
        self.before_train_epoch()

        # we do not ensure that the model is still in train mode to not
        # override any user desired behavior, simply speak out
        if not self.training:
            logging.warn(
                "[stable-SSL] Starting training epoch but model is no longer in "
                "train mode after call to before_train_epoch()."
            )

        if self.config.optim.max_steps < 0:
            max_steps = len(self.train_loader)
        elif 0 < self.config.optim.max_steps < 1:
            logging.info(
                f"[stable-SSL] \t=> Training on {self.config.optim.max_steps*100}% of "
                "the training dataset"
            )
            max_steps = int(self.config.optim.max_steps * len(self.train_loader))
        else:
            max_steps = self.config.optim.max_steps

        for step, data in enumerate(
            tqdm(self.train_loader, total=max_steps, desc=f"Training: {self.epoch=}")
        ):
            # set up the data to have easy access throughout the methods
            self.step = step
            self.data = to_device(data, self.this_device)

            try:
                # call any user specified pre-step functions
                self.before_train_step()

                # perform the gradient step
                self.train_step()

                # call any user specified post-step function
                self.after_train_step()
            except BreakStep:
                logging.warn("[stable-SSL] train_step has been interrupted by user.")

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
            logging.info("[stable-SSL] No val_loader hence skipping eval epoch.")
            return

        # set-up model in eval mode + reset metrics
        self.before_eval_epoch()

        # we do not ensure that the model is still in eval mode to not
        # override any user desired behavior
        if self.training:
            warnings.warn(
                "[stable-SSL] Starting eval epoch but model is not in "
                "eval mode after call to before_eval_epoch()."
            )

        try:
            max_steps = len(self.val_loader)
            with torch.inference_mode():
                for step, data in tqdm(
                    enumerate(self.val_loader),
                    total=max_steps,
                    desc=f"Eval: {self.epoch=}",
                ):
                    self.step = step
                    self.data = to_device(data, self.this_device)

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
            print("[stable-SSL] Eval epoch cut by user...")
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

        if self.config.log.project is not None:
            wandb.log(
                {"train/loss": loss.item(), "epoch": self.epoch, "step": self.step}
            )

        self.scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        self.scaler.unscale_(self.optimizer)
        if self.config.optim.grad_max_norm is not None:
            # Since the gradients of optimizer's assigned params are unscaled,
            # clips as usual:
            torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.config.optim.grad_max_norm
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.scheduler.step()
        if self.config.log.project is not None:
            wandb.log(
                {
                    "train/lr": self.scheduler.get_last_lr()[0],
                    "step": self.step,
                    "epoch": self.epoch,
                }
            )

    def _set_device(self):
        try:
            self.config.hardware = setup_distributed(self.config.hardware)
            self._device = f"cuda:{self.config.hardware.gpu}"
        except RuntimeError as e:
            print(e)
            # self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = f"cuda:{self.config.hardware.gpu}"
            self.config.hardware.gpu = 0
            self.config.hardware.world_size = 1
        torch.cuda.set_device(self.config.hardware.gpu)

    def checkpoint(self):
        # the checkpoint method is called asynchroneously when the slurm manager
        # sends a preemption signal, with the same arguments as the __call__ method
        # "self" is your callable, at its current state.
        # "self" therefore holds the current version of the model:
        print("[stable-SSL] Requeuing...")
        config = copy.deepcopy(self.config)
        config.log.add_version = False
        config.log.folder = self.folder.absolute().as_posix()
        model = type(self)(config)
        return submitit.helpers.DelayedSubmission(model)

    def load_checkpoint(self):
        """load a model and optionnally a scheduler/optimizer/epoch
        from a given path to checkpoint

        Args:
            load_from (str): path to checkpoint
        """
        load_from = Path(self.config.log.load_from)
        if load_from.is_file():
            logging.info(
                f"[stable-SSL] \t=> file {load_from} exists\n\t=> loading it..."
            )
            checkpoint = load_from
        elif (self.folder / "tmp_checkpoint.ckpt").is_file():
            logging.info(
                f"[stable-SSL] \t=> folder {self.folder} contains `tmp_checkpoint.ckpt`"
                " file\n\t=> loading it..."
            )
            checkpoint = self.folder / "tmp_checkpoint.ckpt"
        else:
            logging.info(f"[stable-SSL] \t=> no checkpoint at `{load_from}`")
            logging.info(
                f"[stable-SSL] \t=> no checkpoint at "
                "`{self.folder / 'tmp_checkpoint.ckpt'}`"
            )
            logging.info("[stable-SSL] \t=> training from scratch...")
            self.epoch = 0
            return

        ckpt = torch.load(checkpoint, map_location="cpu")

        for name, model in self.named_children():
            if name not in ckpt:
                logging.info(f"[stable-SSL] \t\t=> {name} not in ckpt, skipping...")
                continue
            model.load_state_dict(ckpt[name])
            logging.info(f"[stable-SSL] \t\t=> {name} successfully loaded...")
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            logging.info("[stable-SSL] \t\t=> optimizer successfully loaded...")
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
            logging.info("[stable-SSL] \t\t=> scheduler successfully loaded...")
        if "epoch" in ckpt:
            self.epoch = ckpt["epoch"]
            logging.info(
                f"[stable-SSL] \t\t=> training will start from epoch {ckpt['epoch']}"
            )
        else:
            self.epoch = 0

    def initialize_modules(self):
        raise NotImplementedError

    def initialize_optimizer(self):
        if self.config.optim.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.config.optim.lr,
                weight_decay=self.config.optim.weight_decay,
            )
        if self.config.optim.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config.optim.lr,
                weight_decay=self.config.optim.weight_decay,
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
            )
        return optimizer

    def initialize_scheduler(self):
        min_lr = self.config.optim.lr * 0.005
        peak_step = 10 * len(self.train_loader)
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

        self.top1 = AverageMeter("Acc@1")
        self.top5 = AverageMeter("Acc@5")

    def before_eval_step(self):
        return

    def after_eval_step(self):
        return

    def after_eval_epoch(self):
        wandb.log(
            {
                "epoch": self.epoch,
                "test/acc1": self.top1.avg,
                "test/acc5": self.top5.avg,
            }
        )

    def eval_step(self):
        output = self.forward(self.data[0])
        if hasattr(self, "classifier"):
            output = self.classifier(output)
        acc1, acc5 = accuracy(output, self.data[1], topk=(1, 5))
        self.top1.update(acc1.item(), self.data[0].size(0))
        self.top5.update(acc5.item(), self.data[0].size(0))

    def compute_loss(self):
        raise NotImplementedError

    def dataset_to_loader(self, dataset):
        if self.config.hardware.world_size > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            assert self.config.optim.batch_size % self.config.hardware.world_size == 0
        else:
            sampler = RandomSampler(dataset)

        per_device_batch_size = (
            self.config.optim.batch_size // self.config.hardware.world_size
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=per_device_batch_size,
            num_workers=self.config.hardware.workers,
            pin_memory=True,
            sampler=sampler,
        )

        return loader

    def initialize_train_loader(self):
        train_dataset = DATASETS[self.config.data.dataset](
            root=self.config.data.data_dir,
            train=True,
            download=True,
            transform=PositivePairSampler(dataset=self.config.data.dataset),
        )

        return self.dataset_to_loader(train_dataset)

    def initialize_val_loader(self):
        eval_dataset = DATASETS[self.config.data.dataset](
            root=self.config.data.data_dir,
            train=False,
            download=True,
            transform=ValSampler(dataset=self.config.data.dataset),
        )

        return self.dataset_to_loader(eval_dataset)
