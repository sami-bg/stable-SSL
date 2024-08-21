import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader, RandomSampler

from stable_ssl.utils import load_model_without_classifier
from stable_ssl.trainer import Trainer
from stable_ssl.config import TrainerConfig
from .positive_pair_sampler import PositivePairSampler, IMAGENET_MEAN, IMAGENET_STD


class SSLTrainer(Trainer):
    def __init__(self, config: TrainerConfig):
        super().__init__(config)

    def initialize_modules(self):
        model, fan_in = load_model_without_classifier(self.config.model.backbone_model)
        # self.model = model.train()  # do we need this ?
        self.model = model

        # TO DO : add config to control the size of the projection head
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(fan_in, 2048, bias=False),
            torch.nn.BatchNorm1d(2048),
            torch.nn.ReLU(True),
            torch.nn.Linear(2048, 512, bias=False),
        )

        self.classifier = torch.nn.Linear(fan_in, 10)

    def forward(self, x):
        return self.model(x)

    def initialize_train_loader(self):
        # dataset = torchvision.datasets.ImageFolder(
        #     self.config.data.data_dir / "train", PositivePairSampler()
        # )

        dataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=PositivePairSampler(),
        )

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
            sampler=RandomSampler(dataset),
        )
        return loader

    def initialize_val_loader(self):

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        # dataset = torchvision.datasets.ImageFolder(
        #     self.config.data.data_dir + "/eval", transform
        # )
        dataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transform,
        )

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
            sampler=RandomSampler(dataset),
        )
        return loader

    def eval_step(self):
        with torch.amp.autocast("cuda", enabled=self.config.hardware.float16):
            preds = self.classifier(self.forward(self.data[0]))

            # STRANGE ERROR : I don't understand why we cannot backward here
            print("Model's requires_grad status:")
            for name, param in self.classifier.named_parameters():
                print(
                    f"{name}: requires_grad={param.requires_grad}, grad={param.grad is not None}"
                )

            loss_classifier = F.cross_entropy(preds, self.data[1])

        if np.isnan(loss_classifier.item()):
            raise NanError

        self.scaler.scale(loss_classifier).backward()
        self.scaler.unscale_(self.optimizer_classifier)
        self.scaler.step(self.optimizer_classifier)
        self.scaler.update()

    def before_eval_epoch(self):
        self.eval()

        # Freeze all parameters in the model
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze only the classifier part for fine-tuning
        for param in self.classifier.parameters():
            param.requires_grad = True

        # TO DO : add config to control this eval optimizer
        self.optimizer_classifier = optim.SGD(
            self.classifier.parameters(), lr=self.config.optim.lr
        )

    def before_eval_step(self):
        self.optimizer_classifier.zero_grad(set_to_none=True)
