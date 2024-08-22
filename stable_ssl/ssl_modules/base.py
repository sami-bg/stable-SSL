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
        # backbone
        model, fan_in = load_model_without_classifier(self.config.model.backbone_model)
        self.backbone = model.train()

        # projector
        sizes = [2048] + list(map(int, self.config.model.projector.split("-")))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # linear probes
        self.classifier = torch.nn.Linear(fan_in, 10)

    def forward(self, x):
        return self.backbone(x)

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

    def compute_loss(self):
        embeds = self.forward(torch.cat([self.data[0], self.data[1]], 0))
        return self.compute_ssl_loss(embeds) + self.compute_classifier_loss(embeds)

    def compute_classifier_loss(self, embeds):
        preds = self.classifier(embeds.detach())
        return F.cross_entropy(preds, self.data[2])

    def compute_ssl_loss(self, embeds):
        raise NotImplementedError
