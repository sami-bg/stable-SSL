import torch
import torchvision
import torch.nn as nn
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

        self.model = model.train()

        # TO DO : add config to control the size of the projection head
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(fan_in, 8096, bias=False),
            torch.nn.BatchNorm1d(8096),
            torch.nn.ReLU(True),
            torch.nn.Linear(8096, 8096, bias=False),
            torch.nn.BatchNorm1d(8096),
            torch.nn.ReLU(True),
            torch.nn.Linear(8096, 8096, bias=False),
        )

        self.classifier = torch.nn.Linear(fan_in, 1000)

    def initialize_train_loader(self):
        dataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=True,
            transform=PositivePairSampler(),
            download=True,
        )
        # dataset = torchvision.datasets.ImageFolder(
        #     self.config.data / "train", Transform()
        # )

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
        dataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        )
        # dataset = torchvision.datasets.ImageFolder(
        #     self.config.data / "eval", Transform()
        # )

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
        loss = self.compute_loss_classifier()
        if np.isnan(loss.item()):
            raise NanError
        loss.backward()
        self.optimizer_classifier.step()
