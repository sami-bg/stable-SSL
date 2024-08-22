import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms

from stable_ssl.utils import load_model
from stable_ssl.trainer import Trainer
from stable_ssl.config import TrainerConfig
from .positive_pair_sampler import PositivePairSampler, IMAGENET_MEAN, IMAGENET_STD


class SSLTrainer(Trainer):
    r"""Base class for training a Self-Supervised Learning (SSL) model.

    Parameters:
    -----------
    config : TrainerConfig
        Parameters for Trainer organized in groups.
        For details, see the `TrainerConfig` class in `config.py`.
    """

    def __init__(self, config: TrainerConfig):
        super().__init__(config)

    def initialize_modules(self):
        # backbone
        model, fan_in = load_model(
            name=self.config.model.backbone_model,
            with_classifier=False,
            pretrained=False,
        )
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

        train_dataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=PositivePairSampler(),
        )

        self.initialize_dataset_loader(train_dataset)

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
        eval_dataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transform,
        )

        return self.initialize_dataset_loader(eval_dataset)

    def compute_loss(self):
        embeds = self.forward(torch.cat([self.data[0][0], self.data[0][1]], 0))
        return self.compute_ssl_loss(embeds) + self.compute_classifier_loss(embeds)

    def compute_classifier_loss(self, embeds):
        preds = self.classifier(embeds.detach())
        return F.cross_entropy(preds, self.data[1])

    def compute_ssl_loss(self, embeds):
        raise NotImplementedError
