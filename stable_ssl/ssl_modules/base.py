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

        return self.dataset_to_loader(train_dataset)

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

        return self.dataset_to_loader(eval_dataset)

    def compute_loss(self):
        embeds = self.forward(torch.cat([self.data[0][0], self.data[0][1]], 0))
        return self.compute_ssl_loss(embeds) + self.compute_classifier_loss(embeds)

    def compute_classifier_loss(self, embeds):
        preds = self.classifier(embeds.detach())
        return F.cross_entropy(preds, self.data[1].repeat(2))

    def compute_ssl_loss(self, embeds):
        raise NotImplementedError
