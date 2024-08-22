from .trainer import Trainer
from stable_ssl.utils import load_model
import torchvision
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms

from stable_ssl.ssl_modules.positive_pair_sampler import (
    PositivePairSampler,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


class SupervisedTrainer(Trainer):
    r"""Base class for training a supervised model.

    Parameters:
    -----------
    config : TrainerConfig
        Parameters for Trainer organized in groups.
        For details, see the `TrainerConfig` class in `config.py`.
    """

    def initialize_modules(self):
        model, _ = load_model(
            name=self.config.model.backbone_model,
            n_classes=10,
            with_classifier=True,
            pretrained=False,
        )
        self.model = model.train()

    def forward(self, x):
        return self.model(x)

    def initialize_train_loader(self):
        # dataset = torchvision.datasets.ImageFolder(
        #     self.config.data.data_dir / "train", PositivePairSampler()
        # )

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        train_dataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transform,
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
        preds = self.forward(self.data[0])
        return F.cross_entropy(preds, self.data[1])
