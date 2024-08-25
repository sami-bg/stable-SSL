import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms

from stable_ssl.utils import load_model
from .trainer import Trainer
from .positive_pair_sampler import (
    PositivePairSampler,
    MEAN,
    STD,
)


class Supervised(Trainer):
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

    def compute_loss(self):
        preds = self.forward(torch.cat([self.data[0][0], self.data[0][1]], 0))
        return F.cross_entropy(preds, self.data[1].repeat(2))
