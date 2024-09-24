import torch
import torch.nn.functional as F
from ..base import BaseModel, BaseModelConfig
from torch import nn
from ...utils import load_model
from dataclasses import dataclass


@dataclass
class SSLConfig(BaseModelConfig):
    """
    Configuration for the SSL model parameters.

    Parameters:
    -----------
    projector : str
        Architecture of the projector head. Default is "2048-128".
    """

    projector: str = "2048-128"


class SSLTrainer(BaseModel):
    r"""Base class for training a Self-Supervised Learning (SSL) model.

    Parameters:
    -----------
    config : TrainerConfig
        Parameters for Trainer organized in groups.
        For details, see the `TrainerConfig` class in `config.py`.
    """

    def initialize_modules(self):
        # backbone
        model, fan_in = load_model(
            name=self.config.model.backbone_model,
            n_classes=self.config.data.num_classes,
            with_classifier=False,
            pretrained=False,
            dataset=self.config.data.dataset,
        )
        self.backbone = model.train()

        # projector
        sizes = [fan_in] + list(map(int, self.config.model.projector.split("-")))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # linear probes
        self.classifier = torch.nn.Linear(fan_in, self.config.data.num_classes)

    def forward(self, x):
        return self.backbone(x)

    def compute_loss(self):
        embed_i = self.forward(self.data[0][0])
        embed_j = self.forward(self.data[0][1])

        # check that it is the right moment to do this
        if self.config.hardware.world_size > 1:
            embed_i = torch.cat(self.gather(z_i), dim=0)
            embed_j = torch.cat(self.gather(z_j), dim=0)

        embeds = torch.cat([embed_i, embed_j], dim=0)
        loss_ssl = self.compute_ssl_loss(embeds)
        loss_classifier = self.compute_classifier_loss(embeds)

        self.log({"train/loss_ssl": loss_ssl, "train/loss_classifier": loss_classifier})
        return loss_ssl + loss_classifier

    def compute_classifier_loss(self, embeds):
        preds = self.classifier(embeds.detach())
        return F.cross_entropy(preds, self.data[1].repeat(2))

    def compute_ssl_loss(self, embeds):
        raise NotImplementedError
