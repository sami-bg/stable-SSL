import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms

import wandb

from stable_ssl.utils import load_model
from stable_ssl.trainer import Trainer
from stable_ssl.config import TrainerConfig


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

        if self.config.log.wandb_project is not None:
            wandb.log(
                {
                    "train/loss_ssl": loss_ssl,
                    "train/loss_classifier": loss_classifier,
                    "step": self.step,
                    "epoch": self.epoch,
                }
            )
        return loss_ssl + loss_classifier

    def compute_classifier_loss(self, embeds):
        preds = self.classifier(embeds.detach())
        return F.cross_entropy(preds, self.data[1].repeat(2))

    def compute_ssl_loss(self, embeds):
        raise NotImplementedError
