import torch
import torch.nn.functional as F
from torch import nn
import wandb

from .simclr import SimCLR
from stable_ssl.utils import load_model, low_resolution_resnet


class AutoCLR(SimCLR):
    def compute_ssl_loss(self, embeds):
        z = self.projector(embeds)
        N = z.size(0) * self.config.hardware.world_size
        batch_size = z.size(0) // 2

        mask = self._mask_correlated_samples(
            batch_size, self.config.hardware.world_size
        ).to(self.this_device)

        features = F.normalize(z, dim=1)
        sim = torch.matmul(features, features.T)

        # retrieve Kth neighbor of each row and each column
        sigma_i = torch.topk(
            sim.fill_diagonal_(0), self.config.model.autoclr_K, dim=1, largest=True
        ).values[:, -1]

        if self.config.log.wandb_project is not None:
            wandb.log(
                {
                    "train/sigma_i_mean": sigma_i.mean(),
                    "train/sigma_i_std": sigma_i.std(),
                    "epoch": self.epoch,
                    "step": self.step,
                }
            )

        sim = sim / sigma_i.unsqueeze(1)

        sim_i_j = torch.diag(sim, batch_size * self.config.hardware.world_size)
        sim_j_i = torch.diag(sim, -batch_size * self.config.hardware.world_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(
            N, 1
        )  # shape (N, 1)

        negative_samples = sim[mask].reshape(N, -1)  # shape (N, N-2)d

        logits = torch.cat(
            (positive_samples, negative_samples), dim=1
        )  # shape (N, N-1)

        logits_num = logits
        logits_denum = torch.logsumexp(logits, dim=1, keepdim=True)  # shape (N, 1)

        num_sim = (-logits_num[:, 0]).sum() / N
        num_entropy = logits_denum[:, 0].sum() / N
        return num_sim + num_entropy
