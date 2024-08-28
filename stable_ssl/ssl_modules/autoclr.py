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

        sim_i_j = torch.diag(sim, N // 2)
        sim_j_i = torch.diag(sim, -N // 2)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0)  # shape (N)

        mask = torch.eye(N, dtype=bool).to(self.this_device)
        negative_samples = sim[~mask].reshape(N, -1)  # shape (N, N-1)

        attraction = -positive_samples.mean()
        repulsion = torch.logsumexp(negative_samples, dim=1).mean()

        return attraction + repulsion
