import torch
import torch.nn.functional as F
from torch import nn

from .base import SSLTrainer
from stable_ssl.config import TrainerConfig
from stable_ssl.utils import load_model, low_resolution_resnet


class SimCLR(SSLTrainer):
    def __init__(self, config: TrainerConfig):
        super().__init__(config)

    def initialize_modules(self):
        # backbone
        model, fan_in = load_model(
            name=self.config.model.backbone_model,
            n_classes=10,
            with_classifier=False,
            pretrained=False,
        )
        if (
            "resnet" in self.config.model.backbone_model
            and self.config.model.backbone_model != "resnet9"
        ):
            model = low_resolution_resnet(model)
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
        self.classifier = torch.nn.Linear(fan_in, 10)

    def forward(self, x):
        return self.backbone(x)

    def compute_ssl_loss(self, embeds):
        z = self.projector(embeds)
        return nt_xent(z, t=self.config.model.temperature)

        N = z.size(0) * self.config.hardware.world_size
        batch_size = z.size(0) // 2

        mask = self._mask_correlated_samples(
            batch_size, self.config.hardware.world_size
        ).to(self.this_device)

        features = F.normalize(z, dim=1)
        sim = torch.matmul(features, features.T) / self.config.model.temperature

        sim_i_j = torch.diag(sim, batch_size * self.config.hardware.world_size)
        sim_j_i = torch.diag(sim, -batch_size * self.config.hardware.world_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(
            N, 1
        )  # shape (N, 1)

        negative_samples = sim[mask].reshape(N, -1)  # shape (N, N-2)

        logits = torch.cat(
            (positive_samples, negative_samples), dim=1
        )  # shape (N, N-1)

        logits_num = logits
        logits_denum = torch.logsumexp(logits, dim=1, keepdim=True)  # shape (N, 1)

        num_sim = (-logits_num[:, 0]).sum() / N
        num_entropy = logits_denum[:, 0].sum() / N
        return num_sim + num_entropy

    @staticmethod
    def _mask_correlated_samples(batch_size, world_size):
        n_samples = batch_size * world_size
        N = 2 * n_samples
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        indices = torch.arange(n_samples)
        mask[indices, n_samples + indices] = 0
        mask[n_samples + indices, indices] = 0
        return mask
