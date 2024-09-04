import torch
import torch.nn.functional as F
from torch import nn

from .base import SSLTrainer
from ..config import TrainerConfig
from ..utils import load_model, low_resolution_resnet


class SimCLR(SSLTrainer):
    def __init__(self, config: TrainerConfig):
        super().__init__(config)

    def initialize_modules(self):
        # backbone
        model, fan_in = load_model(
            name=self.config.model.backbone_model,
            n_classes=self.config.data.num_classes,
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
        self.classifier = torch.nn.Linear(fan_in, self.config.data.num_classes)

    def forward(self, x):
        return self.backbone(x)

    def compute_ssl_loss(self, embeds):
        z = self.projector(embeds)

        N = z.size(0) * self.config.hardware.world_size

        features = F.normalize(z, dim=1)
        sim = torch.matmul(features, features.T) / self.config.model.temperature

        sim_i_j = torch.diag(sim, N // 2)
        sim_j_i = torch.diag(sim, -N // 2)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0)  # shape (N)

        mask = torch.eye(N, dtype=bool).to(self.this_device)
        negative_samples = sim[~mask].reshape(N, -1)  # shape (N, N-1)

        attraction = -positive_samples.mean()
        repulsion = torch.logsumexp(negative_samples, dim=1).mean()

        return attraction + repulsion
