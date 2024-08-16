import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from .base import SSLTrainer
from stable_ssl.config import TrainerConfig


class SimCLR(SSLTrainer):
    """
    SimCLR Loss:
    When using a batch size of 2048, use LARS as optimizer
    with a base learning rate of 0.5, weight decay of 1e-6 and a temperature of 0.15.
    When using a batch size of 256, use LARS as optimizer
    with base learning rate of 1.0, weight decay of 1e-6 and a temperature of 0.15.
    """

    def __init__(self, config: TrainerConfig):
        super().__init__(config)
        self.temperature = 0.15

        self.mask = self._mask_correlated_samples(
            self.config.optim.batch_size, self.config.hardware.world_size
        ).to("cuda")
        # ).to(f"cuda:{self.config.hardware.gpu}")
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self):
        if self.training:
            output = self.model(torch.cat([self.data[0], self.data[1]], 0))
            return self.projector(output), self.classifier(output.detach())
        else:
            return self.classifier(self.model(self.data[0]))

    def compute_loss(self):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N-1) augmented examples within a minibatch as negative examples.
        """
        proj, preds = self.forward()

        z_i, z_j = torch.chunk(proj, 2, dim=0)
        batch_size = z_i.size(0)
        N = 2 * batch_size * self.world_size

        if self.world_size > 1:
            z_i = torch.cat(self.gather(z_i), dim=0)
            z_j = torch.cat(self.gather(z_j), dim=0)

        z = torch.cat((z_i, z_j), dim=0)

        features = F.normalize(z, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature

        sim_i_j = torch.diag(sim, batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        logits_num = logits
        logits_denum = torch.logsumexp(logits, dim=1, keepdim=True)
        num_sim = (-logits_num[:, 0]).sum() / N
        num_entropy = logits_denum[:, 0].sum() / N
        return num_sim + num_entropy

    @staticmethod
    def _mask_correlated_samples(batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask


# class SimCLR(Trainer):
#     def __init__(self, config):
#         super().__init__(config)

#         self.tau = 0.1
#         self.metric = "cosine"

#     def initialize_model(self, arch):
#         model, fan_in = load_without_classifier(self.config.architecture)

#         # get output shape
#         model = model.eval()
#         representation_size = model(torch.zeros((1, 3, 224, 224))).size(1)

#         self.model = model.train()

#         self.projector = torch.nn.Sequential(
#             torch.nn.Linear(representation_size, 8096, bias=False),
#             torch.nn.BatchNorm1d(8096),
#             torch.nn.ReLU(True),
#             torch.nn.Linear(8096, 8096, bias=False),
#             torch.nn.BatchNorm1d(8096),
#             torch.nn.ReLU(True),
#             torch.nn.Linear(8096, 8096, bias=False),
#         )

#         self.classifier = torch.nn.Linear(representation_size, 1000)

#     def initialize_train_loader(self):
#         dataset = torchvision.datasets.CIFAR10(
#             root="./data",
#             train=True,
#             transform=Transform(),
#             download=True,
#         )
#         # dataset = torchvision.datasets.ImageFolder(
#         #     self.config.data / "train", Transform()
#         # )
#         # sampler = torch.utils.data.distributed.DistributedSampler(dataset)
#         assert self.config.optim.batch_size % self.config.hardware.world_size == 0
#         per_device_batch_size = (
#             self.config.optim.batch_size // self.config.hardware.world_size
#         )
#         loader = torch.utils.data.DataLoader(
#             dataset,
#             batch_size=per_device_batch_size,
#             num_workers=self.config.hardware.workers,
#             pin_memory=True,
#             sampler=RandomSampler(dataset),
#         )
#         return loader

#     def initialize_val_loader(self):
#         dataset = torchvision.datasets.CIFAR10(
#             root="./data",
#             train=False,
#             download=True,
#         )
#         # dataset = torchvision.datasets.ImageFolder(
#         #     self.config.data / "eval", Transform()
#         # )
#         # sampler = torch.utils.data.distributed.DistributedSampler(dataset)
#         assert self.config.optim.batch_size % self.config.hardware.world_size == 0
#         per_device_batch_size = (
#             self.config.optim.batch_size // self.config.hardware.world_size
#         )
#         loader = torch.utils.data.DataLoader(
#             dataset,
#             batch_size=per_device_batch_size,
#             num_workers=self.config.hardware.workers,
#             pin_memory=True,
#             sampler=RandomSampler(dataset),
#         )
#         return loader

#     def forward(self):
#         if self.training:
#             output = self.model(torch.cat([self.data[0][0], self.data[0][1]], 0))
#             return self.projector(output), self.classifier(output.detach())
#         else:
#             return self.classifier(self.model(self.data[0]))

#     def compute_loss(self):
#         proj, preds = self.forward()
#         b = proj.size(0) // 2

#         sim = self.metric(z)  # (2b, 2b)
#         sim = sim / self.tau

#         torch.diagonal(sim).sub_(1e9)
#         perm = torch.cat((torch.arange(b) + b, torch.arange(b)), dim=0).to(sim.device)
#         return F.cross_entropy(sim, perm)
