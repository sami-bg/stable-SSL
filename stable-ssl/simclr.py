import torch
import torchvision
import torch.nn.functional as F

from torch.utils.data import DataLoader, RandomSampler

from positive_pair_sampler import Transform
from backbone.base import load_without_classifier
from trainer import Trainer


class SimCLR(Trainer):
    def __init__(self, config):
        super().__init__(config)

        self.tau = 0.1
        self.metric = "cosine"

    def initialize_model(self, arch):
        model, fan_in = load_without_classifier(self.config.architecture)

        # get output shape
        model = model.eval()
        representation_size = model(torch.zeros((1, 3, 224, 224))).size(1)

        self.model = model.train()

        self.projector = torch.nn.Sequential(
            torch.nn.Linear(representation_size, 8096, bias=False),
            torch.nn.BatchNorm1d(8096),
            torch.nn.ReLU(True),
            torch.nn.Linear(8096, 8096, bias=False),
            torch.nn.BatchNorm1d(8096),
            torch.nn.ReLU(True),
            torch.nn.Linear(8096, 8096, bias=False),
        )

        self.classifier = torch.nn.Linear(representation_size, 1000)

    def initialize_train_loader(self):
        dataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=True,
            transform=Transform(),
            download=True,
        )
        # dataset = torchvision.datasets.ImageFolder(
        #     self.config.data / "train", Transform()
        # )
        # sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        assert self.config.optim.batch_size % self.config.hardware.world_size == 0
        per_device_batch_size = (
            self.config.optim.batch_size // self.config.hardware.world_size
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=per_device_batch_size,
            num_workers=self.config.hardware.workers,
            pin_memory=True,
            sampler=RandomSampler(dataset),
        )
        return loader

    def initialize_val_loader(self):
        dataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
        )
        # dataset = torchvision.datasets.ImageFolder(
        #     self.config.data / "eval", Transform()
        # )
        # sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        assert self.config.optim.batch_size % self.config.hardware.world_size == 0
        per_device_batch_size = (
            self.config.optim.batch_size // self.config.hardware.world_size
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=per_device_batch_size,
            num_workers=self.config.hardware.workers,
            pin_memory=True,
            sampler=RandomSampler(dataset),
        )
        return loader

    def forward(self):
        if self.training:
            output = self.model(torch.cat([self.data[0][0], self.data[0][1]], 0))
            return self.projector(output), self.classifier(output.detach())
        else:
            return self.classifier(self.model(self.data[0]))

    def compute_loss(self):
        proj, preds = self.forward()
        b = proj.size(0) // 2

        sim = self.metric(z)  # (2b, 2b)
        sim = sim / self.tau

        torch.diagonal(sim).sub_(1e9)
        perm = torch.cat((torch.arange(b) + b, torch.arange(b)), dim=0).to(sim.device)
        return F.cross_entropy(sim, perm)


def get_metric(name: str):
    """
    Returns a pairwise similarity for a batch tensor: Tensor size (n, d) -> Tensor size (n, n)

    !! Note: Distances are negatives to produce a similarity.
    """

    def cosine(z: torch.Tensor):
        "Cosine similarity"
        z = F.normalize(z, dim=-1)
        return z @ z.T

    def euclidean(z: torch.Tensor):
        "Negative euclidean similarity"
        return -torch.cdist(z, z, p=2.0)

    def sqeuclidean(z: torch.Tensor):
        "Squared euclidean distance using matrix multiply"
        norm = z.pow(2).sum(-1)
        return -(norm[None, :] + norm[:, None] - 2 * (z @ z.T))

    return {
        "cosine": cosine,
        "euclidean": euclidean,
        "sqeuclidean": sqeuclidean,
    }[name]
