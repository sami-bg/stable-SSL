# -*- coding: utf-8 -*-
"""Training/evaluation metrics that are computed at the end of each step."""
#
# Author: @sami-bg
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
from collections import deque

import torch
import torch.distributed as dist
from typing import TYPE_CHECKING
from stable_ssl.utils import warn_once
if TYPE_CHECKING:
    from stable_ssl import BaseTrainer, JointEmbeddingTrainer


def gather_to_rank0(x: torch.Tensor):
    """Gathers a tensor to the rank0 device."""
    if (
        not (dist.is_available() and dist.is_initialized())
        or (world_size := dist.get_world_size()) == 1
    ):
        return x

    if dist.get_rank() == 0:
        output = [torch.zeros_like(x) for _ in range(world_size)]
        dist.gather(x, output, dst=0)
        return torch.cat(output, dim=0)
    else:
        return x


class Monitor:
    """Base class for metrics that are monitored at the end of each step.


    Inheritors must implement a `compute` method, that calculates the metric,
    and a `name` attribute for logging.
    """

    name: str = "monitor"

    def compute(self, x: BaseTrainer):
        """Abstract method that calculates a score given a model."""
        pass


class RankMe(Monitor):
    """RankMe (effective rank) monitor from :cite:`garrido2023rankme`."""

    name = "rankme"

    def __init__(self, limit: int = 8, epsilon: float = 1e-7):
        super().__init__()
        self.global_limit = limit

        num_devices = 1
        if dist.is_available() and dist.is_initialized():
            num_devices = dist.get_world_size()

        assert (
            self.global_limit % num_devices == 0
        ), f"RankMe {limit=} must be divisible by {num_devices=}"
        self.device_limit = self.global_limit // num_devices

        self.epsilon = epsilon
        self.bounded_queue = deque(maxlen=self.device_limit)

    @staticmethod
    def _calculate_rankme(x: torch.Tensor, epsilon: float) -> torch.Tensor:
        # NOTE torch.linalg.svd only supports torch.float32 for now
        if x.dtype != torch.float32:
            warn_once(
                f"RankMe expected tensors of type {torch.float32}, "
                f"but received {x.dtype}, will convert "
                f"{x.dtype}->{torch.float32}"
            )
            x = x.to(torch.float32)

        _u, s, _vh = torch.linalg.svd(x, full_matrices=False)
        p = (s / torch.sum(s, axis=0)) + epsilon
        entropy = -torch.sum(p * torch.log(p))
        return torch.exp(entropy)

    def compute(self, base: BaseTrainer) -> float:
        encoding: list | torch.Tensor = base.latest_representations
        if isinstance(encoding, list):
            # assume a list is of views, where each view is batch_size on the 0th dim
            # (as per JointEmbeddng)
            return [self.compute(batch) for batch in encoding][-1]

        batch_size, *_ = encoding.shape
        encoding = encoding.reshape(batch_size, -1)

        self.bounded_queue.append(encoding)

        encoding = torch.cat(list(self.bounded_queue), dim=0)
        encoding = gather_to_rank0(encoding)

        return RankMe._calculate_rankme(encoding, self.epsilon)


class LiDAR(Monitor):
    """A method for assessing representation quality of JE SSL objectives.

    As introduced in https://arxiv.org/pdf/2312.04000
    """
    name = "LiDAR"    

    def __init__(self, n: int = 1000, epsilon: float = 1e-7, delta: float = 1e-3):
        super().__init__()
        self.n = n
        self.global_limit = n
        self.epsilon = epsilon
        self.delta = delta
        
        self.num_devices = 1
        if dist.is_available() and dist.is_initialized():
            self.num_devices = dist.get_world_size()

        self.queue = None
        self.device_limit = self.global_limit // self.num_devices


    def _init_bounded_queue(self, batch_size: int) -> None:
        if self.queue:
            return

        # NOTE Dynamically create queue, rounding it's capacity to 
        # the nearest batch size.
        self.device_limit = (self.device_limit // batch_size) * batch_size
        self.global_limit = self.device_limit * self.num_devices
        self.queue = deque(maxlen=self.device_limit)

        if self.global_limit != self.n:
            warn_once(f"Received n={self.n} but rounded to {self.global_limit}. "
                       f"To avoid this, make sure n={self.n} and your batch size " 
                       f"({batch_size}) are divisible.")
        logging.info(f'Initialized LiDAR with n={self.global_limit}')
        return


    @staticmethod
    def _calculate_lidar(Sb, Sw, epsilon: float = 1e-7) -> float:
        eigvals_w, eigvecs_w = torch.linalg.eigh(Sw)  
        eigvals_w = torch.clamp(eigvals_w, min=epsilon)

        invsqrt_w = (eigvecs_w * (1.0 / torch.sqrt(eigvals_w))) @ eigvecs_w.transpose(-1, -2)
        Sigma_lidar = invsqrt_w @ Sb @ invsqrt_w

        lam, _ = torch.linalg.eigh(Sigma_lidar)
        lam = torch.clamp(lam, min=0.0)

        lam_sum = lam.sum() + epsilon
        p = lam / lam_sum

        p_log_p = p * torch.log(p + epsilon)

        return float(torch.exp(-p_log_p.sum()))

    def compute(self, base: BaseTrainer) -> float:
        if not isinstance(base, JointEmbeddingTrainer):
            raise NotImplementedError(
                f"LiDAR only implemented for JointEmbeddingTrainer "
                f"and not yet implemented for type {type(base)}"
            )
        
        base: JointEmbeddingTrainer
        
        # latest_embeddings:
        # list of [B, q, D], q: number of augmentations per image (surrogate class), B is batch-size
        self.queue.extend(base.latest_embeddings)

        # gather queue to [N, q, D] where N is the grand mean between-class
        embeddings: torch.Tensor = torch.cat(list(self.queue), dim=0)
        # NOTE Do we have to gather here? Can we not do any computations on-device first?
        embeddings = gather_to_rank0(embeddings)

        n, q, d = embeddings.shape

        class_means = embeddings.mean(dim=1)  # [n, D]
        grand_mean = class_means.mean(dim=0)  # [D]

        # between-class scatter:
        #    Sb = 1/(n-1) Σ_i (mu_i - mu)(mu_i - mu)^T
        diff_b = class_means - grand_mean  # shape [n, D]
        Sb = torch.zeros((d, d), device=embeddings.device)
        for i in range(n):
            v = diff_b[i].unsqueeze(1)  # shape [D,1]
            # [D,1] @ [1,D] => [D,D]
            Sb += v @ v.transpose(0, 1)
        Sb /= (n - 1)

        # within-class scatter:
        Sw = torch.zeros((d, d), device=embeddings.device)
        for i in range(n):
            for j in range(q):
                diff = embeddings[i, j] - class_means[i]  # [D]
                diff = diff.unsqueeze(1)                  # [D,1]
                Sw += diff @ diff.transpose(0, 1)         # [D,D]
        Sw /= (n * (q - 1))

        Sw += self.delta * torch.eye(d, device=embeddings.device)
        return self._calculate_lidar(Sb, Sw, self.epsilon)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rankme = RankMe()
    rankmes = [rankme.compute(torch.randn((8, 16, 16, 14))) for _ in range(25)]

    plt.plot(rankmes)
    plt.show()

