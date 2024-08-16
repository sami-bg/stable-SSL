import torch as ch
import random
import os
import numpy as np
import subprocess
from time import time
import logging
from typing import Tuple
import torch.distributed as dist
from torch.utils.data import Sampler
from typing import Iterable, Iterator, List
import submitit

import torch
from torch.optim import SGD, RMSprop, AdamW
from utils.optim import LARS

DEFAULT_PARAMS_OPTIMIZER = {
    "SGD": SGD([torch.tensor(0)]).defaults,
    "RMSprop": RMSprop([torch.tensor(0)]).defaults,
    "AdamW": AdamW([torch.tensor(0)]).defaults,
    "LARS": LARS([torch.tensor(0)]).defaults,
}


def augment_argparser(parser):
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--scheduler", type=str, default="LinearWarmupThreeStepsAnnealing"
    )
    parser.add_argument("--optimizer", type=str, default="AdamW")


class PositiveBatchSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        labels (Sampler or Iterable): Base sampler. Can be any iterable object
        views (int): number of same class samples in each mini-batch
        batch_size (int): Size of mini-batch.

    Example:
        >>> list(BatchSampler([0,0,0,1,1,1], view=2, batch_size=4, drop_last=False))
        [[0, 2, 5, 4]]
    """

    def __init__(
        self, labels_or_len: Iterable[int], views: int, batch_size: int
    ) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                f"batch_size should be a positive integer value, but got {batch_size=}"
            )
        if not isinstance(views, int) or isinstance(views, bool) or views <= 0:
            raise ValueError(
                f"views should be a positive integer value, but got {views=}"
            )
        self.batch_size = batch_size
        self.views = views
        self.count = 0
        if type(labels_or_len) == int:
            self.n_samples = labels_or_len * views
            self.has_labels = False
            return
        self.has_labels = True
        self.n_samples = len(self.labels)

        self.labels = np.asarray(labels)
        self.unique_labels = np.unique(labels)

        # due to the current implementation we need to make sure that
        # there are enough classes to produce a batch_size given views
        assert len(self.unique_labels) * views >= batch_size

        self.mapper = {}
        for i in self.unique_labels:
            self.mapper[i] = np.flatnonzero(self.labels == i)
            print(f"{i} -> {self.mapper[i]}")

    def __iter__(self) -> Iterator[List[int]]:
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        if not self.has_labels:
            sampler_iter = iter(np.random.permutation(self.n_samples // self.views))
            while True:
                try:
                    batch = [
                        next(sampler_iter) for _ in range(self.batch_size // self.views)
                    ]
                    batch = np.repeat(batch, self.views)
                    yield batch
                except StopIteration:
                    break
        else:
            while self.count < len(self):
                selected_classes = np.random.choice(
                    self.unique_labels,
                    size=self.batch_size // self.views,
                    replace=False,
                )
                batch = []
                for c in selected_classes:
                    batch.extend(
                        np.random.choice(
                            self.mapper[c], size=self.views, replace=False
                        ).tolist()
                    )
                self.count += 1
                yield batch
            self.count = 0

    def __len__(self) -> int:
        return self.n_samples // self.batch_size


class FullGatherLayer(ch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [ch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = ch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def setup_distributed(args):
    logging.info(f"Setting up Distributed model...")
    print("exporting PyTorch distributed environment variables")
    dist_env = submitit.JobEnvironment()
    if "SLURM_JOB_NODELIST" in os.environ:
        cmd = ["scontrol", "show", "hostnames", os.getenv("SLURM_JOB_NODELIST")]
        host_name = subprocess.check_output(cmd).decode().splitlines()[0]
        dist_url = f"tcp://{host_name}:{args.port}"
    else:
        dist_url = f"tcp://localhost:{args.port}"
    print(f"Process group:\n\t{dist_env.num_tasks} tasks")
    print(f"\tmaster: {dist_url}")
    print(f"\trank: {dist_env.global_rank}")
    print(f"\tworld size: {dist_env.num_nodes*dist_env.num_tasks}")
    print(f"\tlocal rank: {dist_env.local_rank}")
    ch.distributed.init_process_group(
        "nccl",
        init_method=dist_url,
        rank=dist_env.global_rank,
        world_size=dist_env.num_nodes * dist_env.num_tasks,
    )
    args.world_size = dist_env.num_nodes * dist_env.num_tasks
    args.gpu = dist_env.local_rank
    assert dist_env.global_rank == ch.distributed.get_rank()
    assert (dist_env.num_nodes * dist_env.num_tasks) == ch.distributed.get_world_size()
    return args


def count_SLURM_jobs(pending=True, running=True):
    if pending and running:
        request = "pending,running"
    elif pending:
        request = "pending"
    else:
        request = "running"
    pipe = subprocess.Popen(
        ["squeue", "-u", os.environ["USER"], "-h", "-t", request, "-r"],
        stdout=subprocess.PIPE,
    )
    output = subprocess.check_output(("wc", "-l"), stdin=pipe.stdout)
    pipe.wait()
    return int(output)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def seed_everything(seed, fast=True):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    ch.manual_seed(seed)
    ch.cuda.manual_seed(seed)
    ch.cuda.manual_seed_all(seed)
    if fast:
        ch.backends.cudnn.deterministic = False
        ch.backends.cudnn.benchmark = True
    else:
        ch.backends.cudnn.deterministic = True
        ch.backends.cudnn.benchmark = False


class SinkhornAttention(ch.jit.ScriptModule):
    def __init__(self, dim: int = -1, sinkhorn_iterations: int = 3) -> None:
        super().__init__()
        self.dim = dim
        self.sinkhorn_iterations = sinkhorn_iterations

    def forward(self, Q: ch.Tensor) -> ch.Tensor:
        Q = ch.softmax(Q, dim=self.dim)
        for _ in range(self.sinkhorn_iterations):
            Q = Q.div(ch.sum(Q, dim=-1, keepdim=True))
            Q = Q.div(ch.sum(Q, dim=-2, keepdim=True))
        return Q


def find_module(model: ch.nn.Module, module: ch.nn.Module):
    names = []
    values = []
    for child_name, child in model.named_modules():
        if isinstance(child, module):
            names.append(child_name)
            values.append(child)
    return names, values


def replace_module(model, replacement_mapping):
    if not isinstance(model, ch.nn.Module):
        raise ValueError("Torch.nn.Module expected as input")
    for name, module in model.named_modules():
        if name == "":
            continue
        replacement = replacement_mapping(name, module)
        module_names = name.split(".")
        # we go down the tree up to the parent
        parent = model
        for name in module_names[:-1]:
            parent = getattr(parent, name)
        setattr(parent, module_names[-1], replacement)
    return model


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


if __name__ == "__main__":
    a = ch.rand(4, 4)
    print(a)
    module = SinkhornAttention()
    ds_a = module(a)
    print(ds_a.sum(0), ds_a.sum(1))
