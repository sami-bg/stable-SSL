"""FSDP2 wiring smoke tests.

Two layers:

1. **Structural** (run anywhere): factory returns the right class, helpers
   detect block classes, callback containers are found.
2. **Wrap+forward+backward** (CPU/gloo or CUDA): a real ``run_distributed``
   group calls ``default_parallelize_fn`` on a model and verifies the result —
   per-block units, root unit, callback containers detached and reattached
   without becoming DTensors, and one forward/backward/step is finite.
"""

import pytest
import torch
import torch.nn as nn

from stable_pretraining.tests.distributed.conftest import (
    run_distributed,
    tiny_backbone,
)


pytestmark = pytest.mark.distributed


# ---------------------------------------------------------------------------
# 1. Structural — no process group required.
# ---------------------------------------------------------------------------


def test_make_fsdp_strategy_returns_model_parallel_strategy():
    from lightning.pytorch.strategies import ModelParallelStrategy

    from stable_pretraining.utils.fsdp import make_fsdp_strategy

    strat = make_fsdp_strategy(data_parallel_size=1, tensor_parallel_size=1)
    assert isinstance(strat, ModelParallelStrategy)


def test_make_fsdp_strategy_stashes_mp_policy():
    from torch.distributed.fsdp import MixedPrecisionPolicy

    from stable_pretraining.utils.fsdp import make_fsdp_strategy

    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
    strat = make_fsdp_strategy(
        data_parallel_size=1, tensor_parallel_size=1, mp_policy=policy
    )
    assert strat._spt_mp_policy is policy


def test_detect_block_classes_finds_repeated_block():
    from stable_pretraining.utils.fsdp import _detect_block_classes

    class FooBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.b = nn.ModuleList([FooBlock(), FooBlock(), FooBlock()])

    assert _detect_block_classes(M()) == {FooBlock}


def test_detect_block_classes_ignores_singleton():
    """A single ``*Block`` instance isn't a stack and shouldn't be auto-wrapped."""
    from stable_pretraining.utils.fsdp import _detect_block_classes

    class OnceBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.b = OnceBlock()

    assert _detect_block_classes(M()) == set()


def test_find_callback_containers_locates_module_dicts():
    import stable_pretraining as spt
    from stable_pretraining.utils.fsdp import find_callback_containers

    module = spt.Module(model=tiny_backbone(), forward=lambda self, b, s: {"loss": 0})
    containers = find_callback_containers(module)
    names = sorted(
        n for n, sub in module.named_modules() if any(sub is c for c in containers)
    )
    assert names == ["callbacks_metrics", "callbacks_modules"]


# ---------------------------------------------------------------------------
# 2. Wrap + forward + backward — real distributed group.
# ---------------------------------------------------------------------------


def _smoke_wrap_and_step(rank: int, world_size: int) -> None:
    """Build a tiny module, parallelize it, run one fwd/bwd/step.

    Asserts:
    - per-block ``fully_shard`` is applied to repeated ``Block`` modules,
    - the root is sharded,
    - callback containers are reattached after the root sweep AND their
      params are NOT DTensors (the detach/reattach contract),
    - one forward+backward+optimizer step runs and produces a finite loss.
    """
    from torch.distributed.tensor import DTensor, init_device_mesh

    import stable_pretraining as spt
    from stable_pretraining.utils.fsdp import (
        default_parallelize_fn,
        find_callback_containers,
    )

    torch.manual_seed(7 + rank)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = nn.Linear(8, 8)
            self.lin2 = nn.Linear(8, 8)

        def forward(self, x):
            return self.lin2(self.lin1(x).relu())

    class TinyViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Block() for _ in range(3)])

        def forward(self, x):
            for blk in self.blocks:
                x = blk(x)
            return x

    def fwd(self, batch, stage):
        out = self.model(batch["image"])
        return {"loss": (out * out).mean(), "embedding": out}

    module = spt.Module(model=TinyViT(), forward=fwd)
    module.callbacks_modules["probe"] = nn.Linear(8, 4)

    mesh = init_device_mesh("cpu", (world_size,))
    default_parallelize_fn(module, mesh)

    # 1) Callback container is back in the tree (reattached).
    containers = find_callback_containers(module)
    assert any(c is module.callbacks_modules for c in containers), (
        "callbacks_modules was not reattached after default_parallelize_fn"
    )

    # 2) Callback params are NOT DTensors (root sweep did not claim them).
    for name, p in module.callbacks_modules.named_parameters():
        assert not isinstance(p, DTensor), (
            f"callback param '{name}' was sharded into a DTensor — "
            f"detach/reattach failed to keep it un-sharded"
        )

    # 3) Backbone Block params ARE DTensors.
    backbone_params = list(module.model.blocks[0].parameters())
    assert backbone_params, "expected the first Block to have parameters"
    assert any(isinstance(p, DTensor) for p in backbone_params), (
        "expected at least one backbone Block param to be a DTensor after wrap"
    )

    # 4) One fwd/bwd/step is finite.
    optim = torch.optim.SGD(
        [p for p in module.parameters() if p.requires_grad], lr=1e-3
    )
    out = module(batch={"image": torch.randn(4, 8)}, stage="fit")
    loss = out["loss"]
    loss.backward()
    optim.step()
    if rank == 0:
        assert torch.isfinite(loss).item(), f"loss not finite: {loss.item()}"


def test_wrap_and_step_cpu_world_size_2():
    """End-to-end: wrap, forward, backward, step on CPU/gloo, world_size=2."""
    run_distributed(_smoke_wrap_and_step, world_size=2, backend="gloo")


# ---------------------------------------------------------------------------
# 3. Root-only fallback (no repeated blocks) still trains.
# ---------------------------------------------------------------------------


def _root_only_path(rank: int, world_size: int) -> None:
    """Models with no repeated ``*Block`` fall back to root-only sharding."""
    from torch.distributed.tensor import DTensor, init_device_mesh

    import stable_pretraining as spt
    from stable_pretraining.utils.fsdp import default_parallelize_fn

    torch.manual_seed(7 + rank)

    class Plain(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, x):
            return self.lin(x)

    def fwd(self, batch, stage):
        out = self.model(batch["image"])
        return {"loss": (out * out).mean()}

    module = spt.Module(model=Plain(), forward=fwd)

    mesh = init_device_mesh("cpu", (world_size,))
    default_parallelize_fn(module, mesh)

    # With no repeated block, only the root is FSDP2-managed; the plain
    # Linear's params still become DTensors (claimed by root).
    assert isinstance(module.model.lin.weight, DTensor)

    optim = torch.optim.SGD(
        [p for p in module.parameters() if p.requires_grad], lr=1e-3
    )
    loss = module(batch={"image": torch.randn(4, 8)}, stage="fit")["loss"]
    loss.backward()
    optim.step()
    if rank == 0:
        assert torch.isfinite(loss).item()


def test_root_only_path_cpu_world_size_2():
    run_distributed(_root_only_path, world_size=2, backend="gloo")
