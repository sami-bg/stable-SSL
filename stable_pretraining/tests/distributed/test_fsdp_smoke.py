"""Phase 1 smoke tests: FSDP strategy construction and basic wrapping.

Two layers of tests:

1. **Structural** (run anywhere): ``make_fsdp_strategy`` constructs, callback
   containers are detected, ``default_auto_wrap_policy`` builds, and the
   FSDP-name-prefix stripping in :class:`Module` works as expected.

2. **Wrap+forward+backward** (CUDA-only): run inside a multi-process group.
   PyTorch's FSDP requires a non-CPU accelerator at runtime
   (see ``torch.distributed.fsdp._init_utils:387-390``), so these tests are
   gated behind ``@pytest.mark.gpu`` and skipped automatically when CUDA is
   unavailable. They are intended for Linux CI with at least 2 GPUs.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from stable_pretraining.tests.distributed.conftest import (
    run_distributed,
    seeded_batch,
    tiny_backbone,
    tiny_projector,
)
from stable_pretraining.utils.fsdp import (
    StableSSLFSDPStrategy,
    default_auto_wrap_policy,
    find_callback_containers,
    make_fsdp_strategy,
)


pytestmark = pytest.mark.distributed


# ---------------------------------------------------------------------------
# Strategy construction (no subprocess required).
# ---------------------------------------------------------------------------


def test_make_fsdp_strategy_constructs():
    strat = make_fsdp_strategy(
        auto_wrap_policy=None,
        sharding_strategy="FULL_SHARD",
        cpu_offload=False,
        state_dict_type="sharded",
    )
    assert isinstance(strat, StableSSLFSDPStrategy)


def test_make_fsdp_strategy_without_callback_exclusion():
    from lightning.pytorch.strategies import FSDPStrategy

    strat = make_fsdp_strategy(ignore_callbacks=False)
    # When ignore_callbacks=False, we get the vanilla FSDPStrategy.
    assert type(strat).__name__ == "FSDPStrategy"
    assert isinstance(strat, FSDPStrategy)


def test_default_auto_wrap_policy_with_blocks():
    """Auto-detect repeated *Block / *Layer classes."""
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    class FakeBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(8, 8)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.b1 = FakeBlock()
            self.b2 = FakeBlock()

    policy = default_auto_wrap_policy(M())
    assert isinstance(policy, ModuleWrapPolicy)


def test_default_auto_wrap_policy_falls_back_to_size():
    """No repeated blocks -> size-based fallback."""
    from functools import partial

    m = nn.Sequential(nn.Linear(4, 4))
    policy = default_auto_wrap_policy(m, min_num_params=1000)
    # functools.partial wrapping size_based_auto_wrap_policy.
    assert isinstance(policy, partial)


def test_find_callback_containers():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.callbacks_modules = nn.ModuleDict(
                {"probe": nn.Linear(8, 4)}
            )
            self.callbacks_metrics = nn.ModuleDict()
            self.backbone = nn.Linear(16, 8)

    found = find_callback_containers(M())
    assert len(found) == 2
    classes = {type(c) for c in found}
    assert classes == {nn.ModuleDict}


# ---------------------------------------------------------------------------
# Multi-process FSDP wrap + forward + backward + step.
# ---------------------------------------------------------------------------


class _SmokeModel(nn.Module):
    """Tiny model with backbone + projector. Returns a scalar loss."""

    def __init__(self):
        super().__init__()
        self.backbone = tiny_backbone()
        self.projector = tiny_projector()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.backbone(x))


def _supervised_one_step(rank: int, world_size: int) -> None:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    torch.manual_seed(42)
    model = _SmokeModel()
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=default_auto_wrap_policy(
            model, min_num_params=10
        ),
        device_id=None,  # CPU
    )

    optimizer = torch.optim.SGD(fsdp_model.parameters(), lr=1e-3)
    batch = seeded_batch(seed=rank, batch_size=4)
    out = fsdp_model(batch["image"])
    target = torch.zeros_like(out)
    loss = ((out - target) ** 2).mean()
    assert torch.isfinite(loss), f"loss is not finite: {loss}"

    loss.backward()
    # Check that gradients exist on at least some parameters
    grad_count = sum(1 for p in fsdp_model.parameters() if p.grad is not None)
    assert grad_count > 0, "no parameter received a gradient"

    optimizer.step()
    optimizer.zero_grad()


@pytest.mark.gpu
def test_fsdp_one_step_supervised():
    """A FSDP-wrapped tiny model trains one step without crashing.

    CUDA-only: PyTorch FSDP requires a non-CPU accelerator at forward time.
    """
    run_distributed(_supervised_one_step, world_size=2)


# ---------------------------------------------------------------------------
# Callbacks-modules exclusion test.
# ---------------------------------------------------------------------------


class _ModelWithCallbacks(nn.Module):
    """Mimics ``stable_pretraining.Module``'s callback container layout."""

    def __init__(self):
        super().__init__()
        self.backbone = tiny_backbone()
        self.callbacks_modules = nn.ModuleDict(
            {"probe": nn.Linear(16, 4)}
        )
        self.callbacks_metrics = nn.ModuleDict()


def _callbacks_not_flat_paramed(rank: int, world_size: int) -> None:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    torch.manual_seed(0)
    model = _ModelWithCallbacks()
    ignored = find_callback_containers(model)
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=default_auto_wrap_policy(model, min_num_params=10),
        ignored_modules=ignored,
        device_id=None,
    )

    # The probe parameter must remain a regular nn.Parameter (i.e. NOT a
    # FlatParameter owned by an FSDP unit). FSDP uses FlatParameter for
    # sharded params; ignored modules retain their original Parameter type.
    probe_weight = fsdp_model.callbacks_modules["probe"].weight
    assert isinstance(probe_weight, torch.nn.Parameter)
    # Concrete check: a sharded param has class FlatParameter
    cls_name = type(probe_weight).__name__
    assert cls_name != "FlatParameter", (
        f"probe weight was sharded by FSDP (type={cls_name}); ignored_modules "
        f"did not protect it"
    )

    # The probe param has world_size 1 numel-equivalence: same shape on all ranks.
    expected_shape = torch.Size([4, 16])
    assert probe_weight.shape == expected_shape, (
        f"rank {rank}: probe weight shape {probe_weight.shape}, expected {expected_shape}"
    )


def test_callbacks_not_sharded_under_fsdp():
    run_distributed(_callbacks_not_flat_paramed, world_size=2)


# ---------------------------------------------------------------------------
# Adversarial: a deliberately bad config produces a clear error.
# ---------------------------------------------------------------------------


def _wrap_everything_with_zero_threshold(rank: int, world_size: int) -> None:
    """Wrap with size_based threshold=0 and assert FSDP either succeeds with
    sensible behavior or raises a clear error rather than corrupting state."""
    from functools import partial

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

    torch.manual_seed(0)
    model = _SmokeModel()
    policy = partial(size_based_auto_wrap_policy, min_num_params=0)
    # This forces FSDP to wrap every recursive child. It should still work
    # (degenerate but valid), not silently corrupt training. Run a forward
    # to confirm it doesn't blow up.
    fsdp_model = FSDP(model, auto_wrap_policy=policy, device_id=None)
    out = fsdp_model(torch.randn(2, 16))
    assert torch.isfinite(out).all(), "forward produced non-finite output"


@pytest.mark.gpu
def test_extreme_wrap_policy_either_works_or_errors_clearly():
    """Adversarial: zero-size wrap threshold should not silently corrupt training.

    CUDA-only: forward pass requires a non-CPU accelerator under FSDP.
    """
    run_distributed(_wrap_everything_with_zero_threshold, world_size=2)


# ---------------------------------------------------------------------------
# Regex matching tolerates FSDP prefixes (covers _strip_fsdp_prefix).
# ---------------------------------------------------------------------------


def test_strip_fsdp_prefix_supports_user_regex():
    """Sanity check that the prefix-stripping helper enables expected regex."""
    import re

    from stable_pretraining.module import _strip_fsdp_prefix

    # User wrote a regex pattern as if FSDP didn't exist.
    pattern = re.compile(r"backbone")
    fsdp_name = "_fsdp_wrapped_module.backbone.layer1"
    assert pattern.match(_strip_fsdp_prefix(fsdp_name)) is not None
