"""FSDP GPU memory and throughput regression guards.

These tests verify that FSDP delivers its principal advertised benefit —
**reduced peak memory** — when sharding optimizer state, parameters, and
gradients across ranks. Without this guard, a future change to the strategy
plumbing (or to PyTorch's defaults) could silently un-shard things.

All tests are CUDA-only and require ``world_size >= 2`` GPUs. They're
intended for nightly CI; running them in the fast suite is wasteful.

Methodology notes
-----------------

- We measure ``torch.cuda.max_memory_allocated`` per rank rather than
  ``max_memory_reserved``, since reserved memory includes caching allocator
  fragmentation that's a function of allocation pattern, not of how much
  state the strategy actually holds.
- For a fair DDP-vs-FSDP comparison we use **AdamW** as the optimizer:
  it carries two moment tensors per parameter, so the optimizer state is
  the dominant term and FSDP's optimizer-state sharding shows up clearly
  in the high-water mark. SGD without momentum would mostly hide the
  difference.
- The model is a deliberately oversized stack of ``Linear`` layers — big
  enough that optimizer state dominates activations, small enough to fit
  in test budget on a 2-GPU dev box.
"""

from __future__ import annotations

import time
from typing import Tuple

import pytest
import torch
import torch.nn as nn

from stable_pretraining.tests.distributed.conftest import run_distributed


pytestmark = [pytest.mark.distributed, pytest.mark.gpu]


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------


def _make_oversized_model(in_dim: int = 1024, hidden: int = 4096, depth: int = 4):
    """Stack of large ``Linear`` layers. ~67M params at the defaults — enough.

    that AdamW's optimizer state is clearly the dominant memory term.
    """
    layers = [nn.Linear(in_dim, hidden), nn.ReLU(inplace=False)]
    for _ in range(depth - 2):
        layers += [nn.Linear(hidden, hidden), nn.ReLU(inplace=False)]
    layers.append(nn.Linear(hidden, in_dim))
    return nn.Sequential(*layers)


def _measure_peak_memory(
    rank: int,
    wrap_kind: str,
    *,
    in_dim: int = 1024,
    hidden: int = 4096,
    depth: int = 4,
    n_steps: int = 3,
    cpu_offload: bool = False,
    activation_checkpointing: bool = False,
) -> Tuple[float, float]:
    """Run ``n_steps`` of forward+backward+step under the chosen wrap.

    Returns:
        ``(peak_alloc_bytes, total_seconds)`` measured on this rank.
    """
    from functools import partial

    from torch.distributed.fsdp import (
        CPUOffload,
        FullyShardedDataParallel as FSDP,
    )
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    from torch.nn.parallel import DistributedDataParallel as DDP

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA test executed without CUDA")

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)
    torch.manual_seed(7)

    model = _make_oversized_model(in_dim=in_dim, hidden=hidden, depth=depth).to(device)

    if wrap_kind == "ddp":
        wrapped = DDP(model, device_ids=[device.index])
    elif wrap_kind == "fsdp":
        kwargs = dict(
            auto_wrap_policy=partial(
                size_based_auto_wrap_policy, min_num_params=1_000_000
            ),
            device_id=device.index,
            use_orig_params=True,
        )
        if cpu_offload:
            kwargs["cpu_offload"] = CPUOffload(offload_params=True)
        wrapped = FSDP(model, **kwargs)
    else:
        raise ValueError(f"unknown wrap_kind={wrap_kind}")

    if activation_checkpointing and wrap_kind == "fsdp":
        # Apply activation checkpointing to all Linear layers under the FSDP unit.
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )

        apply_activation_checkpointing(
            wrapped,
            checkpoint_wrapper_fn=checkpoint_wrapper,
            check_fn=lambda m: isinstance(m, nn.Linear),
        )

    optim = torch.optim.AdamW(wrapped.parameters(), lr=1e-4)

    # Warm-up step (excluded from the measurement) so caching allocator
    # initial fragmentation doesn't pollute the high-water mark.
    x = torch.randn(8, in_dim, device=device)
    out = wrapped(x)
    loss = (out * out).sum()
    loss.backward()
    optim.step()
    optim.zero_grad()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        x = torch.randn(8, in_dim, device=device)
        out = wrapped(x)
        loss = (out * out).sum()
        loss.backward()
        optim.step()
        optim.zero_grad()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated(device)
    return peak, elapsed


# ---------------------------------------------------------------------------
# 1. FSDP optimizer-state memory < DDP.
# ---------------------------------------------------------------------------


def _optimizer_state_lower(rank: int, world_size: int) -> None:
    """FULL_SHARD with AdamW must shave a meaningful slice off DDP's peak."""
    ddp_peak, _ = _measure_peak_memory(rank, "ddp")
    fsdp_peak, _ = _measure_peak_memory(rank, "fsdp")

    # Theoretical lower bound for FULL_SHARD optimizer state alone:
    # ~1/world_size of the DDP value. Empirically, activations and other
    # allocations push this up. We use the plan's formula
    #   threshold = (0.6 + 0.4 / world_size) * DDP
    # which for world_size=2 is 0.8. FSDP must come in under that.
    threshold = (0.6 + 0.4 / world_size) * ddp_peak
    if rank == 0:
        msg = (
            f"\nrank 0: ddp_peak={ddp_peak / 1e6:.1f}MB "
            f"fsdp_peak={fsdp_peak / 1e6:.1f}MB "
            f"threshold={threshold / 1e6:.1f}MB"
        )
        assert fsdp_peak < threshold, f"FSDP did not save enough memory:{msg}"


def test_fsdp_optimizer_state_lower_than_ddp():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires 2+ CUDA devices")
    run_distributed(_optimizer_state_lower, world_size=2, backend="nccl")


# ---------------------------------------------------------------------------
# 2. Small-model throughput: FSDP overhead bounded.
# ---------------------------------------------------------------------------


def _throughput_within_2x(rank: int, world_size: int) -> None:
    """For a small model, FSDP's communication overhead dominates the wall.

    clock. We don't expect FSDP to win — we just guard against pathological
    regressions (e.g. >2x slowdown from a bad default).
    """
    # Smaller dims so communication ratio is higher; same 3 measured steps.
    _, ddp_t = _measure_peak_memory(rank, "ddp", in_dim=128, hidden=256, depth=2)
    _, fsdp_t = _measure_peak_memory(rank, "fsdp", in_dim=128, hidden=256, depth=2)

    if rank == 0:
        # 2.5x ceiling, not 2x, so we don't spuriously fail under load.
        assert fsdp_t < 2.5 * ddp_t, (
            f"FSDP overhead is excessive: ddp_t={ddp_t:.3f}s fsdp_t={fsdp_t:.3f}s"
        )


def test_fsdp_throughput_within_2_5x_of_ddp_small_model():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires 2+ CUDA devices")
    run_distributed(_throughput_within_2x, world_size=2, backend="nccl")


# ---------------------------------------------------------------------------
# 3. FSDP + activation checkpointing: further memory reduction.
# ---------------------------------------------------------------------------


def _ac_further_reduction(rank: int, world_size: int) -> None:
    """Activation checkpointing on top of FSDP must reduce peak further.

    The reduction comes from re-computing activations during backward
    instead of holding them — orthogonal to FSDP's parameter sharding.
    """
    fsdp_peak, _ = _measure_peak_memory(rank, "fsdp", activation_checkpointing=False)
    fsdp_ac_peak, _ = _measure_peak_memory(rank, "fsdp", activation_checkpointing=True)
    if rank == 0:
        # Don't demand a huge gain — activations are not necessarily the
        # dominant term in this synthetic model — just require some
        # measurable reduction.
        assert fsdp_ac_peak < fsdp_peak, (
            f"activation checkpointing did not reduce peak: "
            f"fsdp={fsdp_peak / 1e6:.1f}MB ac={fsdp_ac_peak / 1e6:.1f}MB"
        )


def test_fsdp_with_activation_checkpointing_further_reduction():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires 2+ CUDA devices")
    run_distributed(_ac_further_reduction, world_size=2, backend="nccl")


# ---------------------------------------------------------------------------
# 4. Adversarial: cpu_offload=True drops memory further.
# ---------------------------------------------------------------------------


def _cpu_offload_drops_memory(rank: int, world_size: int) -> None:
    fsdp_peak, _ = _measure_peak_memory(rank, "fsdp", cpu_offload=False)
    fsdp_off_peak, _ = _measure_peak_memory(rank, "fsdp", cpu_offload=True)
    if rank == 0:
        assert fsdp_off_peak < fsdp_peak, (
            f"cpu_offload=True did not reduce peak: "
            f"normal={fsdp_peak / 1e6:.1f}MB offload={fsdp_off_peak / 1e6:.1f}MB"
        )


def test_fsdp_cpu_offload_actually_offloads():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires 2+ CUDA devices")
    run_distributed(_cpu_offload_drops_memory, world_size=2, backend="nccl")
