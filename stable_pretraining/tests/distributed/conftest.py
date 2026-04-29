"""Test infrastructure for distributed (multi-process) tests.

Provides:
- ``run_distributed(fn, world_size, backend="gloo")``: spawn N processes,
  initialize a process group, run ``fn(rank, world_size)`` on each, and
  re-raise the first failure with the originating rank's traceback.
- Module factories (``tiny_module``, ``tiny_module_with_bn``, ``tiny_vit``)
  and ``seeded_batch`` for deterministic test inputs.

All helpers default to gloo/CPU so tests run on a laptop without GPUs.
"""

from __future__ import annotations

import os
import socket
import traceback
from typing import Callable, Optional

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _patch_mps_for_fsdp_on_macos() -> None:
    """Workaround for FSDP+MPS issues on Apple Silicon.

    On Apple Silicon, ``torch.backends.mps.is_available()`` returns True, so
    FSDP's compute-device detection picks the MPS handle, which (a) lacks
    ``current_device`` on current PyTorch and (b) then expects params to be
    on ``mps:0`` even though we're running CPU/gloo tests. Force MPS to look
    unavailable inside the test worker.
    """
    import torch

    if hasattr(torch, "backends") and hasattr(torch.backends, "mps"):
        try:
            torch.backends.mps.is_available = lambda: False  # type: ignore[assignment]
            torch.backends.mps.is_built = lambda: False  # type: ignore[assignment]
        except Exception:  # noqa: BLE001
            pass
    if hasattr(torch, "mps") and not hasattr(torch.mps, "current_device"):
        torch.mps.current_device = lambda: 0  # type: ignore[attr-defined]


def _worker(
    rank: int,
    world_size: int,
    backend: str,
    master_addr: str,
    master_port: int,
    fn: Callable[[int, int], None],
    error_queue: "mp.Queue",
) -> None:
    """Single-process entry point. Initializes the process group and runs ``fn``.

    Any exception from ``fn`` is captured with a full traceback and pushed to
    ``error_queue`` so the parent can re-raise it.
    """
    _patch_mps_for_fsdp_on_macos()
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    try:
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )
    except Exception as e:  # noqa: BLE001
        error_queue.put((rank, f"init_process_group failed: {e!r}\n{traceback.format_exc()}"))
        return

    try:
        fn(rank, world_size)
    except Exception as e:  # noqa: BLE001
        error_queue.put((rank, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
    finally:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:  # noqa: BLE001
                pass


def run_distributed(
    fn: Callable[[int, int], None],
    world_size: int = 2,
    backend: str = "gloo",
    timeout: Optional[float] = 120.0,
) -> None:
    """Run ``fn(rank, world_size)`` in N spawned processes.

    Args:
        fn: Callable invoked in each worker. Must be top-level (picklable).
        world_size: Number of processes to spawn.
        backend: Backend for ``init_process_group``. Default ``gloo`` (CPU).
        timeout: Per-spawn join timeout in seconds.

    Raises:
        AssertionError or RuntimeError: re-raised from the first worker that
            failed, including the originating rank's full traceback.
    """
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed not available")

    ctx = mp.get_context("spawn")
    error_queue: mp.Queue = ctx.Queue()

    master_addr = "127.0.0.1"
    master_port = _find_free_port()

    procs = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_worker,
            args=(
                rank,
                world_size,
                backend,
                master_addr,
                master_port,
                fn,
                error_queue,
            ),
            daemon=False,
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=timeout)

    # Kill stragglers, then surface them as failures.
    stragglers = [p for p in procs if p.is_alive()]
    for p in stragglers:
        p.terminate()
        p.join(timeout=5)

    errors = []
    while not error_queue.empty():
        errors.append(error_queue.get())

    if stragglers:
        ranks = [p.pid for p in stragglers]
        raise RuntimeError(
            f"run_distributed: {len(stragglers)} worker(s) timed out (pids={ranks}). "
            f"Errors collected from finished workers: {errors}"
        )

    if errors:
        # Sort by rank for deterministic test output.
        errors.sort(key=lambda x: x[0])
        joined = "\n\n".join(f"--- rank {r} ---\n{msg}" for r, msg in errors)
        raise AssertionError(f"run_distributed: {len(errors)} worker(s) failed:\n{joined}")

    nonzero = [(p.pid, p.exitcode) for p in procs if p.exitcode not in (0, None)]
    if nonzero:
        raise RuntimeError(
            f"run_distributed: workers exited with non-zero exit codes (pid, exitcode): {nonzero}"
        )


# ---------------------------------------------------------------------------
# Module factories
# ---------------------------------------------------------------------------


class _TinyMLPBackbone(nn.Module):
    """3-layer MLP for tests. Input: ``(B, in_dim)``. Output: ``(B, out_dim)``."""

    def __init__(self, in_dim: int = 16, hidden: int = 32, out_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=False),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=False),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TinyMLPBackboneWithBN(nn.Module):
    """3-layer MLP with BatchNorm1d for buffer-EMA tests."""

    def __init__(self, in_dim: int = 16, hidden: int = 32, out_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=False),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=False),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def tiny_backbone(in_dim: int = 16, hidden: int = 32, out_dim: int = 16) -> nn.Module:
    """Plain 3-layer MLP. Cheap building block for FSDP smoke tests."""
    return _TinyMLPBackbone(in_dim=in_dim, hidden=hidden, out_dim=out_dim)


def tiny_backbone_with_bn(
    in_dim: int = 16, hidden: int = 32, out_dim: int = 16
) -> nn.Module:
    """3-layer MLP with BatchNorm. For testing buffer EMA under FSDP."""
    return _TinyMLPBackboneWithBN(in_dim=in_dim, hidden=hidden, out_dim=out_dim)


def tiny_projector(in_dim: int = 16, hidden: int = 32, out_dim: int = 16) -> nn.Module:
    """2-layer MLP projector head."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(inplace=False),
        nn.Linear(hidden, out_dim),
    )


def seeded_batch(
    seed: int = 0,
    batch_size: int = 8,
    in_dim: int = 16,
    num_classes: int = 4,
    num_views: int = 1,
) -> dict:
    """Deterministic dict-flow batch.

    For ``num_views == 1`` returns ``{"image": tensor, "label": tensor}``.
    For ``num_views > 1`` returns ``{"views": [...], "label": tensor}`` to match
    the multi-view convention used by SSL forward functions.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    if num_views == 1:
        return {
            "image": torch.randn(batch_size, in_dim, generator=g),
            "label": torch.randint(0, num_classes, (batch_size,), generator=g),
        }
    views = []
    for v in range(num_views):
        gv = torch.Generator()
        gv.manual_seed(seed * 1000 + v)
        views.append(
            {
                "image": torch.randn(batch_size, in_dim, generator=gv),
                "label": torch.randint(0, num_classes, (batch_size,), generator=gv),
            }
        )
    return {
        "views": views,
        "label": views[0]["label"],
    }
