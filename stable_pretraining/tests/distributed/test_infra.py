"""Distributed test infrastructure and gather utility fix.

These tests must run on a CPU-only laptop with the gloo backend.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

from stable_pretraining.tests.distributed.conftest import run_distributed
from stable_pretraining.utils.distributed import all_gather, all_reduce


pytestmark = pytest.mark.distributed


# ---------------------------------------------------------------------------
# Adversarial pre-tests: prove the helper does not silently swallow failures.
# ---------------------------------------------------------------------------


def _rank_one_raises(rank: int, world_size: int) -> None:
    if rank == 1:
        raise AssertionError("rank-1 deliberate failure")
    # rank 0 finishes cleanly


def test_run_distributed_propagates_rank1_failure():
    """If rank 1 raises, the parent must report the failure (not green)."""
    with pytest.raises(AssertionError, match="rank-1 deliberate failure"):
        run_distributed(_rank_one_raises, world_size=2)


def _rank_zero_raises(rank: int, world_size: int) -> None:
    if rank == 0:
        raise AssertionError("rank-0 deliberate failure")


def test_run_distributed_propagates_rank0_failure():
    """Symmetric: a rank-0 failure must also surface."""
    with pytest.raises(AssertionError, match="rank-0 deliberate failure"):
        run_distributed(_rank_zero_raises, world_size=2)


# ---------------------------------------------------------------------------
# Sanity tests: the helper itself works on the happy path.
# ---------------------------------------------------------------------------


def _all_ranks_succeed(rank: int, world_size: int) -> None:
    assert dist.is_initialized()
    assert dist.get_rank() == rank
    assert dist.get_world_size() == world_size


def test_run_distributed_happy_path():
    run_distributed(_all_ranks_succeed, world_size=2)


def _all_gather_returns_world_size_tensors(rank: int, world_size: int) -> None:
    t = torch.full((3,), float(rank), dtype=torch.float32)
    gathered = all_gather(t)
    assert isinstance(gathered, tuple), f"expected tuple, got {type(gathered)}"
    assert len(gathered) == world_size, (
        f"expected {world_size} tensors, got {len(gathered)}"
    )
    # Each rank's tensor has its rank value
    for r, gt in enumerate(gathered):
        assert torch.equal(gt, torch.full((3,), float(r), dtype=torch.float32)), (
            f"rank {rank}: gathered[{r}] = {gt}, expected fill({float(r)})"
        )


def test_all_gather_returns_all_ranks_tensors():
    """Regression test for the bug where all_gather discarded its result."""
    run_distributed(_all_gather_returns_world_size_tensors, world_size=2)


def _all_reduce_sums_across_ranks(rank: int, world_size: int) -> None:
    t = torch.full((2, 2), float(rank + 1), dtype=torch.float32)
    reduced = all_reduce(t)  # default op = SUM
    expected = torch.full((2, 2), float(sum(r + 1 for r in range(world_size))))
    assert torch.allclose(reduced, expected), (
        f"rank {rank}: reduced = {reduced}, expected {expected}"
    )


def test_all_reduce_sums_across_ranks():
    """Regression test: all_reduce must return the reduced tensor (functional API)."""
    run_distributed(_all_reduce_sums_across_ranks, world_size=2)


def _all_gather_single_process_returns_one_tuple(rank: int, world_size: int) -> None:
    """In a world_size=1 group, all_gather returns a 1-tuple of the input."""
    t = torch.tensor([1.0, 2.0, 3.0])
    gathered = all_gather(t)
    assert isinstance(gathered, tuple)
    assert len(gathered) == 1
    assert torch.equal(gathered[0], t)


def test_all_gather_world_size_one():
    run_distributed(_all_gather_single_process_returns_one_tuple, world_size=1)


# ---------------------------------------------------------------------------
# Sanity: the no-distributed path still works (no process group initialized).
# ---------------------------------------------------------------------------


def test_all_gather_no_dist_returns_input_tuple():
    """Without an initialized process group, all_gather is a no-op returning ``(t,)``."""
    assert not dist.is_initialized()
    t = torch.arange(4, dtype=torch.float32)
    out = all_gather(t)
    assert isinstance(out, tuple)
    assert len(out) == 1
    assert torch.equal(out[0], t)


def test_all_reduce_no_dist_returns_input():
    assert not dist.is_initialized()
    t = torch.arange(4, dtype=torch.float32)
    out = all_reduce(t)
    assert torch.equal(out, t)
