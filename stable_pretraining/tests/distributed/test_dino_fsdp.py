"""DINO async center update under FSDP.

:meth:`DINOv1Loss.update_center` (``losses/dino.py:215-227``) starts an
:func:`torch.distributed.all_reduce` with ``async_op=True`` for the teacher
center, then waits for completion in :meth:`apply_center_update` on the *next*
training step. The intended overlap pattern:

.. code-block:: text

    teacher_probs = dino_loss.softmax_center_teacher(teacher_logits, temp)
    loss = dino_loss(student_logits, teacher_probs)
    dino_loss.update_center(teacher_logits)   # async all_reduce starts
    loss.backward()                            # overlaps with the all_reduce
    optimizer.step()
    # next iteration: softmax_center_teacher() waits then applies the update

Under FSDP, the backward pass issues its own collectives (reduce-scatter for
gradients, all-gather for sharded params during pre-fetch). On some
backend/version combinations, concurrent collectives can deadlock or produce
incorrect results. This phase exercises the path to ensure neither happens.

Test split:

- **CPU/gloo (no FSDP)** — verify the DINO async center mechanism itself is
  correct: gathered center matches a non-async reference, the cycle of
  ``update_center`` → ``apply_center_update`` produces the EMA we expect.
- **CUDA + FSDP** — stress test for hangs and adversarial overlap with
  FSDP's own collectives. Marked ``@pytest.mark.gpu``.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from stable_pretraining.losses import DINOv1Loss
from stable_pretraining.tests.distributed.conftest import run_distributed


pytestmark = pytest.mark.distributed


# ---------------------------------------------------------------------------
# CPU/gloo: the async center mechanism itself, no FSDP.
# ---------------------------------------------------------------------------


def _async_center_matches_synchronous(rank: int, world_size: int) -> None:
    """The async path must produce the same center as a synchronous reduce.

    We construct teacher logits on each rank, run the async path
    (``update_center`` then ``apply_center_update``), then compare the
    resulting ``self.center`` against a synchronous all-reduced reference
    computed independently.
    """
    torch.manual_seed(7 + rank)
    n_views, batch, out_dim = 2, 4, 8
    teacher_logits = torch.randn(n_views, batch, out_dim)

    loss = DINOv1Loss(center_momentum=0.9)

    # Async path: update + apply.
    loss.update_center(teacher_logits)
    loss.apply_center_update()
    async_center = loss.center.clone()

    # Synchronous reference: do the same math without async_op.
    ref = torch.sum(teacher_logits.mean(1), dim=0, keepdim=True).clone()
    dist.all_reduce(ref)
    ref = ref / (n_views * world_size)
    # First call -> center is just the reference (no EMA blend yet).
    assert torch.allclose(async_center, ref, atol=1e-6, rtol=0), (
        f"rank {rank}: async={async_center} ref={ref}"
    )


def test_dino_async_center_matches_synchronous():
    run_distributed(_async_center_matches_synchronous, world_size=2)


def _async_center_ema_blend(rank: int, world_size: int) -> None:
    """Second update should blend with the existing center via EMA."""
    torch.manual_seed(11 + rank)
    n_views, batch, out_dim = 2, 4, 8
    momentum = 0.9
    loss = DINOv1Loss(center_momentum=momentum)

    # First update establishes the center at the reduced value.
    teacher1 = torch.randn(n_views, batch, out_dim)
    loss.update_center(teacher1)
    loss.apply_center_update()
    c1 = loss.center.clone()

    # Second update should blend: center = c1 * momentum + c2_reduced * (1-momentum).
    teacher2 = torch.randn(n_views, batch, out_dim)
    loss.update_center(teacher2)
    loss.apply_center_update()
    c_blended = loss.center.clone()

    # Compute the expected blend independently.
    ref2 = torch.sum(teacher2.mean(1), dim=0, keepdim=True).clone()
    dist.all_reduce(ref2)
    ref2 = ref2 / (n_views * world_size)
    expected = c1 * momentum + ref2 * (1 - momentum)
    assert torch.allclose(c_blended, expected, atol=1e-6, rtol=0), (
        f"rank {rank}: blended={c_blended} expected={expected}"
    )


def test_dino_async_center_ema_blend():
    run_distributed(_async_center_ema_blend, world_size=2)


def _no_hang_with_repeated_updates(rank: int, world_size: int) -> None:
    """20 cycles of update + apply must not deadlock or produce non-finite values.

    No FSDP here — this is a baseline confirming the async mechanism
    itself is robust under repeated invocation. The FSDP-overlap version
    is in the CUDA-only suite below.
    """
    torch.manual_seed(13 + rank)
    n_views, batch, out_dim = 2, 4, 16
    loss = DINOv1Loss(center_momentum=0.9)

    for step in range(20):
        teacher = torch.randn(n_views, batch, out_dim)
        loss.update_center(teacher)
        loss.apply_center_update()
        assert loss.center is not None
        assert torch.isfinite(loss.center).all(), (
            f"rank {rank} step {step}: center has non-finite values"
        )


def test_dino_no_hang_repeated_updates_cpu():
    run_distributed(_no_hang_with_repeated_updates, world_size=2)


def _interleaved_loss_call(rank: int, world_size: int) -> None:
    """Realistic interleave: ``softmax_center_teacher`` (which calls
    ``apply_center_update``) followed by ``update_center``. This mirrors the
    normal training loop and exercises the lazy-apply behavior."""
    torch.manual_seed(17 + rank)
    n_views, batch, out_dim = 2, 4, 8
    loss = DINOv1Loss(center_momentum=0.9)

    for step in range(10):
        teacher = torch.randn(n_views, batch, out_dim)
        # First step has no pending update; subsequent steps wait on the
        # async reduce queued by the previous step.
        teacher_probs = loss.softmax_center_teacher(
            teacher, teacher_temp=0.04, update_centers=True
        )
        assert torch.isfinite(teacher_probs).all()
        # Probabilities sum to 1 over last dim.
        sums = teacher_probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
        # Queue the next async update.
        loss.update_center(teacher)


def test_dino_interleaved_loss_call_cpu():
    run_distributed(_interleaved_loss_call, world_size=2)


# ---------------------------------------------------------------------------
# CUDA + FSDP: stress test for concurrent-collective deadlock.
# ---------------------------------------------------------------------------


def _dino_with_fsdp_no_hang(rank: int, world_size: int) -> None:
    from functools import partial

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA test executed without CUDA")

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)
    torch.manual_seed(42 + rank)

    # A backbone whose backward pass is large enough to overlap with the
    # async DINO reduce.
    n_views, batch, out_dim = 2, 8, 32
    backbone = nn.Sequential(
        nn.Linear(out_dim, 64),
        nn.ReLU(inplace=False),
        nn.Linear(64, 64),
        nn.ReLU(inplace=False),
        nn.Linear(64, out_dim),
    ).to(device)

    fsdp_backbone = FSDP(
        backbone,
        auto_wrap_policy=partial(size_based_auto_wrap_policy, min_num_params=100),
        device_id=device.index,
        use_orig_params=True,
    )
    optimizer = torch.optim.SGD(fsdp_backbone.parameters(), lr=1e-3)
    dino_loss = DINOv1Loss(temperature_student=0.1, center_momentum=0.9)

    n_steps = 20
    for step in range(n_steps):
        x = torch.randn(n_views, batch, out_dim, device=device)
        # Student/teacher logits via the same backbone (toy setup).
        student_logits = fsdp_backbone(x.view(-1, out_dim)).view(n_views, batch, -1)
        with torch.no_grad():
            teacher_logits = fsdp_backbone(x.view(-1, out_dim)).view(
                n_views, batch, -1
            )

        teacher_probs = dino_loss.softmax_center_teacher(
            teacher_logits, teacher_temp=0.04, update_centers=True
        )
        loss = dino_loss(student_logits, teacher_probs)
        # Queue async center update — overlaps with FSDP's backward collectives.
        dino_loss.update_center(teacher_logits)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        assert torch.isfinite(loss), f"step {step}: loss not finite ({loss.item()})"

    # After the loop, the center exists and is finite.
    assert dino_loss.center is not None
    assert torch.isfinite(dino_loss.center).all()


@pytest.mark.gpu
def test_dino_with_fsdp_no_hang_20_steps():
    """20 steps under FSDP+NCCL with overlapping async center updates.

    Failure modes guarded against: deadlock from concurrent collectives,
    non-finite loss/center values from out-of-order reduce results.
    """
    run_distributed(_dino_with_fsdp_no_hang, world_size=2, backend="nccl")


def _dino_center_ddp_vs_fsdp(rank: int, world_size: int) -> None:
    """Compare the DINO center buffer between DDP and FSDP after N steps.

    The center depends only on teacher logits (which we feed identically
    on both runs), the EMA momentum, and the all-reduce ordering — so DDP
    and FSDP should produce identical centers in fp32 + deterministic mode.
    """
    from functools import partial

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    from torch.nn.parallel import DistributedDataParallel as DDP

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA test executed without CUDA")

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)

    n_steps = 5
    n_views, batch, out_dim = 2, 8, 16

    def make_backbone(seed):
        torch.manual_seed(seed)
        return nn.Sequential(
            nn.Linear(out_dim, 32),
            nn.ReLU(inplace=False),
            nn.Linear(32, out_dim),
        ).to(device)

    # Pre-generate identical batches per rank for both runs.
    torch.manual_seed(99)
    batches = [
        torch.randn(n_views, batch, out_dim, device=device) for _ in range(n_steps)
    ]

    def run_once(wrap_fn):
        backbone = make_backbone(seed=0)
        wrapped = wrap_fn(backbone)
        opt = torch.optim.SGD(wrapped.parameters(), lr=1e-3)
        dino = DINOv1Loss(center_momentum=0.9)
        for x in batches:
            s = wrapped(x.view(-1, out_dim)).view(n_views, batch, -1)
            with torch.no_grad():
                t = wrapped(x.view(-1, out_dim)).view(n_views, batch, -1)
            tp = dino.softmax_center_teacher(t, teacher_temp=0.04, update_centers=True)
            l = dino(s, tp)
            dino.update_center(t)
            l.backward()
            opt.step()
            opt.zero_grad()
        # Force one final apply so the last queued update is realized.
        dino.apply_center_update()
        return dino.center.detach().cpu()

    ddp_center = run_once(lambda m: DDP(m, device_ids=[device.index]))
    fsdp_center = run_once(
        lambda m: FSDP(
            m,
            auto_wrap_policy=partial(size_based_auto_wrap_policy, min_num_params=100),
            device_id=device.index,
            use_orig_params=True,
        )
    )

    if rank == 0:
        assert torch.allclose(ddp_center, fsdp_center, atol=1e-5, rtol=1e-5), (
            f"DINO center diverged between DDP and FSDP: "
            f"max_abs_diff={(ddp_center - fsdp_center).abs().max().item():.3e}"
        )


@pytest.mark.gpu
def test_dino_center_ddp_vs_fsdp_equivalent():
    """DDP and FSDP must produce the same DINO center buffer to tolerance."""
    run_distributed(_dino_center_ddp_vs_fsdp, world_size=2, backend="nccl")


# ---------------------------------------------------------------------------
# Adversarial: explicitly stress concurrent-collective overlap.
# ---------------------------------------------------------------------------


def _adversarial_concurrent_collectives(rank: int, world_size: int) -> None:
    """Issue a DINO async reduce immediately followed by an FSDP forward
    (which triggers all-gather of sharded params). If the implementations
    serialize on the same NCCL stream, this should still complete; if they
    deadlock on the same comm group, the test will time out via
    ``run_distributed``'s join timeout."""
    from functools import partial

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA test executed without CUDA")

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)
    torch.manual_seed(31 + rank)

    out_dim = 32
    backbone = nn.Sequential(
        nn.Linear(out_dim, 128),
        nn.ReLU(inplace=False),
        nn.Linear(128, 128),
        nn.ReLU(inplace=False),
        nn.Linear(128, out_dim),
    ).to(device)
    fsdp_backbone = FSDP(
        backbone,
        auto_wrap_policy=partial(size_based_auto_wrap_policy, min_num_params=50),
        device_id=device.index,
        use_orig_params=True,
    )
    dino = DINOv1Loss(center_momentum=0.9)

    # Ten back-to-back overlap windows.
    for step in range(10):
        teacher_logits = torch.randn(2, 8, out_dim, device=device)
        # Start async reduce.
        dino.update_center(teacher_logits)
        # Immediately drive FSDP collectives on a different tensor.
        x = torch.randn(8, out_dim, device=device, requires_grad=False)
        _ = fsdp_backbone(x).sum()
        # Now wait for the async reduce; it must complete eventually.
        dino.apply_center_update()
        assert torch.isfinite(dino.center).all()


@pytest.mark.gpu
def test_dino_concurrent_collectives_stress():
    """Adversarial: deliberately overlap DINO async reduce with FSDP forward."""
    run_distributed(
        _adversarial_concurrent_collectives, world_size=2, backend="nccl"
    )
