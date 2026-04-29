"""Phase 3: DDP vs FSDP equivalence — the keystone correctness suite.

For each loss/method, we run one forward+backward+step under DDP, then under
FSDP, then assert the loss and post-step parameters match within tolerance.

**Methodology**

Each test spawns ``world_size=2`` workers. In every worker:

1. Seed all RNGs identically.
2. Build two identical model instances (same architecture, same init via
   identical seeding before construction).
3. Wrap one with DDP, the other with FSDP.
4. Feed an identical, deterministically-split batch to both.
5. Run forward + backward + ``optimizer.step()`` once on each.
6. Capture loss (scalar) and full post-step parameters (gathered for FSDP via
   :func:`summon_full_params`).
7. Assert losses are equal (bit-exact: same scalar all-reduce on same inputs).
8. Assert post-step full parameters match within ``atol=1e-6, rtol=1e-5``
   (fp32, deterministic mode).

For multi-step drift, we relax to ``rtol=1e-4`` to account for accumulated
floating-point reduction-order differences.

**Platform**

PyTorch FSDP requires a non-CPU accelerator at forward time
(``torch.distributed.fsdp._init_utils:387-390`` raises if absent). All tests in
this file are gated behind ``@pytest.mark.gpu`` and require **at least 2 CUDA
devices**. They are intended to run on Linux CI.

The structural correctness (e.g. that we can build the wrappers, that the
batch split logic is right) is also exercised by the smaller smoke tests in
``test_fsdp_smoke.py`` and ``test_teacher_student_fsdp.py`` which run on CPU.
"""

from __future__ import annotations

import copy
from typing import Callable, Dict, Tuple

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from stable_pretraining.tests.distributed.conftest import run_distributed


pytestmark = [pytest.mark.distributed, pytest.mark.gpu]


# ---------------------------------------------------------------------------
# Helpers shared by all equivalence tests.
# ---------------------------------------------------------------------------


def _device_for_rank(rank: int) -> torch.device:
    """Pin each rank to its own CUDA device. Required by FSDP."""
    if not torch.cuda.is_available():
        raise RuntimeError("FSDP equivalence tests require CUDA")
    torch.cuda.set_device(rank)
    return torch.device(f"cuda:{rank}")


def _seed_everything(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_tiny_model(seed: int = 0) -> nn.Module:
    """Build a small, FSDP-friendly model with multiple linear layers.

    Two-layer backbone + projector. We need at least two layers so the auto
    wrap policy can produce more than one FSDP unit.
    """
    _seed_everything(seed)
    return nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(inplace=False),
        nn.Linear(32, 32),
        nn.ReLU(inplace=False),
        nn.Linear(32, 16),
    )


def _gather_full_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return a CPU dict of full parameters.

    For DDP-wrapped models, ``state_dict`` already contains full tensors on
    every rank. For FSDP-wrapped models, we summon full params on rank 0.
    """
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if isinstance(model, FSDP):
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            sd = model.state_dict()
        return {k: v.detach().cpu() for k, v in sd.items()}
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def _strip_wrap_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip ``module.`` (DDP) and ``_fsdp_wrapped_module.`` (FSDP) prefixes.

    DDP and FSDP rewrite parameter names with their respective prefixes;
    stripping lets us compare like-for-like.
    """
    out = {}
    for k, v in state_dict.items():
        clean = k
        for prefix in ("module.", "_fsdp_wrapped_module."):
            while clean.startswith(prefix):
                clean = clean[len(prefix) :]
            clean = clean.replace("." + prefix, ".")
        out[clean] = v
    return out


def _make_supervised_loss(model: nn.Module, batch: torch.Tensor, target: torch.Tensor):
    out = model(batch)
    return F.mse_loss(out, target)


def _make_simclr_loss(model: nn.Module, batch: torch.Tensor, _ignored=None):
    """Two-view contrastive: split batch in half as the two views."""
    from stable_pretraining.losses import NTXEntLoss

    half = batch.size(0) // 2
    z_i = model(batch[:half])
    z_j = model(batch[half:])
    return NTXEntLoss(temperature=0.5)(z_i, z_j)


def _make_vicreg_loss(model: nn.Module, batch: torch.Tensor, _ignored=None):
    from stable_pretraining.losses import VICRegLoss

    half = batch.size(0) // 2
    z_i = model(batch[:half])
    z_j = model(batch[half:])
    return VICRegLoss()(z_i, z_j)


def _make_barlow_loss(model: nn.Module, batch: torch.Tensor, _ignored=None):
    from stable_pretraining.losses import BarlowTwinsLoss

    half = batch.size(0) // 2
    z_i = model(batch[:half])
    z_j = model(batch[half:])
    # feature_dim=16 matches model output dim; eager BN required for FSDP.
    return BarlowTwinsLoss(feature_dim=16)(z_i, z_j)


def _wrap_ddp(model: nn.Module, device: torch.device):
    from torch.nn.parallel import DistributedDataParallel as DDP

    return DDP(model.to(device), device_ids=[device.index])


def _wrap_fsdp(model: nn.Module, device: torch.device):
    from functools import partial

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

    policy = partial(size_based_auto_wrap_policy, min_num_params=100)
    return FSDP(
        model.to(device),
        auto_wrap_policy=policy,
        device_id=device.index,
        sharding_strategy=None,  # default FULL_SHARD
        use_orig_params=True,
    )


def _one_step_equivalence(
    rank: int,
    world_size: int,
    *,
    seed: int,
    batch_size: int,
    in_dim: int,
    loss_fn: Callable,
    needs_target: bool = False,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> None:
    """Generic one-step DDP/FSDP equivalence harness."""
    device = _device_for_rank(rank)

    # Identical batch across both runs; rank-aware split mirrors a real
    # DataLoader's DistributedSampler.
    _seed_everything(seed)
    full_batch = torch.randn(batch_size * world_size, in_dim, device=device)
    full_target = torch.randn(batch_size * world_size, in_dim, device=device)
    rank_slice = slice(rank * batch_size, (rank + 1) * batch_size)
    batch = full_batch[rank_slice]
    target = full_target[rank_slice]

    # Two identical model instances (same seed before each construction).
    model_ddp = _build_tiny_model(seed=seed)
    model_fsdp = _build_tiny_model(seed=seed)

    # Sanity: starting weights agree exactly.
    for (na, pa), (nb, pb) in zip(
        model_ddp.named_parameters(), model_fsdp.named_parameters()
    ):
        assert torch.equal(pa, pb), f"init mismatch at {na}/{nb}"

    # --- DDP run --------------------------------------------------------
    ddp = _wrap_ddp(model_ddp, device)
    opt_ddp = torch.optim.SGD(ddp.parameters(), lr=1e-3)
    loss_ddp = loss_fn(ddp, batch, target if needs_target else None)
    loss_ddp.backward()
    opt_ddp.step()
    opt_ddp.zero_grad()
    ddp_loss_value = loss_ddp.detach().cpu()
    ddp_state = _strip_wrap_prefix(_gather_full_params(ddp))

    # --- FSDP run -------------------------------------------------------
    fsdp = _wrap_fsdp(model_fsdp, device)
    opt_fsdp = torch.optim.SGD(fsdp.parameters(), lr=1e-3)
    loss_fsdp = loss_fn(fsdp, batch, target if needs_target else None)
    loss_fsdp.backward()
    opt_fsdp.step()
    opt_fsdp.zero_grad()
    fsdp_loss_value = loss_fsdp.detach().cpu()
    fsdp_state = _strip_wrap_prefix(_gather_full_params(fsdp))

    # --- Compare on rank 0 ---------------------------------------------
    # Loss should match very tightly (same scalar all-reduced from same per-rank inputs).
    assert torch.allclose(ddp_loss_value, fsdp_loss_value, atol=atol, rtol=rtol), (
        f"loss mismatch: ddp={ddp_loss_value.item()} fsdp={fsdp_loss_value.item()}"
    )

    if rank == 0:
        # Full state is only valid on rank 0 (rank0_only=True in summon).
        assert ddp_state.keys() == fsdp_state.keys(), (
            f"state_dict key mismatch:\n  ddp keys: {sorted(ddp_state.keys())}\n  "
            f"fsdp keys: {sorted(fsdp_state.keys())}"
        )
        for key in ddp_state:
            d_t = ddp_state[key]
            f_t = fsdp_state[key]
            assert d_t.shape == f_t.shape, (
                f"shape mismatch for {key}: ddp={tuple(d_t.shape)} fsdp={tuple(f_t.shape)}"
            )
            assert torch.allclose(d_t, f_t, atol=atol, rtol=rtol), (
                f"param mismatch for {key}: max_abs_diff="
                f"{(d_t - f_t).abs().max().item():.3e}"
            )


# ---------------------------------------------------------------------------
# Per-method equivalence tests.
# ---------------------------------------------------------------------------


def _supervised_step(rank, world_size):
    _one_step_equivalence(
        rank,
        world_size,
        seed=42,
        batch_size=8,
        in_dim=16,
        loss_fn=_make_supervised_loss,
        needs_target=True,
    )


def test_supervised_one_step_equivalence():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires 2+ CUDA devices")
    run_distributed(_supervised_step, world_size=2, backend="nccl")


def _simclr_step(rank, world_size):
    _one_step_equivalence(
        rank,
        world_size,
        seed=42,
        batch_size=16,  # split into two views of 8
        in_dim=16,
        loss_fn=_make_simclr_loss,
    )


def test_simclr_one_step_equivalence():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires 2+ CUDA devices")
    run_distributed(_simclr_step, world_size=2, backend="nccl")


def _vicreg_step(rank, world_size):
    _one_step_equivalence(
        rank,
        world_size,
        seed=42,
        batch_size=16,
        in_dim=16,
        loss_fn=_make_vicreg_loss,
    )


def test_vicreg_one_step_equivalence():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires 2+ CUDA devices")
    run_distributed(_vicreg_step, world_size=2, backend="nccl")


def _barlow_step(rank, world_size):
    _one_step_equivalence(
        rank,
        world_size,
        seed=42,
        batch_size=16,
        in_dim=16,
        loss_fn=_make_barlow_loss,
    )


def test_barlow_twins_one_step_equivalence():
    """Also exercises eager BN (feature_dim=16) and the all_reduce fix."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires 2+ CUDA devices")
    run_distributed(_barlow_step, world_size=2, backend="nccl")


# ---------------------------------------------------------------------------
# BYOL: requires a TeacherStudentWrapper. Compare DDP+TS vs FSDP+TS.
# ---------------------------------------------------------------------------


def _byol_step(rank, world_size):
    """BYOL-style: student forward + teacher forward, MSE between projections."""
    from stable_pretraining.backbone.utils import TeacherStudentWrapper
    from stable_pretraining.losses import BYOLLoss

    device = _device_for_rank(rank)
    seed = 7
    batch_size = 8
    in_dim = 16

    _seed_everything(seed)
    full_batch = torch.randn(batch_size * world_size, in_dim, device=device)
    rank_slice = slice(rank * batch_size, (rank + 1) * batch_size)
    batch = full_batch[rank_slice]

    def make_wrapper():
        backbone = _build_tiny_model(seed=seed)
        return TeacherStudentWrapper(
            backbone,
            warm_init=True,
            base_ema_coefficient=0.99,
            final_ema_coefficient=1.0,
        )

    # DDP run
    w_ddp = make_wrapper().to(device)
    from torch.nn.parallel import DistributedDataParallel as DDP

    student_ddp = DDP(w_ddp.student, device_ids=[device.index])
    opt_ddp = torch.optim.SGD(student_ddp.parameters(), lr=1e-3)
    z_s = student_ddp(batch)
    z_t = w_ddp.forward_teacher(batch)
    loss_ddp = BYOLLoss()(z_s, z_t)
    loss_ddp.backward()
    opt_ddp.step()
    opt_ddp.zero_grad()
    w_ddp.update_teacher()
    ddp_student = _strip_wrap_prefix(_gather_full_params(student_ddp))

    # FSDP run
    w_fsdp = make_wrapper().to(device)
    from functools import partial

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

    policy = partial(size_based_auto_wrap_policy, min_num_params=100)
    w_fsdp.fsdp_setup(
        auto_wrap_policy=policy,
        device_id=device.index,
        use_orig_params=True,
    )
    opt_fsdp = torch.optim.SGD(w_fsdp.student.parameters(), lr=1e-3)
    z_s = w_fsdp.student(batch)
    z_t = w_fsdp.forward_teacher(batch)
    loss_fsdp = BYOLLoss()(z_s, z_t)
    loss_fsdp.backward()
    opt_fsdp.step()
    opt_fsdp.zero_grad()
    w_fsdp.update_teacher()
    fsdp_student = _strip_wrap_prefix(_gather_full_params(w_fsdp.student))

    # Compare losses
    assert torch.allclose(
        loss_ddp.detach().cpu(), loss_fsdp.detach().cpu(), atol=1e-6, rtol=1e-5
    ), f"BYOL loss mismatch: ddp={loss_ddp.item()} fsdp={loss_fsdp.item()}"

    # Compare student state on rank 0
    if rank == 0:
        for key in ddp_student:
            assert key in fsdp_student, f"key {key} missing from fsdp state"
            assert torch.allclose(
                ddp_student[key], fsdp_student[key], atol=1e-6, rtol=1e-5
            ), f"BYOL student param mismatch at {key}"


def test_byol_one_step_equivalence():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires 2+ CUDA devices")
    run_distributed(_byol_step, world_size=2, backend="nccl")


# ---------------------------------------------------------------------------
# Multi-step drift: looser tolerance, accumulated FP differences.
# ---------------------------------------------------------------------------


def _multi_step_drift(rank, world_size):
    """5 supervised steps. Drift bounded by atol=5e-5, rtol=1e-4."""
    device = _device_for_rank(rank)
    seed = 11
    batch_size = 8
    in_dim = 16
    n_steps = 5

    _seed_everything(seed)
    full_batches = [
        torch.randn(batch_size * world_size, in_dim, device=device)
        for _ in range(n_steps)
    ]
    full_targets = [
        torch.randn(batch_size * world_size, in_dim, device=device)
        for _ in range(n_steps)
    ]

    def get_step(i):
        sl = slice(rank * batch_size, (rank + 1) * batch_size)
        return full_batches[i][sl], full_targets[i][sl]

    model_ddp = _build_tiny_model(seed=seed)
    model_fsdp = _build_tiny_model(seed=seed)

    ddp = _wrap_ddp(model_ddp, device)
    fsdp = _wrap_fsdp(model_fsdp, device)
    opt_ddp = torch.optim.SGD(ddp.parameters(), lr=1e-3)
    opt_fsdp = torch.optim.SGD(fsdp.parameters(), lr=1e-3)

    for i in range(n_steps):
        b, t = get_step(i)
        loss_d = F.mse_loss(ddp(b), t)
        loss_d.backward()
        opt_ddp.step()
        opt_ddp.zero_grad()

        loss_f = F.mse_loss(fsdp(b), t)
        loss_f.backward()
        opt_fsdp.step()
        opt_fsdp.zero_grad()

    ddp_state = _strip_wrap_prefix(_gather_full_params(ddp))
    fsdp_state = _strip_wrap_prefix(_gather_full_params(fsdp))

    if rank == 0:
        for key in ddp_state:
            d_t = ddp_state[key]
            f_t = fsdp_state[key]
            diff = (d_t - f_t).abs().max().item()
            assert diff < 5e-4, (
                f"drift at {key} after {n_steps} steps: max_abs_diff={diff:.3e}"
            )


def test_drift_after_5_steps_bounded():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires 2+ CUDA devices")
    run_distributed(_multi_step_drift, world_size=2, backend="nccl")


# ---------------------------------------------------------------------------
# Adversarial: world_size=1 sanity. The strategy abstraction itself should
# not introduce bias when there's only one rank.
# ---------------------------------------------------------------------------


def _world_size_one_sanity(rank, world_size):
    _one_step_equivalence(
        rank,
        world_size,
        seed=42,
        batch_size=16,
        in_dim=16,
        loss_fn=_make_supervised_loss,
        needs_target=True,
    )


def test_world_size_1_fsdp_matches_world_size_1_ddp():
    """Sanity: the strategy wrappers themselves don't change semantics."""
    if torch.cuda.device_count() < 1:
        pytest.skip("requires 1+ CUDA device")
    run_distributed(_world_size_one_sanity, world_size=1, backend="nccl")
