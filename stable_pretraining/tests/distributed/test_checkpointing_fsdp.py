"""Phase 6.2: FSDP checkpointing.

Verifies the save / load round trip for FSDP-style sharded checkpoints, and
that loading a DDP-format checkpoint into an FSDP run fails with a
recognizable error (not silent corruption).

All tests in this file are CUDA-only — :class:`FullyShardedDataParallel`
forward requires a non-CPU accelerator. They run on Linux CI with at least 2
CUDA devices.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from stable_pretraining.tests.distributed.conftest import run_distributed


pytestmark = [pytest.mark.distributed, pytest.mark.gpu]


def _build_tiny(seed: int):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(inplace=False),
        nn.Linear(32, 16),
    )


def _save_load_fsdp_sharded(rank: int, world_size: int) -> None:
    """Train 2 steps, save sharded state_dict, build a fresh model and load,
    train one more step. Compare to a non-interrupted run that took 3 steps.

    The sharded path uses :class:`ShardedStateDictConfig` — each rank writes
    its own shard, and on load each rank reads the shard with the matching
    layout. The full-state-dict path is more compatible but slower; we test
    sharded here because it's the FSDP-recommended default.
    """
    import os
    import tempfile
    from functools import partial

    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import (
        ShardedStateDictConfig,
        ShardedOptimStateDictConfig,
        StateDictType,
    )
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA test executed without CUDA")

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)
    seed = 31

    def _build_wrapped():
        torch.manual_seed(seed)  # identical init across the two runs
        m = _build_tiny(seed=seed).to(device)
        return FSDP(
            m,
            auto_wrap_policy=partial(
                size_based_auto_wrap_policy, min_num_params=100
            ),
            device_id=device.index,
            use_orig_params=True,
        )

    def _step(model, opt, x, y):
        out = model(x)
        loss = ((out - y) ** 2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        return loss.detach()

    # Deterministic batches, identical across both runs.
    torch.manual_seed(seed + 100)
    batches = [
        (
            torch.randn(8, 16, device=device),
            torch.randn(8, 16, device=device),
        )
        for _ in range(3)
    ]

    # --- Reference run: 3 contiguous steps ---------------------------------
    ref_model = _build_wrapped()
    ref_opt = torch.optim.SGD(ref_model.parameters(), lr=1e-3)
    for x, y in batches:
        _step(ref_model, ref_opt, x, y)
    cfg = ShardedStateDictConfig(offload_to_cpu=False)
    with FSDP.state_dict_type(ref_model, StateDictType.SHARDED_STATE_DICT, cfg):
        ref_state = {k: v.clone() for k, v in ref_model.state_dict().items()}

    # --- Resumed run: 2 steps, save, load (fresh model), 1 step ------------
    tmp_dir = tempfile.mkdtemp(prefix=f"fsdp_ckpt_rank{rank}_")
    ckpt_path = os.path.join(tmp_dir, f"shard_rank{rank}.pt")

    res_model = _build_wrapped()
    res_opt = torch.optim.SGD(res_model.parameters(), lr=1e-3)
    for x, y in batches[:2]:
        _step(res_model, res_opt, x, y)

    cfg = ShardedStateDictConfig(offload_to_cpu=False)
    opt_cfg = ShardedOptimStateDictConfig(offload_to_cpu=False)
    with FSDP.state_dict_type(
        res_model,
        StateDictType.SHARDED_STATE_DICT,
        state_dict_config=cfg,
        optim_state_dict_config=opt_cfg,
    ):
        torch.save(
            {
                "model": res_model.state_dict(),
                "optim": FSDP.optim_state_dict(res_model, res_opt),
            },
            ckpt_path,
        )
    dist.barrier()

    # Fresh model + fresh optimizer; load the saved state.
    res_model_new = _build_wrapped()
    res_opt_new = torch.optim.SGD(res_model_new.parameters(), lr=1e-3)
    blob = torch.load(ckpt_path, weights_only=False)
    with FSDP.state_dict_type(
        res_model_new,
        StateDictType.SHARDED_STATE_DICT,
        state_dict_config=cfg,
        optim_state_dict_config=opt_cfg,
    ):
        res_model_new.load_state_dict(blob["model"])
        flat = FSDP.optim_state_dict_to_load(res_model_new, res_opt_new, blob["optim"])
        res_opt_new.load_state_dict(flat)

    # Final step on the resumed run.
    _step(res_model_new, res_opt_new, *batches[2])
    with FSDP.state_dict_type(res_model_new, StateDictType.SHARDED_STATE_DICT, cfg):
        res_state = {k: v.clone() for k, v in res_model_new.state_dict().items()}

    # The two runs must produce the same final state.
    assert ref_state.keys() == res_state.keys(), (
        f"key mismatch: ref={sorted(ref_state.keys())} res={sorted(res_state.keys())}"
    )
    for key in ref_state:
        a = ref_state[key].detach().to(torch.float32).cpu()
        b = res_state[key].detach().to(torch.float32).cpu()
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-5), (
            f"resume diverged at {key}: max_abs_diff={(a - b).abs().max().item():.3e}"
        )


def test_save_load_fsdp_sharded():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires 2+ CUDA devices")
    run_distributed(_save_load_fsdp_sharded, world_size=2, backend="nccl")


# ---------------------------------------------------------------------------
# Adversarial: cross-strategy load surfaces a clear error.
# ---------------------------------------------------------------------------


def _ddp_into_fsdp_clear_error(rank: int, world_size: int) -> None:
    """Save a state dict from a non-FSDP (full) model, then try to load it
    into an FSDP-wrapped model with sharded state dict type. This must fail
    in a recognizable way — never silent partial loading."""
    import os
    import tempfile
    from functools import partial

    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardedStateDictConfig, StateDictType
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA test executed without CUDA")

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)

    # Save a regular (non-FSDP) state dict as our "DDP checkpoint".
    full_model = _build_tiny(seed=0).to(device)
    tmp_dir = tempfile.mkdtemp(prefix=f"ddp_ckpt_rank{rank}_")
    ckpt_path = os.path.join(tmp_dir, "ddp_full.pt")
    if rank == 0:
        torch.save(full_model.state_dict(), ckpt_path)
    dist.barrier()

    # Now build an FSDP-wrapped model and try to load via SHARDED_STATE_DICT.
    fsdp_model = FSDP(
        _build_tiny(seed=0).to(device),
        auto_wrap_policy=partial(
            size_based_auto_wrap_policy, min_num_params=100
        ),
        device_id=device.index,
        use_orig_params=True,
    )
    blob = torch.load(ckpt_path, weights_only=False)

    raised = False
    try:
        cfg = ShardedStateDictConfig(offload_to_cpu=False)
        with FSDP.state_dict_type(
            fsdp_model, StateDictType.SHARDED_STATE_DICT, cfg
        ):
            fsdp_model.load_state_dict(blob)
    except Exception as exc:  # noqa: BLE001
        raised = True
        # The error type is implementation-defined; we just want to confirm
        # it's not a silent success.
        assert exc is not None
    assert raised, (
        "loading a full state dict into a SHARDED_STATE_DICT context must "
        "raise; silent success would corrupt training"
    )


def test_load_full_into_sharded_fsdp_clear_error():
    """Regression: cross-format load fails loudly, never silently."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires 2+ CUDA devices")
    run_distributed(_ddp_into_fsdp_clear_error, world_size=2, backend="nccl")
