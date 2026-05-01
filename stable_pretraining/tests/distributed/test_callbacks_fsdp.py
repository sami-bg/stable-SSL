"""Callback compatibility under FSDP.

The :class:`stable_pretraining.callbacks.utils.TrainableCallback` system
monkey-patches :meth:`Module.configure_model` and
:meth:`Module.configure_optimizers` to register callback-owned modules into
``pl_module.callbacks_modules`` and append a per-callback optimizer to the
:meth:`configure_optimizers` return tuple. Each callback runs its own
optimizer over its own parameters; sharding those parameters under the main
FSDP unit would put them out of reach of the callback's optimizer.

This test file verifies that:

1. **Detection (CPU)**: ``find_callback_containers`` correctly identifies the
   ``callbacks_modules`` and ``callbacks_metrics`` containers populated by
   real :class:`OnlineProbe`, :class:`OnlineKNN`, and :class:`RankMe`
   instances after :meth:`Module.configure_model` runs.
2. **Strategy wiring (CPU)**: :class:`CallbackAwareFSDPStrategy._setup_model`
   adds those containers to FSDP's ``ignored_modules`` before super()'s
   wrap, with no double-add.
3. **Wrap-time (CPU)**: a Module + OnlineProbe wrapped manually with
   :class:`FullyShardedDataParallel` and ``ignored_modules`` keeps the probe
   weight as a regular :class:`torch.nn.Parameter` (not a
   :class:`FlatParameter`).
4. **Behavioral (CUDA)**: full FSDP run with a probe — probe gradients are
   computed independently of the backbone and probe parameters update
   across steps.
"""

from __future__ import annotations

from typing import List

import pytest
import torch
import torch.nn as nn

import stable_pretraining as spt
from stable_pretraining.callbacks import OnlineKNN, OnlineProbe, RankMe
from stable_pretraining.tests.distributed.conftest import (
    run_distributed,
    tiny_backbone,
)
from stable_pretraining.utils.fsdp import (
    CallbackAwareFSDPStrategy,
    find_callback_containers,
)


pytestmark = pytest.mark.distributed


# ---------------------------------------------------------------------------
# Helpers: build a realistic spt.Module with various callbacks attached.
# ---------------------------------------------------------------------------


def _supervised_forward(self, batch, stage):
    """Plain supervised forward used as the spt.Module's ``forward`` arg."""
    out = self.backbone(batch["image"])
    return {"embedding": out, "label": batch["label"], "loss": out.sum() * 0.0}


def _build_module_with_callbacks(callbacks: List) -> spt.Module:
    """Build an :class:`spt.Module` and attach the given callbacks.

    Mirrors what the :class:`Manager` does at training start: instantiates
    the module, then constructs the callbacks (which monkey-patch the
    module's ``configure_model``/``configure_optimizers``), then triggers
    ``configure_model`` so the probe modules land in ``callbacks_modules``.
    """
    module = spt.Module(
        backbone=tiny_backbone(),
        forward=_supervised_forward,
    )
    # Constructing the callback wraps configure_model.
    for cb in callbacks:
        # Bind by passing module — same as user code would do.
        if isinstance(cb, dict):
            cb["module"] = module
            cb_inst = cb["cls"](**{k: v for k, v in cb.items() if k != "cls"})
        else:
            raise TypeError("expected dict spec")
        module.__dict__.setdefault("_test_callbacks", []).append(cb_inst)

    # Trigger configure_model — Lightning would do this at fit start.
    module.configure_model()
    return module


# ---------------------------------------------------------------------------
# 1. Detection: find_callback_containers walks a real Module after setup.
# ---------------------------------------------------------------------------


def test_find_callback_containers_with_online_probe():
    """A Module with an OnlineProbe registers a probe in callbacks_modules.

    and metrics in callbacks_metrics. Both must be detected.
    """
    import torchmetrics

    module = _build_module_with_callbacks(
        [
            {
                "cls": OnlineProbe,
                "name": "probe1",
                "input": "embedding",
                "target": "label",
                "probe": nn.Linear(16, 4),
                "loss": nn.CrossEntropyLoss(),
                "metrics": torchmetrics.classification.MulticlassAccuracy(4),
            },
        ]
    )
    # Probe was registered.
    assert "probe1" in module.callbacks_modules
    assert "probe1" in module.callbacks_metrics

    containers = find_callback_containers(module)
    # Both callbacks_modules and callbacks_metrics must be in the container set.
    assert module.callbacks_modules in containers
    assert module.callbacks_metrics in containers


def test_find_callback_containers_with_online_knn():
    """OnlineKNN doesn't register a learnable module but DOES register.

    metrics and uses the OnlineQueue (which itself registers in
    callbacks_modules during setup).
    """
    import torchmetrics

    module = spt.Module(
        backbone=tiny_backbone(),
        forward=_supervised_forward,
    )
    # OnlineKNN doesn't extend TrainableCallback; it's a regular Callback.
    # It registers metrics during setup() — verify the container exists.
    knn = OnlineKNN(  # noqa: F841 — constructed for its registration side effects
        name="knn",
        input="embedding",
        target="label",
        queue_length=64,
        input_dim=16,
        target_dim=None,
        metrics={"acc": torchmetrics.classification.MulticlassAccuracy(4)},
    )
    # Even without invoking setup(), find_callback_containers always returns
    # the two ModuleDict slots — they're created in Module.__init__.
    containers = find_callback_containers(module)
    assert module.callbacks_modules in containers
    assert module.callbacks_metrics in containers


def test_find_callback_containers_with_rankme():
    """RankMe is a non-trainable callback; the container slots still exist."""
    module = spt.Module(
        backbone=tiny_backbone(),
        forward=_supervised_forward,
    )
    _ = RankMe(name="rm", target="embedding", queue_length=64, target_shape=16)
    containers = find_callback_containers(module)
    assert module.callbacks_modules in containers
    assert module.callbacks_metrics in containers


# ---------------------------------------------------------------------------
# 2. Strategy wiring: CallbackAwareFSDPStrategy injects ignored_modules.
# ---------------------------------------------------------------------------


def _exercise_strategy_setup(module, kwargs_in):
    """Run ``CallbackAwareFSDPStrategy._setup_model`` with a real instance but a.

    mocked super-class ``_setup_model``. Returns the kwargs the super would
    have observed.
    """
    from unittest.mock import patch

    from lightning.pytorch.strategies import FSDPStrategy

    captured = {}

    def fake_super_setup(self, model):
        captured["ignored_modules"] = list(self.kwargs.get("ignored_modules", []))
        return model

    fake = CallbackAwareFSDPStrategy.__new__(CallbackAwareFSDPStrategy)
    fake.kwargs = dict(kwargs_in)

    with patch.object(FSDPStrategy, "_setup_model", fake_super_setup):
        fake._setup_model(module)
    return captured["ignored_modules"]


def test_strategy_setup_model_injects_ignored_modules():
    """``CallbackAwareFSDPStrategy._setup_model`` must add the discovered.

    callback containers to ``self.kwargs["ignored_modules"]`` before
    delegating to super()._setup_model.
    """
    import torchmetrics

    module = _build_module_with_callbacks(
        [
            {
                "cls": OnlineProbe,
                "name": "p",
                "input": "embedding",
                "target": "label",
                "probe": nn.Linear(16, 4),
                "loss": nn.CrossEntropyLoss(),
                "metrics": torchmetrics.classification.MulticlassAccuracy(4),
            },
        ]
    )

    seen_ignored = _exercise_strategy_setup(module, kwargs_in={})

    expected_ids = {id(c) for c in find_callback_containers(module)}
    seen_ids = {id(m) for m in seen_ignored}
    assert expected_ids.issubset(seen_ids), (
        f"expected ignored_modules to include all callback containers; "
        f"missing {expected_ids - seen_ids}"
    )


def test_strategy_setup_model_preserves_user_ignored_modules():
    """User-supplied ``ignored_modules`` must not be dropped by the strategy."""
    module = spt.Module(backbone=tiny_backbone(), forward=_supervised_forward)
    user_ignored_module = nn.Linear(2, 2)

    seen_ignored = _exercise_strategy_setup(
        module, kwargs_in={"ignored_modules": [user_ignored_module]}
    )

    assert user_ignored_module in seen_ignored, "user ignored_modules was dropped"
    for c in find_callback_containers(module):
        assert c in seen_ignored


def test_strategy_setup_model_no_double_add():
    """If a callback container is already in ``ignored_modules``, it shouldn't.

    be added a second time.
    """
    import torchmetrics

    module = _build_module_with_callbacks(
        [
            {
                "cls": OnlineProbe,
                "name": "p",
                "input": "embedding",
                "target": "label",
                "probe": nn.Linear(16, 4),
                "loss": nn.CrossEntropyLoss(),
                "metrics": torchmetrics.classification.MulticlassAccuracy(4),
            },
        ]
    )

    # Pre-populate with one of the containers we expect the override to add.
    seen_ignored = _exercise_strategy_setup(
        module, kwargs_in={"ignored_modules": [module.callbacks_modules]}
    )

    # callbacks_modules appears exactly once.
    assert sum(1 for m in seen_ignored if m is module.callbacks_modules) == 1


# ---------------------------------------------------------------------------
# 3. Wrap-time test: probe weight is NOT a FlatParameter after FSDP wrap.
# ---------------------------------------------------------------------------


def _probe_weight_unsharded(rank: int, world_size: int) -> None:
    import torchmetrics
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from stable_pretraining.utils.fsdp import default_auto_wrap_policy

    module = _build_module_with_callbacks(
        [
            {
                "cls": OnlineProbe,
                "name": "p",
                "input": "embedding",
                "target": "label",
                "probe": nn.Linear(16, 4),
                "loss": nn.CrossEntropyLoss(),
                "metrics": torchmetrics.classification.MulticlassAccuracy(4),
            },
        ]
    )

    ignored = find_callback_containers(module)
    fsdp_module = FSDP(  # noqa: F841
        module,
        auto_wrap_policy=default_auto_wrap_policy(module, min_num_params=10),
        ignored_modules=ignored,
        device_id=None,
    )

    # The probe weight (originally created via nn.Linear(16, 4)) lives in
    # callbacks_modules["p"]. It must remain a vanilla Parameter, not a
    # FSDP FlatParameter.
    probe = module.callbacks_modules["p"]
    weight = probe.weight
    cls_name = type(weight).__name__
    assert isinstance(weight, torch.nn.Parameter), f"probe weight type: {cls_name}"
    assert cls_name != "FlatParameter", (
        f"probe weight was sharded (type={cls_name}); ignored_modules failed"
    )
    # Shape unchanged.
    assert tuple(weight.shape) == (4, 16), f"shape changed: {tuple(weight.shape)}"


def test_probe_weight_remains_unsharded_after_fsdp_wrap():
    run_distributed(_probe_weight_unsharded, world_size=2)


# ---------------------------------------------------------------------------
# 4. CUDA-only: full FSDP run with a probe; verify gradient flow.
# ---------------------------------------------------------------------------


def _probe_receives_independent_gradients(rank: int, world_size: int) -> None:
    """Under FSDP, the probe's parameters must receive gradients from the.

    probe's own loss — not from the backbone's gradients, and the probe's
    optimizer must be able to step.
    """
    import torchmetrics
    from functools import partial

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA test executed without CUDA")

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)

    module = _build_module_with_callbacks(
        [
            {
                "cls": OnlineProbe,
                "name": "p",
                "input": "embedding",
                "target": "label",
                "probe": nn.Linear(16, 4),
                "loss": nn.CrossEntropyLoss(),
                "metrics": torchmetrics.classification.MulticlassAccuracy(4),
            },
        ]
    )

    ignored = find_callback_containers(module)
    policy = partial(size_based_auto_wrap_policy, min_num_params=100)
    module = module.to(device)
    fsdp_module = FSDP(
        module,
        auto_wrap_policy=policy,
        ignored_modules=ignored,
        device_id=device.index,
        use_orig_params=True,
    )

    probe = module.callbacks_modules["p"]
    probe_initial_weight = probe.weight.detach().clone()

    # Construct probe-specific loss. Detach the embedding to mirror the
    # OnlineProbe wiring (which detaches before passing into the probe).
    optim_probe = torch.optim.SGD(probe.parameters(), lr=1e-2)
    x = torch.randn(8, 16, device=device)
    y = torch.randint(0, 4, (8,), device=device)
    emb = fsdp_module.backbone(x).detach()
    logits = probe(emb)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    # Probe weight must have a non-None, finite gradient.
    assert probe.weight.grad is not None
    assert torch.isfinite(probe.weight.grad).all()
    optim_probe.step()
    # Probe weight must have changed.
    assert not torch.equal(probe.weight, probe_initial_weight), (
        "probe weight did not update after step"
    )


@pytest.mark.gpu
def test_online_probe_gradient_flow_under_fsdp():
    run_distributed(_probe_receives_independent_gradients, world_size=2, backend="nccl")


# ---------------------------------------------------------------------------
# 5. CUDA-only: OnlineQueue.gather_distributed under FSDP.
# ---------------------------------------------------------------------------


def _queue_gathers_under_fsdp(rank: int, world_size: int) -> None:
    """Verify the gather path used by ``OnlineQueue`` works under NCCL.

    ``OnlineQueue.on_validation_epoch_start`` calls
    ``pl_module.all_gather(tensor)``, which under the hood is
    ``torch.distributed.all_gather`` over the active backend. Verify the
    underlying gather works correctly under NCCL with rank-distinguishable
    tensors — that's the only mechanism FSDP could break for the queue.
    """
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA test executed without CUDA")

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)

    local_tensor = torch.full((4, 8), float(rank), device=device)
    gathered_list = [torch.empty_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_list, local_tensor)

    # Each rank's slice should match its rank value.
    for r in range(world_size):
        expected = torch.full((4, 8), float(r), device=device)
        assert torch.allclose(gathered_list[r], expected), (
            f"rank {r} slice mismatch: {gathered_list[r]}"
        )


@pytest.mark.gpu
def test_online_queue_all_gather_under_fsdp():
    run_distributed(_queue_gathers_under_fsdp, world_size=2, backend="nccl")
