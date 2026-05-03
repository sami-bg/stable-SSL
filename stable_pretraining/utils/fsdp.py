"""FSDP2 integration helpers — see ``docs/source/fsdp.rst`` for design notes.

Wires :func:`torch.distributed.fsdp.fully_shard` (FSDP2) into Lightning by
registering a thin :class:`ModelParallelStrategy` subclass under the
``"fsdp2"`` :class:`StrategyRegistry` name. After import, users can do
``Trainer(strategy="fsdp2")`` and the rest is automatic.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

import torch
import torch.nn as nn
from loguru import logger as logging

try:
    from lightning.pytorch.strategies import (
        ModelParallelStrategy,
        StrategyRegistry,
    )
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor import DTensor

    _FSDP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FSDP_AVAILABLE = False
    ModelParallelStrategy = object  # type: ignore[misc,assignment]
    StrategyRegistry = None  # type: ignore[assignment]
    MixedPrecisionPolicy = object  # type: ignore[misc,assignment]
    DTensor = object  # type: ignore[misc,assignment]
    fully_shard = None  # type: ignore[assignment]


__all__ = [
    "default_parallelize_fn",
    "StablePretrainingFSDP2",
    "UnsupportedModelError",
    "assert_aligned_wrapping",
    "find_callback_containers",
    "recognized_block_classes",
    "is_fsdp_strategy",
    "describe_fsdp_strategy",
]


class UnsupportedModelError(RuntimeError):
    """Raised when :func:`default_parallelize_fn` recognizes no block class.

    The default parallelize_fn ships with a curated registry of transformer
    block classes from timm, HuggingFace Transformers, and torchvision
    (:func:`recognized_block_classes`). Models outside that registry need a
    custom ``parallelize_fn`` — see ``docs/source/fsdp.rst``. The error is
    raised at strategy setup, not at training time, so a misconfiguration
    surfaces immediately rather than as silently-degraded throughput.
    """


def find_callback_containers(model: nn.Module) -> list[nn.Module]:
    """Return the ``callbacks_modules`` / ``callbacks_metrics`` containers on ``model``."""
    containers: list[nn.Module] = []
    for name, sub in model.named_modules():
        if name.endswith("callbacks_modules") or name.endswith("callbacks_metrics"):
            containers.append(sub)
    return containers


def _find_container_attachment(model: nn.Module, container: nn.Module):
    """Return ``(parent, attr_name)`` such that ``getattr(parent, attr_name) is container``."""
    for parent in model.modules():
        for attr_name, child in parent._modules.items():
            if child is container:
                return parent, attr_name
    raise RuntimeError(f"Could not locate container {container!r} in model tree.")


_RECOGNIZED_BLOCK_CLASSES: Optional[set[type[nn.Module]]] = None


def recognized_block_classes() -> set[type[nn.Module]]:
    """Lazy singleton registry of transformer / residual block classes we recognize.

    Imported via try/except from {timm, transformers, torchvision} so we
    don't hard-depend on any of them. The set is empty if none of those
    packages are installed.

    Currently registered: ``timm.models.vision_transformer.Block``,
    ``transformers.models.vit.modeling_vit.ViTLayer``,
    ``torchvision.models.resnet.{BasicBlock, Bottleneck}``. Add to this
    list when a benchmark surfaces a new architecture; until then,
    out-of-scope models route through the user-supplied ``parallelize_fn``
    escape hatch on :class:`stable_pretraining.Module`.
    """
    global _RECOGNIZED_BLOCK_CLASSES
    if _RECOGNIZED_BLOCK_CLASSES is not None:
        return _RECOGNIZED_BLOCK_CLASSES
    classes: set[type[nn.Module]] = set()
    try:
        from timm.models.vision_transformer import Block as TimmViTBlock

        classes.add(TimmViTBlock)
    except ImportError:
        pass
    try:
        from transformers.models.vit.modeling_vit import ViTLayer

        classes.add(ViTLayer)
    except ImportError:
        pass
    try:
        from torchvision.models.resnet import BasicBlock, Bottleneck

        classes.update({BasicBlock, Bottleneck})
    except ImportError:
        pass
    _RECOGNIZED_BLOCK_CLASSES = classes
    return classes


def default_parallelize_fn(
    model: nn.Module,
    device_mesh,
    *,
    block_classes: Optional[Iterable[type[nn.Module]]] = None,
    mp_policy: Optional["MixedPrecisionPolicy"] = None,
):
    r"""Apply FSDP2 sharding: per-block (auto-detected) + root.

    Decides *what* gets sharded (which modules become FSDP units): block
    classes are sharded per-instance, the root is sharded last, and
    callback containers are detached around the root sweep so they stay
    plain ``nn.Module``\ s.

    ``mp_policy`` controls *how* the sharded units cast dtypes — orthogonal
    to "what gets sharded". For standard mixed precision use
    ``Trainer(precision="bf16-mixed")``; for non-default policies (e.g.
    ``reduce_dtype`` ≠ ``param_dtype``) pass a :class:`MixedPrecisionPolicy`
    here. See ``docs/source/fsdp.rst``.
    """
    if not _FSDP_AVAILABLE:
        raise RuntimeError("FSDP is not available in this PyTorch build.")

    # 2-D mesh from ModelParallelStrategy (always 2-D, even at TP=1) → take DP submesh.
    if device_mesh.ndim > 1:
        mesh_dim_names = getattr(device_mesh, "mesh_dim_names", None)
        if mesh_dim_names and "data_parallel" in mesh_dim_names:
            shard_mesh = device_mesh["data_parallel"]
        else:
            raise RuntimeError(
                f"default_parallelize_fn received a {device_mesh.ndim}-D mesh "
                f"without a 'data_parallel' dim name. Pass a custom "
                f"``parallelize_fn`` to ``spt.Module`` for non-standard mesh layouts."
            )
    else:
        shard_mesh = device_mesh

    if block_classes is None:
        recognized = recognized_block_classes()
        present_types = {type(m) for m in model.modules()}
        block_classes = present_types & recognized
        if not block_classes:
            registered_names = sorted(c.__name__ for c in recognized) or [
                "(none — install timm / transformers / torchvision)"
            ]
            raise UnsupportedModelError(
                f"default_parallelize_fn could not find a recognized "
                f"transformer block class in {type(model).__name__}. "
                f"Recognized classes (from installed packages): "
                f"{registered_names}. For unsupported models, pass a "
                f"custom ``parallelize_fn`` to ``spt.Module`` (or a "
                f"``block_classes={{YourBlock}}`` kwarg here); see "
                f"``docs/source/fsdp.rst``."
            )
    block_classes = set(block_classes)
    logging.info(
        f"default_parallelize_fn: per-block fully_shard over "
        f"{[c.__name__ for c in block_classes]}"
    )

    shard_kwargs = {"mesh": shard_mesh}
    if mp_policy is not None:
        shard_kwargs["mp_policy"] = mp_policy

    # Per-block first, then root. Order matters: nested units must be created
    # before the parent, so the parent sees them as already-sharded children.
    # Exclude ``model`` itself: ``nn.Module.modules()`` yields ``self`` first,
    # and if ``type(model)`` happens to be in ``block_classes`` the root would
    # be sharded twice (once here, once in the explicit ``fully_shard(model)``
    # call below) — ``fully_shard`` is not idempotent.
    blocks_to_shard = [
        sub
        for sub in model.modules()
        if sub is not model and type(sub) in block_classes
    ]
    for sub in blocks_to_shard:
        fully_shard(sub, **shard_kwargs)

    # Detach callback containers around the root sweep. In the standard
    # TrainableCallback wrap chain, ``callbacks_modules`` is empty at this
    # point (callbacks register modules *after* configure_model returns), so
    # detach/reattach is a no-op. But this function may also be called
    # directly (tests, custom Module subclasses) with populated containers —
    # in that case the root ``fully_shard`` would claim the callback params
    # as DTensors and break each callback's standalone optimizer. Detaching
    # makes the function safe regardless of call context.
    detached: list[tuple[nn.Module, str, nn.Module]] = []
    for c in find_callback_containers(model):
        parent, attr_name = _find_container_attachment(model, c)
        # Direct ``_modules`` mutation (rather than ``setattr``) bypasses any
        # ``__setattr__`` hooks the parent class might bolt on, and avoids
        # leaving a ``None`` placeholder in the dict that ``fully_shard``'s
        # traversal might react to.
        del parent._modules[attr_name]
        detached.append((parent, attr_name, c))

    try:
        fully_shard(model, **shard_kwargs)
    finally:
        # Reverse so nested containers (rare) restore correctly.
        for parent, attr_name, c in reversed(detached):
            parent._modules[attr_name] = c

    logging.info(
        f"default_parallelize_fn: applied fully_shard to "
        f"{len(blocks_to_shard) + 1} module(s) (per-block units + root); "
        f"detached {len(detached)} callback container(s) around root sweep"
    )
    return model


def assert_aligned_wrapping(student: nn.Module, teacher: nn.Module) -> None:
    """Assert ``student`` and ``teacher`` have identical FSDP shard layouts.

    Required by :class:`TeacherStudentWrapper.update_teacher`'s in-place EMA via
    ``zip(teacher.parameters(), student.parameters())``. For DTensor params the
    check covers shape + dtype + ``placements`` + ``device_mesh`` (see
    ``docs/source/fsdp.rst`` for why same-shape-different-placement is a real
    silent-corruption hazard).
    """
    s_params = list(student.parameters())
    t_params = list(teacher.parameters())
    if len(s_params) != len(t_params):
        raise AssertionError(
            f"FSDP wrapping mismatch: student has {len(s_params)} parameter "
            f"tensors, teacher has {len(t_params)}."
        )
    for i, (sp, tp) in enumerate(zip(s_params, t_params)):
        if sp.shape != tp.shape:
            raise AssertionError(
                f"FSDP wrapping mismatch at parameter {i}: student shape "
                f"{tuple(sp.shape)} vs teacher shape {tuple(tp.shape)}"
            )
        if sp.dtype != tp.dtype:
            raise AssertionError(
                f"FSDP wrapping mismatch at parameter {i}: student dtype "
                f"{sp.dtype} vs teacher dtype {tp.dtype}"
            )
        if isinstance(sp, DTensor) or isinstance(tp, DTensor):
            if not (isinstance(sp, DTensor) and isinstance(tp, DTensor)):
                raise AssertionError(
                    f"FSDP wrapping mismatch at parameter {i}: one side is a "
                    f"DTensor and the other is a plain Tensor."
                )
            if sp.placements != tp.placements:
                raise AssertionError(
                    f"FSDP wrapping mismatch at parameter {i}: student "
                    f"placements {sp.placements} vs teacher placements "
                    f"{tp.placements}."
                )
            if sp.device_mesh != tp.device_mesh:
                raise AssertionError(
                    f"FSDP wrapping mismatch at parameter {i}: student and "
                    f"teacher use different device meshes."
                )

    s_bufs = list(student.buffers())
    t_bufs = list(teacher.buffers())
    if len(s_bufs) != len(t_bufs):
        raise AssertionError(
            f"FSDP wrapping mismatch: student has {len(s_bufs)} buffers, "
            f"teacher has {len(t_bufs)}"
        )
    for i, (sb, tb) in enumerate(zip(s_bufs, t_bufs)):
        if sb.shape != tb.shape:
            raise AssertionError(
                f"FSDP wrapping mismatch at buffer {i}: student shape "
                f"{tuple(sb.shape)} vs teacher shape {tuple(tb.shape)}"
            )
        if sb.dtype != tb.dtype:
            raise AssertionError(
                f"FSDP wrapping mismatch at buffer {i}: student dtype "
                f"{sb.dtype} vs teacher dtype {tb.dtype}"
            )


class StablePretrainingFSDP2(ModelParallelStrategy):  # type: ignore[misc]
    """:class:`ModelParallelStrategy` with auto-computed ``data_parallel_size``.

    Lightning's ``"auto"`` resolves ``data_parallel_size`` to ``num_nodes``
    (= 1 on single-node multi-GPU), which would fail the
    ``data_parallel_size * tensor_parallel_size == world_size`` check. This
    subclass instead reads ``LOCAL_WORLD_SIZE`` (set by ``torchrun``) or
    ``torch.cuda.device_count()`` when the user leaves it as ``"auto"``.

    Registered under the name ``"fsdp2"`` in :class:`StrategyRegistry`, so
    ``Trainer(strategy="fsdp2")`` is valid. Sharding is dispatched by
    :meth:`stable_pretraining.Module.configure_model` to the
    ``parallelize_fn`` callable passed via ``Module(parallelize_fn=...)``
    (defaults to :func:`default_parallelize_fn`).

    Optional ``mp_policy`` is stashed on the instance as ``_spt_mp_policy``;
    :func:`default_parallelize_fn` reads it via ``trainer.strategy`` and
    forwards it to ``fully_shard``. Custom ``parallelize_fn`` callables are
    free to ignore it and inject their own.
    """

    def __init__(
        self,
        *args,
        mp_policy: Optional["MixedPrecisionPolicy"] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._spt_mp_policy = mp_policy

    def setup_environment(self) -> None:
        if self._data_parallel_size == "auto":
            local_world = os.environ.get("LOCAL_WORLD_SIZE")
            if local_world is not None:
                self._data_parallel_size = int(local_world)
                source = "LOCAL_WORLD_SIZE"
            else:
                self._data_parallel_size = max(1, torch.cuda.device_count())
                source = "torch.cuda.device_count()"
            logging.info(
                f"StablePretrainingFSDP2: data_parallel_size="
                f"{self._data_parallel_size} (inferred from {source})"
            )
        super().setup_environment()


if _FSDP_AVAILABLE and StrategyRegistry is not None:
    StrategyRegistry.register(
        "fsdp2",
        StablePretrainingFSDP2,
        description=(
            "FSDP2 (fully_shard via ModelParallelStrategy) with auto "
            "data_parallel_size. Sharding is dispatched by "
            "stable_pretraining.Module.configure_model."
        ),
    )


def is_fsdp_strategy(strategy_or_trainer) -> bool:
    """Return True if the argument is (or wraps) an FSDP2 strategy."""
    if not _FSDP_AVAILABLE:
        return False
    strat = getattr(strategy_or_trainer, "strategy", strategy_or_trainer)
    return isinstance(strat, ModelParallelStrategy)


def describe_fsdp_strategy(strategy_or_trainer) -> dict:
    """Return a serializable summary of the FSDP2 strategy's relevant settings."""
    if not is_fsdp_strategy(strategy_or_trainer):
        return {"is_fsdp": False}
    strat = getattr(strategy_or_trainer, "strategy", strategy_or_trainer)
    mp_policy = getattr(strat, "_spt_mp_policy", None)
    return {
        "is_fsdp": True,
        "subclass": type(strat).__name__,
        "data_parallel_size": getattr(strat, "_data_parallel_size", None),
        "tensor_parallel_size": getattr(strat, "_tensor_parallel_size", None),
        "save_distributed_checkpoint": getattr(
            strat, "_save_distributed_checkpoint", None
        ),
        "mp_policy": type(mp_policy).__name__ if mp_policy is not None else None,
    }
