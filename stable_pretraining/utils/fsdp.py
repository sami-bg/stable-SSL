"""FSDP2 integration helpers for stable-pretraining.

Wires :func:`torch.distributed.fsdp.fully_shard` (FSDP2) into Lightning via
:class:`~lightning.pytorch.strategies.ModelParallelStrategy`. Replaces the
older FSDP1 (``FullyShardedDataParallel`` / Lightning's ``FSDPStrategy``)
integration that lived here previously — see the FSDP-incompatibility
discussion in the FSDP docs for the multi-forward-per-step rationale.

Provides:
- :func:`default_parallelize_fn`: the callable Lightning's
  :class:`ModelParallelStrategy` invokes during model setup. It detects
  transformer / residual block classes, applies ``fully_shard`` per block
  and once at the root, and detaches the ``callbacks_modules`` /
  ``callbacks_metrics`` containers around the root call so the root sweep
  doesn't claim them as DTensors.
- :func:`make_fsdp_strategy`: a Hydra-friendly factory returning a
  :class:`ModelParallelStrategy` configured with sensible defaults.
- :func:`find_callback_containers`: walks the module tree and returns the
  ``ModuleDict`` containers used by ``stable_pretraining.Module`` to host
  callback-owned modules and metrics.
- :func:`is_fsdp_strategy` / :func:`describe_fsdp_strategy`: detection +
  introspection helpers used by ``Manager`` and tests.

Why FSDP2 over FSDP1
--------------------
FSDP1 implements per-unit sharding via a flat training-state machine and
autograd hooks attached at wrap time. Methods that perform **multiple
forward passes per training step** through the same wrapped unit (LeJEPA,
DINO, IJEPA, BYOL with multi-view) confuse this state machine: the post-
backward hook for forward-1 fires before the pre-backward hook for
forward-2 has transitioned state, and FSDP1 asserts
``Expects BACKWARD_PRE or BACKWARD_POST state but got
HandleTrainingState.FORWARD``. FSDP2 (``fully_shard``) drops the flat
state machine and uses standard PyTorch hooks per call, so multi-forward
is supported by design. This is the principal reason for the swap.
"""

from __future__ import annotations

import os
from typing import Iterable, Literal, Optional, Union

import torch
import torch.nn as nn
from loguru import logger as logging

try:
    from lightning.pytorch.strategies import ModelParallelStrategy
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor import DTensor

    _FSDP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FSDP_AVAILABLE = False
    ModelParallelStrategy = object  # type: ignore[misc,assignment]
    MixedPrecisionPolicy = object  # type: ignore[misc,assignment]
    DTensor = object  # type: ignore[misc,assignment]
    fully_shard = None  # type: ignore[assignment]


__all__ = [
    "default_parallelize_fn",
    "make_fsdp_strategy",
    "find_callback_containers",
    "assert_aligned_wrapping",
    "is_fsdp_strategy",
    "describe_fsdp_strategy",
]


def is_fsdp_strategy(strategy_or_trainer) -> bool:
    """Return True if the argument is (or wraps) Lightning's FSDP2 strategy.

    Accepts either a Lightning :class:`Strategy` instance or a Lightning
    :class:`Trainer` (in which case ``trainer.strategy`` is inspected).

    Returns False (without raising) if FSDP is not importable, so callers
    can use this in conditional code without guarding the import themselves.
    """
    if not _FSDP_AVAILABLE:
        return False
    strat = getattr(strategy_or_trainer, "strategy", strategy_or_trainer)
    return isinstance(strat, ModelParallelStrategy)


def describe_fsdp_strategy(strategy_or_trainer) -> dict:
    """Return a serializable summary of the FSDP strategy's relevant settings.

    Useful for logging and for tests that want to assert on the strategy
    configuration without poking at private attributes from many places.
    """
    if not is_fsdp_strategy(strategy_or_trainer):
        return {"is_fsdp": False}
    strat = getattr(strategy_or_trainer, "strategy", strategy_or_trainer)
    mp_policy = getattr(strat, "_spt_mp_policy", None) or getattr(
        strat, "mp_policy", None
    )
    # Lightning's ModelParallelStrategy stores these as private attributes;
    # try the underscored name first, fall back to the public name in case
    # Lightning ever flips the convention.
    return {
        "is_fsdp": True,
        "subclass": type(strat).__name__,
        "data_parallel_size": getattr(strat, "_data_parallel_size", None)
        or getattr(strat, "data_parallel_size", None),
        "tensor_parallel_size": getattr(strat, "_tensor_parallel_size", None)
        or getattr(strat, "tensor_parallel_size", None),
        "save_distributed_checkpoint": getattr(
            strat, "_save_distributed_checkpoint", None
        )
        if getattr(strat, "_save_distributed_checkpoint", None) is not None
        else getattr(strat, "save_distributed_checkpoint", None),
        "mp_policy": type(mp_policy).__name__ if mp_policy is not None else None,
    }


def assert_aligned_wrapping(student: nn.Module, teacher: nn.Module) -> None:
    """Assert that ``student`` and ``teacher`` have identical FSDP shard layouts.

    Required by :class:`TeacherStudentWrapper.update_teacher`, which performs
    in-place EMA via ``zip(teacher.parameters(), student.parameters())``. If
    the two modules are wrapped with mismatched FSDP placements, ``zip`` will
    silently pair shards that do not correspond — corruption, not a crash.

    Under FSDP2 each parameter is a ``DTensor``. For two DTensors to be
    safe for elementwise EMA, they must agree on:

    - ``shape`` and ``dtype`` (local-shard view; cheap first cut).
    - ``placements`` — same shard/replicate spec on each mesh dim. Two
      DTensors with equal local shapes can still have different placements
      (``Shard(0)`` vs ``Shard(1)`` on equal-sized dims; or in degenerate
      cases ``Shard`` vs ``Replicate``); without this check they'd silently
      pair to non-corresponding shards.
    - ``device_mesh`` — same physical mesh.

    Raises:
        AssertionError: with a description of the first mismatch found.
    """
    s_params = list(student.parameters())
    t_params = list(teacher.parameters())
    if len(s_params) != len(t_params):
        raise AssertionError(
            f"FSDP wrapping mismatch: student has {len(s_params)} parameter "
            f"tensors, teacher has {len(t_params)}. Both must be sharded with "
            f"the same parallelize_fn on the same device mesh."
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
        # DTensor-aware checks for FSDP2-managed params.
        if isinstance(sp, DTensor) or isinstance(tp, DTensor):
            if not (isinstance(sp, DTensor) and isinstance(tp, DTensor)):
                raise AssertionError(
                    f"FSDP wrapping mismatch at parameter {i}: one side is a "
                    f"DTensor and the other is a plain Tensor (student="
                    f"{type(sp).__name__}, teacher={type(tp).__name__}). "
                    f"Both modules must be sharded by the same parallelize_fn."
                )
            if sp.placements != tp.placements:
                raise AssertionError(
                    f"FSDP wrapping mismatch at parameter {i}: student "
                    f"placements {sp.placements} vs teacher placements "
                    f"{tp.placements}. Same local shape with different "
                    f"placements would silently pair non-corresponding shards "
                    f"during EMA."
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


def find_callback_containers(model: nn.Module) -> list[nn.Module]:
    """Return the list of ``callbacks_modules`` / ``callbacks_metrics`` containers.

    These are :class:`torch.nn.ModuleDict` instances on
    :class:`stable_pretraining.Module` that hold callback-owned modules and
    callback-owned metrics. They must be excluded from FSDP sharding because
    each callback (e.g. :class:`OnlineProbe`) manages its own optimizer over
    these parameters.
    """
    containers: list[nn.Module] = []
    for name, sub in model.named_modules():
        if name.endswith("callbacks_modules") or name.endswith("callbacks_metrics"):
            containers.append(sub)
    return containers


def _find_container_attachment(model: nn.Module, container: nn.Module):
    """Return ``(parent, attr_name)`` such that ``getattr(parent, attr_name) is container``.

    Used by :func:`default_parallelize_fn` to detach callback containers
    from the module tree around the root ``fully_shard`` call.
    """
    for parent in model.modules():
        for attr_name, child in parent._modules.items():
            if child is container:
                return parent, attr_name
    raise RuntimeError(
        f"Could not locate container {container!r} in model tree — "
        f"this is a bug in find_callback_containers / detach logic."
    )


def _detect_block_classes(model: nn.Module) -> set[type[nn.Module]]:
    """Best-effort detection of transformer / residual block classes in ``model``.

    Heuristic: any class whose name ends with ``Block`` or ``Layer`` and which
    is repeated in the model is a candidate for per-block FSDP wrapping. This
    matches conventions in timm ViT (``Block``), HuggingFace ViT
    (``ViTLayer``), torchvision ResNet (``BasicBlock`` / ``Bottleneck``), and
    similar libraries.
    """
    counts: dict[type[nn.Module], int] = {}
    for sub in model.modules():
        cls = type(sub)
        name = cls.__name__
        if name.endswith("Block") or name.endswith("Layer"):
            counts[cls] = counts.get(cls, 0) + 1
    return {cls for cls, n in counts.items() if n >= 2}


def default_parallelize_fn(
    model: nn.Module,
    device_mesh,
    *,
    block_classes: Optional[Iterable[type[nn.Module]]] = None,
    mp_policy: Optional["MixedPrecisionPolicy"] = None,
    ignore_callbacks: bool = True,
):
    """Apply FSDP2 sharding to ``model``, called by ``ModelParallelStrategy``.

    Walks the module tree and applies :func:`fully_shard` to:

    1. Each module whose class is in ``block_classes`` (auto-detected if
       ``None``: any ``*Block`` or ``*Layer`` class repeated in the model).
    2. The root model itself (composition order matters: per-block units
       must be created *before* the root unit so the root sees them as
       already-sharded children).

    Modules under ``callbacks_modules`` or ``callbacks_metrics`` are
    deliberately excluded from sharding when ``ignore_callbacks=True``:
    they belong to per-callback optimizers and sharding them would put
    their parameters out of reach of those optimizers.

    The exclusion uses **detach-around-root**: the containers are popped
    out of the module tree before ``fully_shard(model)`` and reattached
    after. Skipping them in the per-block loop alone is *not* sufficient —
    the root ``fully_shard`` call sweeps every parameter reachable from
    ``model`` that isn't already in a nested FSDP unit, and would happily
    convert callback container params to DTensors otherwise. The
    detach/reattach pattern is the standard FSDP2 way to exclude submodules
    from the root sweep without giving them their own FSDP unit.

    The ``device_mesh`` may be either a 1-D mesh (pure FSDP2) or a 2-D mesh
    when Lightning's ``ModelParallelStrategy`` is configured with
    ``tensor_parallel_size > 1``. In the 2-D case we slice out the
    ``data_parallel`` axis so ``fully_shard`` sees exactly the DP submesh
    it expects; without this slice ``fully_shard`` would interpret the TP
    dim as an HSDP replicate axis (silently wrong, no error).

    Args:
        model: The LightningModule to shard.
        device_mesh: The device mesh provided by Lightning's
            ``ModelParallelStrategy`` (1-D or 2-D).
        block_classes: Explicit list of block classes to shard
            individually. ``None`` (default) auto-detects via
            :func:`_detect_block_classes`. When detection finds nothing
            (unconventional class names, single-block models), a warning
            is logged and we fall back to root-only sharding.
        mp_policy: Optional :class:`MixedPrecisionPolicy` for FSDP2.
        ignore_callbacks: When ``True`` (default), detach
            ``callbacks_modules`` / ``callbacks_metrics`` containers
            around the root ``fully_shard`` call so their parameters
            remain regular ``nn.Parameter`` (un-sharded).

    Returns:
        The sharded model. ``fully_shard`` mutates the model in place
        (each managed parameter becomes a ``DTensor``); the returned
        reference is the same object.
    """
    if not _FSDP_AVAILABLE:
        raise RuntimeError("FSDP is not available in this PyTorch build.")

    # Slice out the data-parallel submesh when Lightning hands us a 2-D
    # mesh (TP > 1). For pure FSDP2 (TP=1) the mesh is already 1-D.
    if hasattr(device_mesh, "ndim") and device_mesh.ndim > 1:
        mesh_dim_names = getattr(device_mesh, "mesh_dim_names", None)
        if mesh_dim_names and "data_parallel" in mesh_dim_names:
            shard_mesh = device_mesh["data_parallel"]
            logging.info(
                f"default_parallelize_fn: 2-D mesh detected, sharding on "
                f"data_parallel submesh of size {shard_mesh.size()}"
            )
        else:
            raise RuntimeError(
                f"default_parallelize_fn received a {device_mesh.ndim}-D mesh "
                f"without a 'data_parallel' dim name. Expected either a 1-D "
                f"mesh or a mesh with a 'data_parallel' axis. Override "
                f"``Module.configure_model`` for custom mesh layouts."
            )
    else:
        shard_mesh = device_mesh

    # Detect blocks. Empty result -> root-only sharding (correct, but slow);
    # log a warning so users with unusual class names see why.
    if block_classes is None:
        block_classes = _detect_block_classes(model)
    block_classes = set(block_classes)
    if block_classes:
        logging.info(
            f"default_parallelize_fn: per-block fully_shard over "
            f"{[c.__name__ for c in block_classes]}"
        )
    else:
        logging.warning(
            "default_parallelize_fn: no repeated *Block/*Layer classes "
            "detected in the model — falling back to root-only sharding "
            "(no per-block compute/comm overlap). For custom architectures "
            "pass ``block_classes`` explicitly via a Module subclass that "
            "overrides ``apply_fsdp2_sharding_if_needed``."
        )

    shard_kwargs = {"mesh": shard_mesh}
    if mp_policy is not None:
        shard_kwargs["mp_policy"] = mp_policy

    # Per-block sharding. We collect block ids first, then apply
    # ``fully_shard`` so iteration over ``model.modules()`` isn't perturbed
    # mid-walk by the in-place wrapping.
    blocks_to_shard = [sub for sub in model.modules() if type(sub) in block_classes]
    for sub in blocks_to_shard:
        fully_shard(sub, **shard_kwargs)
    sharded_count = len(blocks_to_shard)

    # Detach callback containers before the root sweep so the root unit
    # doesn't claim their parameters.
    detached: list[tuple[nn.Module, str, nn.Module]] = []
    if ignore_callbacks:
        containers = find_callback_containers(model)
        for c in containers:
            parent, attr_name = _find_container_attachment(model, c)
            del parent._modules[attr_name]
            detached.append((parent, attr_name, c))
        if detached:
            logging.info(
                f"default_parallelize_fn: detached {len(detached)} callback "
                f"container(s) around root fully_shard call"
            )

    try:
        fully_shard(model, **shard_kwargs)
        sharded_count += 1
    finally:
        # Reattach in reverse so nested containers (rare) restore correctly.
        for parent, attr_name, c in reversed(detached):
            parent._modules[attr_name] = c

    logging.info(
        f"default_parallelize_fn: applied fully_shard to {sharded_count} "
        f"module(s) (per-block units + root)"
    )
    return model


def make_fsdp_strategy(
    *,
    save_distributed_checkpoint: bool = True,
    data_parallel_size: Optional[Union[Literal["auto"], int]] = None,
    tensor_parallel_size: Union[Literal["auto"], int] = 1,
    mp_policy: Optional["MixedPrecisionPolicy"] = None,
    **kwargs,
):
    """Hydra-friendly factory for Lightning's FSDP2-backed :class:`ModelParallelStrategy`.

    The actual FSDP2 sharding is applied via :func:`default_parallelize_fn`
    inside :meth:`stable_pretraining.Module.apply_fsdp2_sharding_if_needed`
    (Lightning's intended FSDP2 hook). This factory's job is just to
    construct the strategy with reasonable parallelism and checkpointing
    defaults, and to surface ``mp_policy`` as the one knob users commonly
    want to tune without subclassing ``Module``.

    Args:
        save_distributed_checkpoint: When ``True`` (default), Lightning
            saves checkpoints using ``torch.distributed.checkpoint`` (sharded
            DTensor). Set ``False`` to gather to a single full state dict
            at save time (slower, scales worse, but a single file).
        data_parallel_size: Forwarded to :class:`ModelParallelStrategy`.
            ``None`` (default) auto-computes: prefers ``LOCAL_WORLD_SIZE``
            (set by ``torchrun`` and friends), falls back to
            ``torch.cuda.device_count()``. Lightning's own ``"auto"``
            resolves to ``num_nodes`` (=1 single-node) which would leave
            the data-parallel axis at size 1 and fail the
            ``data_parallel_size * tensor_parallel_size == world_size``
            check; this default avoids that footgun. The inferred value
            is logged. Pass an explicit integer for unusual layouts.
        tensor_parallel_size: Forwarded to :class:`ModelParallelStrategy`.
            Default ``1`` means no tensor parallelism (pure FSDP2). Pass
            ``>1`` only with a custom :meth:`Module.apply_fsdp2_sharding_if_needed`
            that knows how to apply tensor parallelism — the default
            :func:`default_parallelize_fn` slices out the DP submesh and
            does pure FSDP2.
        mp_policy: Optional :class:`MixedPrecisionPolicy` controlling
            FSDP2 mixed-precision (param dtype, reduce dtype, output
            dtype). Stashed on the strategy as ``_spt_mp_policy`` and
            picked up by ``default_parallelize_fn`` via the strategy ref
            on the LightningModule's trainer.
        **kwargs: Forwarded to :class:`ModelParallelStrategy`
            (``process_group_backend``, ``timeout``).

    Returns:
        A :class:`ModelParallelStrategy` ready to pass into ``Trainer(strategy=...)``.

    Note:
        :meth:`stable_pretraining.Module.configure_model` checks for
        ``self._device_mesh`` (set by ``ModelParallelStrategy``) and
        dispatches to :func:`default_parallelize_fn`. Custom subclasses of
        ``Module`` can override :meth:`apply_fsdp2_sharding_if_needed` to
        use a different parallelize policy (e.g. tensor parallelism,
        custom block selection, hybrid sharding).
    """
    if not _FSDP_AVAILABLE:
        raise RuntimeError("FSDP is not available in this PyTorch build.")

    if data_parallel_size is None:
        env_local_world = os.environ.get("LOCAL_WORLD_SIZE")
        if env_local_world is not None:
            data_parallel_size = int(env_local_world)
            source = "LOCAL_WORLD_SIZE"
        else:
            data_parallel_size = max(1, torch.cuda.device_count())
            source = "torch.cuda.device_count()"
        logging.info(
            f"make_fsdp_strategy: data_parallel_size={data_parallel_size} "
            f"(inferred from {source})"
        )

    strategy = ModelParallelStrategy(
        data_parallel_size=data_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        save_distributed_checkpoint=save_distributed_checkpoint,
        **kwargs,
    )
    # Stash mp_policy under a namespaced attribute so
    # ``Module.apply_fsdp2_sharding_if_needed`` can read it off the strategy
    # without us touching Lightning's private surface.
    strategy._spt_mp_policy = mp_policy
    return strategy
