"""FSDP integration helpers for stable-pretraining.

Provides:
- :func:`default_auto_wrap_policy`: a sensible default policy that wraps
  transformer / residual blocks.
- :func:`make_fsdp_strategy`: a Hydra-friendly factory returning a
  :class:`CallbackAwareFSDPStrategy` that automatically excludes
  ``callbacks_modules`` and ``callbacks_metrics`` from sharding.
- :class:`CallbackAwareFSDPStrategy`: subclass of Lightning's
  :class:`~lightning.pytorch.strategies.FSDPStrategy` that injects callback
  containers into ``ignored_modules`` at model setup time, since these
  containers hold callback-owned modules with their own optimizers.

The ``callbacks_modules`` and ``callbacks_metrics`` exclusion is required
because :class:`stable_pretraining.Module` stores
:class:`~stable_pretraining.callbacks.utils.TrainableCallback` modules (online
probes, KNN, RankMe, ...) inside these :class:`torch.nn.ModuleDict` containers.
Each such callback owns its own optimizer; sharding their parameters under the
main FSDP unit would put them out of reach of the callback's optimizer.
"""

from __future__ import annotations

from typing import Iterable, Literal, Optional, Set, Type, Union

import torch.nn as nn
from loguru import logger as logging

try:
    from lightning.pytorch.strategies import FSDPStrategy
    from torch.distributed.fsdp import (
        CPUOffload,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import (
        ModuleWrapPolicy,
        size_based_auto_wrap_policy,
    )

    _FSDP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FSDP_AVAILABLE = False
    FSDPStrategy = object  # type: ignore[misc,assignment]


__all__ = [
    "default_auto_wrap_policy",
    "make_fsdp_strategy",
    "CallbackAwareFSDPStrategy",
    "find_callback_containers",
    "assert_aligned_wrapping",
    "is_fsdp_strategy",
    "describe_fsdp_strategy",
]


def is_fsdp_strategy(strategy_or_trainer) -> bool:
    """Return True if the argument is (or wraps) an FSDP strategy.

    Accepts either a Lightning :class:`Strategy` instance or a Lightning
    :class:`Trainer` (in which case ``trainer.strategy`` is inspected).

    Returns False (without raising) if FSDP is not importable, so callers
    can use this in conditional code without guarding the import themselves.
    """
    if not _FSDP_AVAILABLE:
        return False
    # Trainer case
    strat = getattr(strategy_or_trainer, "strategy", strategy_or_trainer)
    return isinstance(strat, FSDPStrategy)


def describe_fsdp_strategy(strategy_or_trainer) -> dict:
    """Return a serializable summary of an FSDP strategy's relevant settings.

    Useful for logging and for tests that want to assert on the strategy
    configuration without poking at private attributes from many places.
    """
    if not is_fsdp_strategy(strategy_or_trainer):
        return {"is_fsdp": False}
    strat = getattr(strategy_or_trainer, "strategy", strategy_or_trainer)
    sharding = getattr(strat, "sharding_strategy", None)
    state_dict_type = getattr(strat, "_state_dict_type", None) or getattr(
        strat, "state_dict_type", None
    )
    ignored = (
        list(strat.kwargs.get("ignored_modules", []))
        if hasattr(strat, "kwargs")
        else []
    )
    auto_wrap = (
        strat.kwargs.get("auto_wrap_policy") if hasattr(strat, "kwargs") else None
    )
    return {
        "is_fsdp": True,
        "subclass": type(strat).__name__,
        "sharding_strategy": str(sharding) if sharding is not None else None,
        "state_dict_type": str(state_dict_type)
        if state_dict_type is not None
        else None,
        "n_ignored_modules": len(ignored),
        "auto_wrap_policy": type(auto_wrap).__name__ if auto_wrap is not None else None,
    }


def assert_aligned_wrapping(student: nn.Module, teacher: nn.Module) -> None:
    """Assert that ``student`` and ``teacher`` have identical FSDP shard layouts.

    Required by :class:`TeacherStudentWrapper.update_teacher`, which performs
    in-place EMA via ``zip(teacher.parameters(), student.parameters())``. If
    the two modules are wrapped with mismatched FSDP policies, ``zip`` will
    silently pair shards that do not correspond — corruption, not a crash.

    The check covers:

    1. Same number of parameter tensors yielded by ``parameters()``.
    2. Each pair has the same ``shape`` and ``dtype``.
    3. Same number and shapes of buffers.

    Raises:
        AssertionError: with a description of the first mismatch found.
    """
    s_params = list(student.parameters())
    t_params = list(teacher.parameters())
    if len(s_params) != len(t_params):
        raise AssertionError(
            f"FSDP wrapping mismatch: student has {len(s_params)} parameter "
            f"tensors, teacher has {len(t_params)}. Both must be wrapped with "
            f"the same auto_wrap_policy."
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


def _detect_block_classes(model: nn.Module) -> Set[Type[nn.Module]]:
    """Best-effort detection of transformer / residual block classes in ``model``.

    Heuristic: any class whose name ends with ``Block`` or ``Layer`` and which
    is repeated in the model is a candidate for per-block FSDP wrapping. This
    matches conventions in timm ViT (``Block``), HuggingFace ViT
    (``ViTLayer``), torchvision ResNet (``BasicBlock`` / ``Bottleneck``), and
    similar libraries.
    """
    counts: dict[Type[nn.Module], int] = {}
    for sub in model.modules():
        cls = type(sub)
        name = cls.__name__
        if name.endswith("Block") or name.endswith("Layer"):
            counts[cls] = counts.get(cls, 0) + 1
    # Only keep block classes that appear more than once (a stack of them).
    return {cls for cls, n in counts.items() if n >= 2}


def default_auto_wrap_policy(
    model: Optional[nn.Module] = None,
    *,
    block_classes: Optional[Iterable[Type[nn.Module]]] = None,
    min_num_params: int = 100_000,
):
    """Build a default FSDP auto-wrap policy.

    If ``block_classes`` is provided, returns a :class:`ModuleWrapPolicy` for
    those classes. Otherwise, if ``model`` is provided, classes are auto-detected
    via :func:`_detect_block_classes`. As a last resort (no blocks found and no
    model given), falls back to :func:`size_based_auto_wrap_policy` with
    ``min_num_params`` as the threshold.

    Args:
        model: The model to inspect for transformer/residual block classes.
        block_classes: Explicit list of block classes to wrap.
        min_num_params: Threshold for the size-based fallback.

    Returns:
        A callable suitable for ``FSDPStrategy(auto_wrap_policy=...)``.
    """
    if not _FSDP_AVAILABLE:
        raise RuntimeError("FSDP is not available in this PyTorch build.")

    if block_classes is None and model is not None:
        block_classes = _detect_block_classes(model)

    if block_classes:
        block_classes = set(block_classes)
        logging.info(
            f"default_auto_wrap_policy: ModuleWrapPolicy over "
            f"{[c.__name__ for c in block_classes]}"
        )
        return ModuleWrapPolicy(block_classes)

    logging.info(
        f"default_auto_wrap_policy: no block classes detected, "
        f"falling back to size_based_auto_wrap_policy(min_num_params={min_num_params})"
    )
    from functools import partial

    return partial(size_based_auto_wrap_policy, min_num_params=min_num_params)


class CallbackAwareFSDPStrategy(FSDPStrategy):  # type: ignore[misc]
    """:class:`FSDPStrategy` that auto-excludes callback containers from sharding.

    On model setup, walks the LightningModule tree and adds every
    ``callbacks_modules`` / ``callbacks_metrics`` :class:`torch.nn.ModuleDict`
    found to FSDP's ``ignored_modules``. Existing user-supplied
    ``ignored_modules`` are preserved.

    Most users should construct this via :func:`make_fsdp_strategy` rather
    than instantiating it directly. Subclass this class if you need to extend
    the wrap-time hook (e.g. to add cluster- or project-specific
    ``ignored_modules`` beyond the callback containers).
    """

    def _setup_model(self, model: nn.Module) -> nn.Module:  # type: ignore[override]
        # Find callback containers and add them to ignored_modules so FSDP does
        # not flatten/shard their parameters. This must happen before super()
        # wraps the model.
        containers = find_callback_containers(model)
        if containers:
            existing = list(self.kwargs.get("ignored_modules", []) or [])
            # Deduplicate by id while preserving order.
            seen_ids = {id(m) for m in existing}
            for c in containers:
                if id(c) not in seen_ids:
                    existing.append(c)
                    seen_ids.add(id(c))
            self.kwargs["ignored_modules"] = existing
            logging.info(
                f"CallbackAwareFSDPStrategy: excluding {len(containers)} callback "
                f"container(s) from FSDP sharding"
            )
        return super()._setup_model(model)


def make_fsdp_strategy(
    *,
    auto_wrap_policy=None,
    sharding_strategy: Union[str, "ShardingStrategy"] = "FULL_SHARD",
    cpu_offload: Union[bool, "CPUOffload", None] = False,
    state_dict_type: Literal["full", "sharded"] = "sharded",
    ignore_callbacks: bool = True,
    **kwargs,
):
    """Hydra-friendly factory for an FSDP strategy with sensible defaults.

    Args:
        auto_wrap_policy: Optional callable or set of nn.Module classes. If
            ``None``, the user is expected to set the policy by other means
            (e.g. via the model's transformer block class) or rely on
            size-based wrapping. Pass a callable like the one returned by
            :func:`default_auto_wrap_policy` for typical SSL workloads.
        sharding_strategy: String name (e.g. ``"FULL_SHARD"``,
            ``"SHARD_GRAD_OP"``, ``"NO_SHARD"``) or a
            :class:`ShardingStrategy` enum value.
        cpu_offload: ``True`` to enable parameter CPU offload.
        state_dict_type: ``"sharded"`` or ``"full"``. Sharded scales better;
            full is more compatible with non-FSDP loading.
        ignore_callbacks: When ``True`` (default), returns
            :class:`CallbackAwareFSDPStrategy` which excludes
            ``callbacks_modules`` / ``callbacks_metrics`` from sharding. Set
            ``False`` to use the vanilla :class:`FSDPStrategy`.
        **kwargs: Forwarded to :class:`FSDPStrategy` and ultimately to
            :class:`FullyShardedDataParallel`.
    """
    if not _FSDP_AVAILABLE:
        raise RuntimeError("FSDP is not available in this PyTorch build.")

    if isinstance(sharding_strategy, str):
        sharding_strategy = ShardingStrategy[sharding_strategy]

    if isinstance(cpu_offload, bool):
        cpu_offload = CPUOffload(offload_params=cpu_offload)

    cls = CallbackAwareFSDPStrategy if ignore_callbacks else FSDPStrategy
    return cls(
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        cpu_offload=cpu_offload,
        state_dict_type=state_dict_type,
        **kwargs,
    )
