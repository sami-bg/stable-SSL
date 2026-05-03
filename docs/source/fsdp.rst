Distributed Training with FSDP
==============================

``stable-pretraining`` supports `PyTorch's FSDP
<https://pytorch.org/docs/stable/fsdp.html>`_ as a first-class training
strategy alongside DDP. This page covers when to reach for FSDP, how to
opt in, and the known limitations.


When to use FSDP vs DDP
-----------------------

DDP replicates parameters, gradients, and optimizer state on every rank.
FSDP shards them. The trade-off is memory vs. communication.

**Use DDP when:**

- The full model + AdamW optimizer state comfortably fits on a single GPU.
  As a rough rule of thumb, anything up to ViT-Large with standard SSL
  setups runs fine on DDP and is faster than FSDP on small clusters.
- You're optimizing for throughput on a small number of GPUs.

**Reach for FSDP when:**

- Optimizer state or activations dominate memory and you'd otherwise OOM
  (e.g. large EMA-teacher methods like DINOv2 with a ViT-Large student).
- You want to fit a larger batch on the same hardware.
- You're scaling to many GPUs and DDP's full replication is wasteful.

When in doubt, start with DDP. Switch to FSDP only when memory is the
binding constraint.


Opting in
---------

The simplest way to use FSDP is via :func:`make_fsdp_strategy`:

.. code-block:: python

    import lightning as pl
    from stable_pretraining.utils.fsdp import (
        default_auto_wrap_policy,
        make_fsdp_strategy,
    )

    strategy = make_fsdp_strategy(
        auto_wrap_policy=default_auto_wrap_policy(model),
        sharding_strategy="FULL_SHARD",
        cpu_offload=False,
        state_dict_type="sharded",
    )

    trainer = pl.Trainer(
        strategy=strategy,
        accelerator="gpu",
        devices=num_gpus,
        ...
    )

The factory returns a :class:`CallbackAwareFSDPStrategy`, which behaves
exactly like Lightning's :class:`~lightning.pytorch.strategies.FSDPStrategy`
plus one critical fix — see :ref:`callback_aware_strategy` below.

For a complete worked example, compare these two scripts side by side:

- ``benchmarks/imagenet10/lejepa_vit_small.py`` (single GPU variant)
- ``benchmarks/imagenet10/lejepa_vit_small_fsdp.py`` (FSDP variant)

The diff is intentionally small: imports, a strategy block, and the
``strategy=`` argument to ``pl.Trainer``. Everything else (model, data,
callbacks, optimizer) is identical.


What the integration provides
-----------------------------

The :mod:`stable_pretraining.utils.fsdp` module exposes:

- :func:`make_fsdp_strategy` — Hydra-friendly factory. The recommended
  entry point.
- :func:`default_auto_wrap_policy` — auto-detects repeated ``*Block`` /
  ``*Layer`` classes (timm ViT, torchvision ResNet) and returns a
  :class:`ModuleWrapPolicy` over them. Falls back to
  :func:`size_based_auto_wrap_policy` when no repeated blocks exist.
- :class:`CallbackAwareFSDPStrategy` — see below.
- :class:`TeacherStudentWrapper.fsdp_setup` —
  :class:`~stable_pretraining.TeacherStudentWrapper` does an in-place EMA
  via ``zip(teacher.parameters(), student.parameters())``. Under FSDP
  this is correct **only if** student and teacher have identical shard
  layout. ``fsdp_setup(auto_wrap_policy=...)`` wraps both with the same
  policy and asserts alignment so a mismatched policy fails loudly
  instead of silently corrupting the EMA.
- :func:`is_fsdp_strategy` / :func:`describe_fsdp_strategy` — detection
  and a serializable summary of the strategy's settings, used by
  :class:`~stable_pretraining.Manager` for FSDP-specific logging.


.. _callback_aware_strategy:

Why ``CallbackAwareFSDPStrategy`` exists
----------------------------------------

The strategy returned by :func:`make_fsdp_strategy` is a subclass of
Lightning's :class:`FSDPStrategy` that auto-excludes
``module.callbacks_modules`` and ``module.callbacks_metrics`` from
sharding. This is required because of how online evaluation callbacks
register themselves.

:class:`~stable_pretraining.callbacks.OnlineProbe` (and other
:class:`TrainableCallback` subclasses) store their probe / KNN queue /
metric modules in ``pl_module.callbacks_modules``, and each owns its
**own optimizer** built over those parameters. That optimizer expects
full parameter tensors and knows nothing about distributed training.

If FSDP shards those parameters, the probe's optimizer ends up holding
references to tensors that no longer exist in the form they were when
the optimizer was constructed. Updates either silently corrupt or
visibly fail. ``CallbackAwareFSDPStrategy`` fixes this by injecting the
callback containers into FSDP's ``ignored_modules`` at wrap time —
their parameters stay as regular ``nn.Parameter`` on every rank, the
probe optimizer stays correct.

Users don't need to think about this; it happens automatically. The
class is documented here for the few cases where you'd want to
subclass it (custom cluster-specific ``ignored_modules``, additional
wrap-time hooks).


Overrides
----------------

When the auto-detected wrap policy isn't what you want, pass an
explicit one. ``auto_wrap_policy`` accepts any callable matching FSDP's
expected signature:

.. code-block:: python

    from torch.distributed.fsdp.wrap import ModuleWrapPolicy
    from timm.models.vision_transformer import Block as ViTBlock

    strategy = make_fsdp_strategy(
        auto_wrap_policy=ModuleWrapPolicy({ViTBlock}),
    )

Other arguments of interest:

- ``sharding_strategy``: ``"FULL_SHARD"`` (default — most memory savings),
  ``"SHARD_GRAD_OP"`` (don't shard params, only grads + optim state —
  faster, less savings), ``"NO_SHARD"`` (degenerate, equivalent to DDP).
- ``cpu_offload=True`` — offload sharded params to CPU when not in use.
  Further reduces GPU memory at the cost of throughput.
- ``state_dict_type="sharded"`` (default — fast save / load, scales to
  large models) vs ``"full"`` (compatible with non-FSDP loading but
  slower and bottlenecked through rank 0).
- ``ignore_callbacks=False`` — fall back to vanilla
  :class:`FSDPStrategy`. Only useful if you have a reason to bypass
  the callback exclusion.

Any additional ``**kwargs`` to :func:`make_fsdp_strategy` are forwarded
to :class:`FSDPStrategy` and ultimately to
:class:`FullyShardedDataParallel`.


See also
--------

- :func:`stable_pretraining.utils.fsdp.make_fsdp_strategy`
- :class:`stable_pretraining.utils.fsdp.CallbackAwareFSDPStrategy`
- :func:`stable_pretraining.utils.fsdp.default_auto_wrap_policy`
- :class:`stable_pretraining.TeacherStudentWrapper` — see ``fsdp_setup``
- :class:`stable_pretraining.callbacks.OnlineProbe`
- ``benchmarks/imagenet10/lejepa-vit-small-fsdp.py`` — worked example
