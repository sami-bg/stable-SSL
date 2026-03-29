import time
from typing import Any, List, Optional

import torch
from lightning.pytorch.callbacks import Callback
from loguru import logger


class CPUOffloadCallback(Callback):
    """Offload checkpoint tensors to CPU during save to reduce GPU memory usage.

    This callback intercepts checkpoint saving and moves all PyTorch tensors
    (model weights, optimizer states, scheduler states) from GPU to CPU before
    writing to disk. Prevents GPU OOM for large models (2B+ parameters).

    **Compatible Strategies:**
    - DDP (DistributedDataParallel)
    - Single GPU training

    **Incompatible Strategies (auto-disabled):**
    - FSDP (uses sharded checkpointing)
    - DeepSpeed (has custom checkpoint mechanism)

    Args:
        offload_keys: Keys to offload. Defaults to
            ``['state_dict', 'optimizer_states', 'lr_schedulers']``.
    """

    def __init__(self, offload_keys: Optional[List[str]] = None):
        super().__init__()
        self.offload_keys = offload_keys or [
            "state_dict",
            "optimizer_states",
            "lr_schedulers",
        ]
        self._checkpoint_count = 0
        self._total_time = 0.0
        self._total_memory_freed = 0.0
        self._is_enabled = True

    def setup(self, trainer, pl_module, stage: str):
        strategy_name = trainer.strategy.__class__.__name__
        self._is_enabled = self._check_strategy_compatibility(
            trainer.strategy, strategy_name
        )
        if not self._is_enabled:
            logger.warning(
                f"CPUOffloadCallback disabled: incompatible strategy '{strategy_name}'"
            )
        else:
            logger.info(f"CPUOffloadCallback enabled (strategy: {strategy_name})")

    def _check_strategy_compatibility(self, strategy, strategy_name: str) -> bool:
        from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy

        if isinstance(strategy, (DDPStrategy, SingleDeviceStrategy)):
            return True

        try:
            from lightning.pytorch.strategies import FSDPStrategy

            if isinstance(strategy, FSDPStrategy):
                return False
        except ImportError:
            pass

        try:
            from lightning.pytorch.strategies import DeepSpeedStrategy

            if isinstance(strategy, DeepSpeedStrategy):
                return False
        except ImportError:
            pass

        # Unknown strategy — be conservative
        logger.warning(
            f"CPUOffloadCallback: unknown strategy '{strategy_name}', disabling"
        )
        return False

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if not self._is_enabled or trainer.global_rank != 0:
            return

        self._checkpoint_count += 1
        start_time = time.time()

        gpu_mem_before = (
            torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        )

        # Move GPU tensors to CPU
        gpu_tensors_moved = 0
        gpu_bytes_moved = 0
        for key in self.offload_keys:
            if key in checkpoint:
                moved, nbytes = self._safe_to_cpu(checkpoint, key)
                gpu_tensors_moved += moved
                gpu_bytes_moved += nbytes

        elapsed = time.time() - start_time
        self._total_time += elapsed

        gpu_mem_after = (
            torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        )
        mem_freed = gpu_mem_before - gpu_mem_after
        self._total_memory_freed += max(mem_freed, 0)

        if gpu_tensors_moved > 0:
            logger.success(
                f"Checkpoint #{self._checkpoint_count}: "
                f"moved {gpu_tensors_moved} GPU tensors "
                f"({gpu_bytes_moved / 1e9:.2f} GB) to CPU in {elapsed:.1f}s, "
                f"freed {mem_freed:.2f} GB"
            )
        else:
            logger.info(
                f"Checkpoint #{self._checkpoint_count}: "
                f"all tensors already on CPU ({elapsed:.1f}s)"
            )

    def _safe_to_cpu(self, parent, key) -> tuple:
        """Recursively move GPU tensors to CPU. Returns (count, bytes)."""
        obj = parent[key]
        moved, nbytes = self._to_cpu_recursive(obj, parent, key)
        return moved, nbytes

    def _to_cpu_recursive(self, obj: Any, parent: Any = None, key: Any = None) -> tuple:
        """Walk the checkpoint tree, copying GPU tensors to CPU.

        Uses ``.detach().cpu()`` so the live optimizer/model state on GPU is
        not affected — the checkpoint gets a CPU *copy*, the originals stay
        on their device.
        """
        if isinstance(obj, torch.Tensor):
            if obj.is_cuda:
                nbytes = obj.element_size() * obj.nelement()
                parent[key] = obj.detach().clone().cpu()
                return 1, nbytes
            return 0, 0

        if isinstance(obj, dict):
            total_moved, total_bytes = 0, 0
            for k in obj:
                m, b = self._to_cpu_recursive(obj[k], obj, k)
                total_moved += m
                total_bytes += b
            return total_moved, total_bytes

        if isinstance(obj, (list, tuple)):
            total_moved, total_bytes = 0, 0
            items = list(obj) if isinstance(obj, tuple) else obj
            for i in range(len(items)):
                m, b = self._to_cpu_recursive(items[i], items, i)
                total_moved += m
                total_bytes += b
            if isinstance(obj, tuple):
                parent[key] = tuple(items)
            return total_moved, total_bytes

        # Scalars, strings, etc. — skip silently
        return 0, 0

    def on_exception(self, trainer, pl_module, exception):
        if trainer.global_rank == 0:
            logger.error(
                f"CPUOffloadCallback: exception during training "
                f"({self._checkpoint_count} checkpoints saved before failure)"
            )

    def state_dict(self):
        return {
            "checkpoint_count": self._checkpoint_count,
            "total_time": self._total_time,
            "total_memory_freed": self._total_memory_freed,
            "is_enabled": self._is_enabled,
        }

    def load_state_dict(self, state_dict):
        self._checkpoint_count = state_dict.get("checkpoint_count", 0)
        self._total_time = state_dict.get("total_time", 0.0)
        self._total_memory_freed = state_dict.get("total_memory_freed", 0.0)
        self._is_enabled = state_dict.get("is_enabled", True)
        logger.info(
            f"CPUOffloadCallback: restored state "
            f"({self._checkpoint_count} prior checkpoints)"
        )
