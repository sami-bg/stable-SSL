from typing import Dict, List

import torch
from torch import nn
from lightning.pytorch.callbacks import Callback
from loguru import logger


class LogUnusedParametersOnce(Callback):
    """Lightning callback that logs parameters which do NOT receive gradients.

    - Registers hooks on all leaf parameters (requires_grad=True).
    - After the first backward pass, logs unused parameters via loguru.
    - Removes all hooks and disables itself for the rest of training.
    """

    def __init__(self, verbose: bool = True):
        super().__init__()
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._used_flags: Dict[nn.Parameter, bool] = {}
        self._enabled: bool = True
        self._verbose = verbose

    def _register_hooks(self, model: nn.Module):
        """Attach hooks to all leaf parameters that require gradient."""
        assert not self._hooks, "Hooks already registered"

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if not p.is_leaf:
                # Only track leaf parameters, those are relevant for DDP warnings
                continue

            self._used_flags[p] = False

            def make_hook(param):
                def hook(grad):
                    self._used_flags[param] = True

                return hook

            h = p.register_hook(make_hook(p))
            self._hooks.append(h)

        if self._verbose:
            logger.info(
                f"[LogUnusedParametersOnce] Registered hooks on "
                f"{len(self._used_flags)} leaf parameters."
            )

    def _remove_hooks(self):
        """Remove all hooks and clear state."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._used_flags.clear()

    def _report_and_disable(self, pl_module: nn.Module):
        """Report unused parameters to loguru and disable further tracking."""
        # Build map name -> param for convenience
        name_by_param = {p: n for n, p in pl_module.named_parameters()}

        unused_names = [
            name_by_param[p] for p, used in self._used_flags.items() if not used
        ]

        if not unused_names:
            logger.info(
                "[LogUnusedParametersOnce] All tracked parameters received gradients "
                "on the first backward pass."
            )
        else:
            logger.warning(
                "[LogUnusedParametersOnce] The following parameters did NOT receive "
                "gradients on the first backward pass (potentially causing "
                "Lightning's 'unused parameters' error):"
            )
            for name in unused_names:
                logger.warning(f"  - {name}")

        # Clean up hooks and disable
        self._remove_hooks()
        self._enabled = False
        if self._verbose:
            logger.info("[LogUnusedParametersOnce] Hooks removed, callback disabled.")

    # --- Lightning hooks ---

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Register hooks right before the first training batch starts."""
        if not self._enabled:
            return

        # Only do this once, on the very first train batch of the whole run
        if trainer.global_step == 0 and batch_idx == 0:
            # Ensure clean state
            self._remove_hooks()
            self._used_flags.clear()
            self._register_hooks(pl_module)

    def on_after_backward(self, trainer, pl_module):
        """After the first backward pass, report unused params and detach hooks."""
        if not self._enabled:
            return

        # Only report once, right after the first backward call
        self._report_and_disable(pl_module)
