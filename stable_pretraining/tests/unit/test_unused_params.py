# test_log_unused_parameters_once.py

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

from loguru import logger


# ---- Bring in the callback implementation under test ----


class LogUnusedParametersOnce(Callback):
    """Lightning callback that logs parameters which do NOT receive gradients.

    on the first training batch only.

    - Registers hooks on all leaf parameters (requires_grad=True).
    - After the first backward pass, logs unused parameters via loguru.
    - Removes all hooks and disables itself for the rest of training.
    """

    def __init__(self, verbose: bool = True):
        super().__init__()
        self._hooks = []
        self._used_flags = {}
        self._enabled = True
        self._verbose = verbose

    def _register_hooks(self, model: nn.Module):
        assert not self._hooks, "Hooks already registered"

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if not p.is_leaf:
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
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._used_flags.clear()

    def _report_and_disable(self, pl_module: nn.Module):
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

        self._remove_hooks()
        self._enabled = False
        if self._verbose:
            logger.info("[LogUnusedParametersOnce] Hooks removed, callback disabled.")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not self._enabled:
            return

        if trainer.global_step == 0 and batch_idx == 0:
            self._remove_hooks()
            self._used_flags.clear()
            self._register_hooks(pl_module)

    def on_after_backward(self, trainer, pl_module):
        if not self._enabled:
            return

        self._report_and_disable(pl_module)


# ---- Minimal dataset and LightningModule fixtures ----


class RandomDataset(Dataset):
    """Minimal testing class."""

    def __init__(self, length: int = 16, in_dim: int = 8, out_dim: int = 4):
        self.length = length
        self.in_dim = in_dim
        self.out_dim = out_dim

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = torch.randn(self.in_dim)
        y = torch.randint(0, self.out_dim, (1,)).item()
        return x, y


class SimpleModelAllUsed(pl.LightningModule):
    """Model where all parameters are used for the loss."""

    def __init__(self, in_dim: int = 8, out_dim: int = 4):
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


class ModelWithUnusedParam(pl.LightningModule):
    """Model that contains a parameter that does not affect the loss."""

    def __init__(self, in_dim: int = 8, out_dim: int = 4):
        super().__init__()
        self.used_layer = nn.Linear(in_dim, out_dim)
        # This layer is never used in forward -> its parameters should remain without gradients
        self.unused_layer = nn.Linear(in_dim, out_dim)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        # Only use used_layer
        return self.used_layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def _make_trainer(callback: Callback, max_steps: int = 1):
    """Utility to create a tiny trainer for unit tests."""
    return pl.Trainer(
        accelerator="cpu",
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
        callbacks=[callback],
        log_every_n_steps=1,
    )


# ---- Tests ----


@pytest.mark.unit
def test_all_parameters_used(caplog):
    """All parameters receive grads -> callback should log 'all used' and no unused names."""
    callback = LogUnusedParametersOnce(verbose=True)
    model = SimpleModelAllUsed()
    dataset = RandomDataset()
    loader = DataLoader(dataset, batch_size=4)

    trainer = _make_trainer(callback)
    trainer.fit(model, train_dataloaders=loader)

    # Ensure callback disabled after first backward
    assert not callback._enabled
    assert len(callback._hooks) == 0
    assert callback._used_flags == {}

    # Check logs contain "All tracked parameters received gradients"
    text = caplog.text
    assert "[LogUnusedParametersOnce] All tracked parameters received gradients" in text
    # Should not contain per-parameter "did NOT receive gradients" warnings
    assert "did NOT receive gradients" not in text


@pytest.mark.unit
def test_unused_parameters_logged(caplog):
    """ModelWithUnusedParam has some parameters unused -> they should be logged."""
    callback = LogUnusedParametersOnce(verbose=True)
    model = ModelWithUnusedParam()
    dataset = RandomDataset()
    loader = DataLoader(dataset, batch_size=4)

    trainer = _make_trainer(callback)
    trainer.fit(model, train_dataloaders=loader)

    assert not callback._enabled
    assert len(callback._hooks) == 0
    assert callback._used_flags == {}

    text = caplog.text

    # Should contain the main warning header
    assert "did NOT receive gradients on the first backward pass" in text

    # All params of unused_layer should appear in logs
    for name, _ in model.unused_layer.named_parameters():
        full_name = f"unused_layer.{name}"
        assert full_name in text


@pytest.mark.unit
def test_callback_runs_only_once(caplog):
    """Callback should register hooks and report only once, even if trainer runs multiple steps."""

    # Make a trainer that will do 2 steps, but callback should disable after first step
    class TwoStepTrainer(pl.Trainer):
        pass

    callback = LogUnusedParametersOnce(verbose=True)
    model = SimpleModelAllUsed()
    dataset = RandomDataset(length=32)
    loader = DataLoader(dataset, batch_size=4)

    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=1,
        limit_train_batches=2,
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
        callbacks=[callback],
        log_every_n_steps=1,
    )

    trainer.fit(model, train_dataloaders=loader)

    text = caplog.text

    # Should see registration once
    assert text.count("Registered hooks on") == 1
    # Should see the disable log once
    assert text.count("Hooks removed, callback disabled.") == 1
