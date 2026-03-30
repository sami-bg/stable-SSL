import os
import time

from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import RichProgressBar as _RichProgressBar

from .checkpoint_sklearn import SklearnCheckpoint, WandbCheckpoint
from .env_info import EnvironmentDumpCallback
from .hf_models import HuggingFaceCheckpointCallback
from .registry import ModuleRegistryCallback
from .trainer_info import LoggingCallback, ModuleSummary, SLURMInfo, TrainerInfo
from .unused_parameters import LogUnusedParametersOnce


class RichProgressBar(_RichProgressBar):
    """RichProgressBar with a workaround for a known Rich/Lightning bug.

    Lightning's ``_stop_progress`` can call ``Live.stop()`` when the live stack
    is already empty, raising ``IndexError: pop from empty list``.  This
    subclass catches that error so teardown completes cleanly.
    """

    def _stop_progress(self) -> None:
        try:
            super()._stop_progress()
        except IndexError:
            pass


class PrintProgressBar(Callback):
    """Plain-text progress logger for non-interactive environments (SLURM, CI).

    Prints a one-liner every ``log_every_n_steps`` training batches so that
    progress shows up in slurm .out files and the wandb Logs tab.
    """

    def __init__(self, log_every_n_steps: int = 50):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self._epoch_start = None

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._epoch_start = time.time()

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx
    ) -> None:
        if (batch_idx + 1) % self.log_every_n_steps != 0:
            return

        total = trainer.num_training_batches
        epoch = trainer.current_epoch
        max_epochs = trainer.max_epochs or "?"
        elapsed = time.time() - self._epoch_start if self._epoch_start else 0
        it_s = (batch_idx + 1) / elapsed if elapsed > 0 else 0

        # Grab metrics on the progress bar
        metrics = trainer.progress_bar_metrics
        metrics_str = " | ".join(f"{k}: {v:.4g}" for k, v in metrics.items())

        print(
            f"[Epoch {epoch}/{max_epochs}] "
            f"step {batch_idx + 1}/{total} "
            f"({it_s:.1f} it/s)" + (f" | {metrics_str}" if metrics_str else ""),
            flush=True,
        )

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        elapsed = time.time() - self._epoch_start if self._epoch_start else 0
        epoch = trainer.current_epoch
        max_epochs = trainer.max_epochs or "?"
        metrics = trainer.progress_bar_metrics
        metrics_str = " | ".join(f"{k}: {v:.4g}" for k, v in metrics.items())
        print(
            f"[Epoch {epoch}/{max_epochs}] done in {elapsed:.1f}s"
            + (f" | {metrics_str}" if metrics_str else ""),
            flush=True,
        )


def _make_progress_bar():
    """Create a progress bar callback.

    If stdout is a tty (local shell, interactive srun, etc.), uses Rich for
    a nice live display.  Otherwise (sbatch, Hydra multirun, piped output),
    falls back to a plain-text line printer that shows up in slurm .out
    files and the wandb Logs tab.
    """
    if os.isatty(1):
        return RichProgressBar()
    return PrintProgressBar()


def default():
    """Factory function that returns default callbacks."""
    callbacks = [
        _make_progress_bar(),
        ModuleRegistryCallback(),
        LoggingCallback(),
        EnvironmentDumpCallback(async_dump=True),
        TrainerInfo(),
        SklearnCheckpoint(),
        WandbCheckpoint(),
        ModuleSummary(),
        SLURMInfo(),
        LogUnusedParametersOnce(),
        HuggingFaceCheckpointCallback(),
    ]

    return callbacks
