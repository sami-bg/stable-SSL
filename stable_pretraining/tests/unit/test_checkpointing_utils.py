import pytest
from pathlib import Path
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from stable_pretraining.utils.checkpointing_utils import configure_checkpointing


# A simple mock trainer for testing purposes
class BoringTrainer(pl.Trainer):
    """A mock Trainer that disables some default callbacks to keep testing predictable."""

    def __init__(self, **kwargs):
        # Disable some default callbacks to make testing more predictable
        kwargs.setdefault("enable_progress_bar", False)
        kwargs.setdefault("enable_model_summary", False)
        super().__init__(**kwargs)


@pytest.mark.unit
class TestConfigureCheckpointing:
    """Tests the `configure_checkpointing` utility function across various user scenarios."""

    def test_case_1_intentional_ckpt_path_and_callback(self, tmp_path: Path):
        """Tests Case 1: The user provides a `ckpt_path` and a matching `ModelCheckpoint` callback.

        This scenario represents a correctly configured setup where the user's intent to save/resume
        from a specific path is perfectly aligned with their callback configuration.

        Expectation: The function should recognize the valid setup and make no changes to the
                     trainer's callbacks.
        """
        ckpt_path = tmp_path / "checkpoints" / "last.ckpt"
        ckpt_path.parent.mkdir()

        callbacks = [ModelCheckpoint(dirpath=str(ckpt_path.parent))]
        trainer = BoringTrainer(callbacks=callbacks, default_root_dir=str(tmp_path))

        initial_callback_count = len(trainer.callbacks)

        configure_checkpointing(trainer, ckpt_path)

        assert len(trainer.callbacks) == initial_callback_count
        assert 1 == sum(isinstance(cb, ModelCheckpoint) for cb in trainer.callbacks)

    def test_case_2_intentional_ckpt_path_but_no_callback(self, tmp_path: Path):
        """Tests Case 2: The user provides a `ckpt_path` but forgets the `ModelCheckpoint` callback.

        This is the critical "safety net" scenario. The user has signaled their intent to save a
        checkpoint by providing a path, but has not configured the means to do so.

        Expectation: The function should detect the mismatch and automatically add a new
                     `ModelCheckpoint` callback that saves to the specified path.
        """
        ckpt_path = tmp_path / "checkpoints" / "safety_net.ckpt"

        trainer = BoringTrainer(callbacks=[], default_root_dir=str(tmp_path))

        initial_callback_count = len(trainer.callbacks)

        configure_checkpointing(trainer, ckpt_path)

        assert len(trainer.callbacks) == initial_callback_count + 1
        new_callback = trainer.callbacks[-1]
        assert isinstance(new_callback, ModelCheckpoint)
        assert Path(new_callback.dirpath).resolve() == ckpt_path.parent.resolve()
        assert new_callback.filename == ckpt_path.with_suffix("").name

    def test_case_3_no_checkpointing_but_callback(self, tmp_path: Path):
        """Tests Case 3: The user provides a `ModelCheckpoint` callback but no `ckpt_path`.

        In this scenario, the user is managing their own checkpointing (e.g., saving to a
        logger-defined directory) and has not asked the Manager to handle a specific resume path.

        Expectation: The function should respect the user's setup and make no changes.
        """
        user_dir = tmp_path / "user_checkpoints"
        callbacks = [ModelCheckpoint(dirpath=str(user_dir))]
        trainer = BoringTrainer(callbacks=callbacks, default_root_dir=str(tmp_path))

        initial_callback_count = len(trainer.callbacks)

        # ckpt_path is None, simulating the user not providing it to the Manager
        configure_checkpointing(trainer, None)

        assert len(trainer.callbacks) == initial_callback_count
        assert Path(trainer.callbacks[-1].dirpath).resolve() == user_dir.resolve()

    def test_case_4_no_checkpointing_no_callback(self, tmp_path: Path):
        """Tests Case 4: The user provides no `ckpt_path` and no `ModelCheckpoint` callback.

        This represents the user's intent to run a session without saving any model checkpoints.

        Expectation: The function should do nothing and the trainer should have no
                     `ModelCheckpoint` callbacks.
        """
        trainer = BoringTrainer(callbacks=[], default_root_dir=str(tmp_path))

        initial_callback_count = len(trainer.callbacks)

        configure_checkpointing(trainer, None)

        assert len(trainer.callbacks) == initial_callback_count
