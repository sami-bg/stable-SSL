import pytest
from pathlib import Path
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from stable_pretraining.utils.checkpointing_utils import configure_checkpointing

# A simple mock trainer for testing purposes
class BoringTrainer(pl.Trainer):
    def __init__(self, **kwargs):
        # Disable some default callbacks to make testing more predictable
        kwargs.setdefault("enable_progress_bar", False)
        kwargs.setdefault("enable_model_summary", False)
        super().__init__(**kwargs)


@pytest.mark.unit
class TestConfigureCheckpointing:

    def test_case_1_intentional_ckpt_path_and_callback(self, tmp_path: Path):
        """
        Tests Case 1: User provides a ckpt_path and a matching ModelCheckpoint callback.
        Expectation: No changes to the trainer's callbacks.
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
        """
        Tests Case 2: User provides a ckpt_path but forgets the callback.
        Expectation: A new ModelCheckpoint callback is added automatically.
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
        """
        Tests Case 3: User provides a ModelCheckpoint callback but no ckpt_path.
        Expectation: No changes to the trainer's callbacks.
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
        """
        Tests Case 4: User provides no ckpt_path and no ModelCheckpoint callback.
        Expectation: No changes to the trainer's callbacks.
        """

        trainer = BoringTrainer(callbacks=[], default_root_dir=str(tmp_path))
        
        initial_callback_count = len(trainer.callbacks)
        
        configure_checkpointing(trainer, None)
        
        assert len(trainer.callbacks) == initial_callback_count
