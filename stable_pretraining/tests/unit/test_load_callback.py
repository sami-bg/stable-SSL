"""Unit tests for StrictCheckpointCallback.

Run with: pytest test_strict_checkpoint_callback.py -v
Run only unit tests: pytest test_strict_checkpoint_callback.py -v -m unit
"""

from typing import Any, Dict
import lightning.pytorch as pl
import torch
import torch.nn as nn
import pytest
from unittest.mock import Mock
import copy
from stable_pretraining.callbacks import StrictCheckpointCallback


# ============================================================================
# TEST FIXTURES AND HELPERS
# ============================================================================


def compare_state_dicts(
    state_dict1: Dict[str, torch.Tensor], state_dict2: Dict[str, torch.Tensor]
) -> bool:
    """Compare two state dictionaries containing tensors.

    Args:
        state_dict1: First state dictionary
        state_dict2: Second state dictionary

    Returns:
        True if state dicts are equal, False otherwise
    """
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        return False

    for key in state_dict1.keys():
        if not torch.equal(state_dict1[key], state_dict2[key]):
            return False

    return True


def compare_checkpoints(
    checkpoint1: Dict[str, Any], checkpoint2: Dict[str, Any]
) -> bool:
    """Compare two checkpoint dictionaries.

    Args:
        checkpoint1: First checkpoint
        checkpoint2: Second checkpoint

    Returns:
        True if checkpoints are equal, False otherwise
    """
    if set(checkpoint1.keys()) != set(checkpoint2.keys()):
        return False

    for key in checkpoint1.keys():
        if key == "state_dict":
            if not compare_state_dicts(checkpoint1[key], checkpoint2[key]):
                return False
        elif isinstance(checkpoint1[key], torch.Tensor):
            if not torch.equal(checkpoint1[key], checkpoint2[key]):
                return False
        else:
            if checkpoint1[key] != checkpoint2[key]:
                return False

    return True


class SimpleModel(pl.LightningModule):
    """A simple model for testing."""

    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        return self.layer2(x)


# ============================================================================
# UNIT TESTS
# ============================================================================


@pytest.mark.unit
class TestStrictCheckpointCallback:
    """Unit tests for StrictCheckpointCallback."""

    @pytest.fixture
    def mock_trainer(self):
        """Create a mock trainer."""
        return Mock(spec=pl.Trainer)

    @pytest.fixture
    def simple_model(self):
        """Create a simple model."""
        return SimpleModel()

    @pytest.fixture
    def perfect_checkpoint(self, simple_model):
        """Create a checkpoint that perfectly matches the model."""
        return {
            "state_dict": copy.deepcopy(simple_model.state_dict()),
            "optimizer_states": [{"state": {}}],
            "lr_schedulers": [{"scheduler": {}}],
        }

    def test_initialization_strict_true(self):
        """Test callback initialization with strict=True."""
        callback = StrictCheckpointCallback(strict=True)
        assert callback.strict is True

    def test_initialization_strict_false(self):
        """Test callback initialization with strict=False."""
        callback = StrictCheckpointCallback(strict=False)
        assert callback.strict is False

    def test_initialization_default(self):
        """Test callback initialization with default parameters."""
        callback = StrictCheckpointCallback()
        assert callback.strict is True

    def test_strict_true_no_modification(
        self, mock_trainer, simple_model, perfect_checkpoint
    ):
        """Test that strict=True doesn't modify the checkpoint."""
        callback = StrictCheckpointCallback(strict=True)
        original_checkpoint = copy.deepcopy(perfect_checkpoint)

        callback.on_load_checkpoint(mock_trainer, simple_model, perfect_checkpoint)

        # Checkpoint should remain unchanged
        assert compare_checkpoints(perfect_checkpoint, original_checkpoint)

    def test_strict_false_perfect_match(
        self, mock_trainer, simple_model, perfect_checkpoint
    ):
        """Test strict=False with perfectly matching checkpoint."""
        callback = StrictCheckpointCallback(strict=False)

        callback.on_load_checkpoint(mock_trainer, simple_model, perfect_checkpoint)

        # All keys should be preserved
        assert len(perfect_checkpoint["state_dict"]) == len(simple_model.state_dict())
        # Optimizer states should remain since there are no mismatches
        assert "optimizer_states" in perfect_checkpoint
        assert "lr_schedulers" in perfect_checkpoint

    def test_missing_keys_in_checkpoint(self, mock_trainer, simple_model):
        """Test handling of missing keys in checkpoint."""
        callback = StrictCheckpointCallback(strict=False)

        # Checkpoint missing layer2 parameters
        checkpoint = {
            "state_dict": {
                "layer1.weight": torch.randn(20, 10),
                "layer1.bias": torch.randn(20),
            },
            "optimizer_states": [{"state": {}}],
        }

        callback.on_load_checkpoint(mock_trainer, simple_model, checkpoint)

        # Only layer1 parameters should be in checkpoint
        assert "layer1.weight" in checkpoint["state_dict"]
        assert "layer1.bias" in checkpoint["state_dict"]
        assert "layer2.weight" not in checkpoint["state_dict"]
        assert "layer2.bias" not in checkpoint["state_dict"]

        # Optimizer states should be cleared
        assert "optimizer_states" not in checkpoint

    def test_extra_keys_in_checkpoint(
        self, mock_trainer, simple_model, perfect_checkpoint
    ):
        """Test handling of extra keys in checkpoint that don't exist in model."""
        callback = StrictCheckpointCallback(strict=False)

        # Add extra keys to checkpoint
        perfect_checkpoint["state_dict"]["layer3.weight"] = torch.randn(10, 5)
        perfect_checkpoint["state_dict"]["layer3.bias"] = torch.randn(10)

        callback.on_load_checkpoint(mock_trainer, simple_model, perfect_checkpoint)

        # Extra keys should be removed
        assert "layer3.weight" not in perfect_checkpoint["state_dict"]
        assert "layer3.bias" not in perfect_checkpoint["state_dict"]

        # Original keys should still be there
        assert "layer1.weight" in perfect_checkpoint["state_dict"]
        assert "layer2.weight" in perfect_checkpoint["state_dict"]

        # Optimizer states should be cleared due to mismatches
        assert "optimizer_states" not in perfect_checkpoint

    def test_shape_mismatch(self, mock_trainer, simple_model):
        """Test handling of shape mismatches."""
        callback = StrictCheckpointCallback(strict=False)

        checkpoint = {
            "state_dict": {
                "layer1.weight": torch.randn(30, 10),  # Wrong shape: should be (20, 10)
                "layer1.bias": torch.randn(20),
                "layer2.weight": torch.randn(5, 20),
                "layer2.bias": torch.randn(5),
            },
            "optimizer_states": [{"state": {}}],
        }

        callback.on_load_checkpoint(mock_trainer, simple_model, checkpoint)

        # Mismatched key should be removed
        assert "layer1.weight" not in checkpoint["state_dict"]

        # Correctly shaped keys should remain
        assert "layer1.bias" in checkpoint["state_dict"]
        assert "layer2.weight" in checkpoint["state_dict"]
        assert "layer2.bias" in checkpoint["state_dict"]

        # Optimizer states should be cleared
        assert "optimizer_states" not in checkpoint

    def test_no_state_dict_in_checkpoint(self, mock_trainer, simple_model):
        """Test handling when checkpoint has no state_dict."""
        callback = StrictCheckpointCallback(strict=False)

        checkpoint = {"optimizer_states": [{"state": {}}]}

        # Should not raise an error
        callback.on_load_checkpoint(mock_trainer, simple_model, checkpoint)

        # Checkpoint should remain unchanged
        assert "state_dict" not in checkpoint
        assert "optimizer_states" in checkpoint

    def test_optimizer_and_scheduler_clearing(self, mock_trainer, simple_model):
        """Test that optimizer and scheduler states are cleared on mismatches."""
        callback = StrictCheckpointCallback(strict=False)

        checkpoint = {
            "state_dict": {
                "layer1.weight": torch.randn(20, 10),
                "layer1.bias": torch.randn(20),
                "extra_layer.weight": torch.randn(5, 5),  # Extra key
            },
            "optimizer_states": [{"state": {"step": 100}}],
            "lr_schedulers": [{"scheduler": {"last_epoch": 50}}],
            "epoch": 10,
            "global_step": 1000,
        }

        callback.on_load_checkpoint(mock_trainer, simple_model, checkpoint)

        # Optimizer and scheduler states should be cleared
        assert "optimizer_states" not in checkpoint
        assert "lr_schedulers" not in checkpoint

        # Other checkpoint data should remain
        assert checkpoint["epoch"] == 10
        assert checkpoint["global_step"] == 1000

    def test_empty_state_dict(self, mock_trainer, simple_model):
        """Test handling of empty state dict in checkpoint."""
        callback = StrictCheckpointCallback(strict=False)

        checkpoint = {"state_dict": {}, "optimizer_states": [{"state": {}}]}

        callback.on_load_checkpoint(mock_trainer, simple_model, checkpoint)

        # State dict should be empty
        assert len(checkpoint["state_dict"]) == 0

        # Optimizer states should be cleared (due to missing keys)
        assert "optimizer_states" not in checkpoint

    def test_multiple_shape_mismatches(self, mock_trainer, simple_model):
        """Test handling of multiple shape mismatches."""
        callback = StrictCheckpointCallback(strict=False)

        checkpoint = {
            "state_dict": {
                "layer1.weight": torch.randn(30, 15),  # Wrong shape
                "layer1.bias": torch.randn(30),  # Wrong shape
                "layer2.weight": torch.randn(10, 25),  # Wrong shape
                "layer2.bias": torch.randn(10),  # Wrong shape
            }
        }

        callback.on_load_checkpoint(mock_trainer, simple_model, checkpoint)

        # All keys should be filtered out due to shape mismatches
        assert len(checkpoint["state_dict"]) == 0

    def test_partial_match(self, mock_trainer, simple_model):
        """Test partial matching of checkpoint to model."""
        callback = StrictCheckpointCallback(strict=False)

        checkpoint = {
            "state_dict": {
                "layer1.weight": torch.randn(20, 10),  # Correct
                "layer1.bias": torch.randn(20),  # Correct
                # layer2 parameters missing
                "extra_layer.weight": torch.randn(5, 5),  # Extra
            },
            "optimizer_states": [{"state": {}}],
        }

        callback.on_load_checkpoint(mock_trainer, simple_model, checkpoint)

        # Only matching keys should remain
        assert "layer1.weight" in checkpoint["state_dict"]
        assert "layer1.bias" in checkpoint["state_dict"]
        assert "layer2.weight" not in checkpoint["state_dict"]
        assert "extra_layer.weight" not in checkpoint["state_dict"]

        # Should have 2 out of 4 parameters
        assert len(checkpoint["state_dict"]) == 2

    def test_large_number_of_mismatches_logging(self, mock_trainer, simple_model):
        """Test logging behavior with many mismatches (>10)."""
        callback = StrictCheckpointCallback(strict=False)

        # Create checkpoint with many extra keys
        checkpoint_state = simple_model.state_dict().copy()
        for i in range(15):
            checkpoint_state[f"extra_layer_{i}.weight"] = torch.randn(5, 5)

        checkpoint = {
            "state_dict": checkpoint_state,
            "optimizer_states": [{"state": {}}],
        }

        # Should not raise an error even with many mismatches
        callback.on_load_checkpoint(mock_trainer, simple_model, checkpoint)

        # Extra keys should be removed
        for i in range(15):
            assert f"extra_layer_{i}.weight" not in checkpoint["state_dict"]

        # Original keys should remain
        assert "layer1.weight" in checkpoint["state_dict"]
        assert "layer2.weight" in checkpoint["state_dict"]

    def test_checkpoint_modification_in_place(self, mock_trainer, simple_model):
        """Test that checkpoint is modified in-place."""
        callback = StrictCheckpointCallback(strict=False)

        checkpoint = {
            "state_dict": {
                "layer1.weight": torch.randn(20, 10),
                "extra_key": torch.randn(5, 5),
            }
        }

        # Keep reference to original checkpoint object
        checkpoint_id = id(checkpoint)

        callback.on_load_checkpoint(mock_trainer, simple_model, checkpoint)

        # Should be the same object (modified in-place)
        assert id(checkpoint) == checkpoint_id

        # But contents should be different
        assert "extra_key" not in checkpoint["state_dict"]


@pytest.mark.unit
class TestStrictCheckpointCallbackIntegration:
    """Integration-style unit tests for StrictCheckpointCallback."""

    def test_workflow_fine_tuning_smaller_model(self):
        """Test workflow for fine-tuning with a smaller model."""
        # Original larger model
        large_model = SimpleModel(input_size=10, hidden_size=50, output_size=5)
        large_checkpoint = {
            "state_dict": large_model.state_dict(),
            "optimizer_states": [{"state": {}}],
        }

        # Smaller model for fine-tuning
        small_model = SimpleModel(input_size=10, hidden_size=20, output_size=5)

        callback = StrictCheckpointCallback(strict=False)
        mock_trainer = Mock(spec=pl.Trainer)

        # Should handle the shape mismatch gracefully
        callback.on_load_checkpoint(mock_trainer, small_model, large_checkpoint)

        # Only matching shapes should be loaded (biases might match if sizes align)
        assert "optimizer_states" not in large_checkpoint

    def test_workflow_adding_new_layers(self):
        """Test workflow when new layers are added to the model."""
        # Original model with 2 layers
        old_model = SimpleModel(input_size=10, hidden_size=20, output_size=5)
        old_checkpoint = {
            "state_dict": old_model.state_dict(),
            "optimizer_states": [{"state": {}}],
        }

        # New model with additional layer
        class ExtendedModel(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(10, 20)
                self.layer2 = nn.Linear(20, 5)
                self.layer3 = nn.Linear(5, 3)  # New layer

        new_model = ExtendedModel()

        callback = StrictCheckpointCallback(strict=False)
        mock_trainer = Mock(spec=pl.Trainer)

        callback.on_load_checkpoint(mock_trainer, new_model, old_checkpoint)

        # Old layers should be loaded
        assert "layer1.weight" in old_checkpoint["state_dict"]
        assert "layer2.weight" in old_checkpoint["state_dict"]
        # New layer won't be in checkpoint
        assert "layer3.weight" not in old_checkpoint["state_dict"]

        # Optimizer states should be cleared
        assert "optimizer_states" not in old_checkpoint


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])
