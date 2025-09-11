import pytest
import torch

from stable_pretraining.data.transforms import ContextTargetsMultiBlockMask
from stable_pretraining.data.utils import apply_masks


@pytest.mark.unit
class TestMasking:
    """Unit tests for multi-block masking and application."""

    def test_multiblock_mask_transform_and_application(self):
        """Tests mask generation, properties, and application to a tensor."""
        # Config
        patch_size = 16
        img_size = 224
        batch_size = 1  # Test with a single item for simplicity
        patch_dim = 128
        num_patches = (img_size // patch_size) ** 2
        num_target_views = 4

        # Create transform, data
        transform = ContextTargetsMultiBlockMask(patch_size=patch_size)
        sample = {"image": torch.randn(3, img_size, img_size)}
        x_patches = torch.randn(batch_size, num_patches, patch_dim)

        # Generate masks for one sample
        transformed_sample = transform(sample)
        context_mask = transformed_sample["mask_context"]
        target_masks = transformed_sample["masks_target"]

        # Apply the generated masks to the patch tensor
        # Note: we apply each mask individually for shape assertions, as K can vary
        masked_context = apply_masks(x_patches, context_mask.unsqueeze(0))
        masked_targets = [apply_masks(x_patches, t.unsqueeze(0)) for t in target_masks]

        # Assertions on the transform output
        assert "mask_context" in transformed_sample
        assert "masks_target" in transformed_sample
        assert isinstance(context_mask, torch.Tensor) and context_mask.ndim == 1
        assert isinstance(target_masks, list) and len(target_masks) == num_target_views

        # Assertions on mask properties
        all_masks = [context_mask] + target_masks
        all_indices = torch.cat(all_masks)
        assert torch.all(all_indices >= 0), "Found negative indices in mask"
        assert torch.all(all_indices < num_patches), "Found out-of-bounds indices"

        # Check that context and targets are disjoint
        context_set = set(context_mask.tolist())
        for i, target_mask in enumerate(target_masks):
            target_set = set(target_mask.tolist())
            assert context_set.isdisjoint(target_set), f"Context intersects target {i}"

        # Assertions on the applied masks (shapes)
        assert masked_context.shape == (batch_size, len(context_mask), patch_dim)
        for i, masked_target in enumerate(masked_targets):
            assert masked_target.shape == (batch_size, len(target_masks[i]), patch_dim)

        # Assertions on the applied masks (values)
        # Pick the first patch from the context view and verify its content
        first_context_idx = context_mask[0].item()
        expected_patch = x_patches[0, first_context_idx, :]
        actual_patch = masked_context[0, 0, :]
        assert torch.equal(expected_patch, actual_patch), "Value mismatch after masking"
