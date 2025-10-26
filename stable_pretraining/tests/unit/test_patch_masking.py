import pytest
import torch
from PIL import Image
import numpy as np

# Assume PatchMasking is defined in patch_masking.py
from stable_pretraining.data.transforms import PatchMasking


@pytest.mark.unit
@pytest.mark.parametrize("input_type", ["pil", "tensor"])
def test_patch_masking_transform(input_type):
    # Create a dummy image (3x32x32)
    np_img = np.ones((32, 32, 3), dtype=np.uint8) * 255
    if input_type == "pil":
        img = Image.fromarray(np_img)
    else:
        img = torch.from_numpy(np_img).permute(2, 0, 1)  # C, H, W

    sample = {"image": img}
    patch_size = 8
    drop_ratio = 0.5
    transform = PatchMasking(
        patch_size=patch_size,
        drop_ratio=drop_ratio,
        source="image",
        target="masked_image",
    )

    out = transform(sample)

    # Check output keys
    assert "masked_image" in out
    assert "patch_mask" in out

    # Check mask shape
    n_patches_h = 32 // patch_size
    n_patches_w = 32 // patch_size
    mask = out["patch_mask"]
    assert mask.shape == (n_patches_h, n_patches_w)
    assert mask.dtype == torch.bool

    # Check that masked_image is still an image of the same size
    masked_img = out["masked_image"]
    if input_type == "pil":
        assert isinstance(masked_img, Image.Image)
        assert masked_img.size == (32, 32)
        masked_img_tensor = (
            torch.from_numpy(np.array(masked_img)).permute(2, 0, 1).float() / 255.0
        )
    else:
        assert isinstance(masked_img, torch.Tensor)
        assert masked_img.shape == (3, 32, 32)
        masked_img_tensor = masked_img

    # Check that at least one patch is masked (all zeros in at least one patch)
    found_masked = False
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            h_start = i * patch_size
            w_start = j * patch_size
            patch = masked_img_tensor[
                :, h_start : h_start + patch_size, w_start : w_start + patch_size
            ]
            if not mask[i, j]:
                assert torch.allclose(patch, torch.zeros_like(patch), atol=1e-3)
                found_masked = True
    assert found_masked, "At least one patch should be masked"


if __name__ == "__main__":
    pytest.main([__file__])
