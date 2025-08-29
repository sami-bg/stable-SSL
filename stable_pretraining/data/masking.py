import math
import torch

def _sample_block_size(
    height: int,
    width: int,
    min_scale: float,
    max_scale: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
):
    """Sample a single block mask for an image.

    Args:
        height (int): Height of the image in patches.
        width (int): Width of the image in patches.
        min_scale (float): Minimum scale factor for block area relative to total image area.
        max_scale (float): Maximum scale factor for block area relative to total image area.
        min_aspect_ratio (float): Minimum aspect ratio (height/width) for the block.
        max_aspect_ratio (float): Maximum aspect ratio (height/width) for the block.

    Returns:
        tuple[int, int]: A tuple (h, w) containing the sampled block height and width.
    """
    _rand = torch.rand(1).item()
    mask_scale = min_scale + _rand * (max_scale - min_scale)
    max_keep = int(height * width * mask_scale)
    aspect_ratio = min_aspect_ratio + _rand * (max_aspect_ratio - min_aspect_ratio)

    # Compute block height and width (given scale and aspect-ratio)
    h = int(round(math.sqrt(max_keep * aspect_ratio)))
    h = min(h, height - 1)

    w = int(round(math.sqrt(max_keep / aspect_ratio)))
    w = min(w, width - 1)

    return (h, w)


def _sample_block_mask(
    image_size: tuple[int, int],
    block_size: tuple[int, int],
    min_keep: int = 1,
):
    """Sample a single block mask for an image.
    Because mask positions are chosen randomly, we can occasionally end up with a mask that is too small.
    This function will retry until a valid mask is found.

    Args:
        image_size: Tuple[int, int] - Size of the image in patches
        block_size: Tuple[int, int] - Size of the block in patches
        min_keep (int): Minimum number of patches that must be in the block.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - mask: Binary tensor indices of patches exposed to encoder (1 = visible, 0 = masked)
            - pred_mask: Binary tensor where of combined target block masks to be predicted (1 = visible, 0 = masked)
    """
    h, w = block_size
    height, width = image_size
    max_attempts = 20

    for _ in range(max_attempts):
        top = torch.randint(0, height - h + 1, (1,)).item()
        left = torch.randint(0, width - w + 1, (1,)).item()

        mask = torch.zeros((height, width), dtype=torch.int32)
        mask[top : top + h, left : left + w] = 1

        # Return the first mask that satisfies min_keep.
        if torch.sum(mask) >= min_keep:
            return mask

    # If we run out of attempts, return whatever we had last.
    else:
        return mask


def multi_block_mask(
    height: int,
    width: int,
    block_scales: list[tuple[float, float]] = [(0.85, 1.0), *((0.15, 0.2),) * 4],
    aspect_ratios: list[tuple[float, float]] = [(1.0, 1.0), *((0.75, 1.5),) * 4],
    min_keep: int = 1,
    seed: int = 0,
) -> list[torch.Tensor, ...]:
    g = torch.Generator()
    g.manual_seed(seed)

    # mapping from unique combinations of size x aspect ratio to the block size.
    block_scale_to_size = {
        (scale, ratio): _sample_block_size(
            height, width, scale[0], scale[1], ratio[0], ratio[1]
        )
        for scale, ratio in set(zip(block_scales, aspect_ratios))
    }

    masks: list[torch.Tensor] = [
        _sample_block_mask(
            (height, width), block_scale_to_size[((sh, sw), (ah, aw))], min_keep
        )
        for (sh, sw), (ah, aw) in zip(block_scales, aspect_ratios)
    ]
    # -- Return binary masks
    return masks
