import math
from multiprocessing import Value

import numpy as np
import torch
from torch import nn

# NOTE I have been doing some research on collators and have devised the following
# parent class based on what I've seen:


class Collator(nn.Module):
    """Base class for collators that keeps a synced itr_counter across all processes."""

    def __init__(self):
        self._itr_counter = Value("i", -1)

    def step(self):
        i = self._itr_counter
        with self._itr_counter.get_lock():
            i.value += 1
            v = i.value
        return v

    def __call__(self, batch: list):
        pass


class TubeMaskCollator(Collator):
    """Tube masking collator that extends a spatial mask across all frames."""

    # NOTE: Will write a thorough docstring later, but this is the tubelet strategy
    # as seen is VideoMAE{1, 2}, V-JEPA, and even ViViT
    def __init__(
        self,
        ratio: list[float] | float,
        img_dimensions_hw: tuple[int, int],
        num_frames: int,
        patch_size: tuple[int, int],
        tubelet_size: int,
    ):
        super(TubeMaskCollator, self).__init__()
        if isinstance(ratio, float):
            ratio = [ratio]
        self.ratios = ratio

        self.mask_generators = []
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.duration = num_frames // tubelet_size

        img_height, img_width = img_dimensions_hw
        assert img_height % patch_size[0] == 0 and img_width % patch_size[1] == 0, (
            f"Image dimensions must be cleanly divisible by patch size! "
            f"h: {img_height}/{patch_size[0]} = {img_height / patch_size[0]} "
            f"w: {img_width}/{patch_size[1]}  = {img_width / patch_size[1]} "
        )

        # height/width of grid of patches
        self.grid_height, self.grid_width = (
            img_height // patch_size[0],
            img_width // patch_size[1],
        )
        # number of patches per frame
        self.num_patches_spatial = self.grid_height * self.grid_width

    def sample_mask(self, ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
        # NOTE number of patches to keep
        num_keep_spatial = int(self.num_patches_spatial * (1.0 - ratio))
        mask = np.hstack(
            [
                np.zeros(self.num_patches_spatial - num_keep_spatial),
                np.ones(num_keep_spatial),
            ]
        )
        np.random.shuffle(mask)
        mask = torch.tensor(np.tile(mask, (self.duration, 1)))
        mask = mask.flatten()
        mask_p = torch.argwhere(mask == 0).squeeze()
        mask_e = torch.nonzero(mask).squeeze()
        return mask_e, mask_p

    def generate_masks(
        self, batch_size: int, ratio: float
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        masks_enc, masks_pred = [], []
        for _ in range(batch_size):
            mask_e, mask_p = self.sample_mask(ratio)
            masks_enc.append(mask_e)
            masks_pred.append(mask_p)

        collated_masks_enc = torch.utils.data.default_collate(masks_enc)
        collated_masks_pred = torch.utils.data.default_collate(masks_pred)
        return collated_masks_enc, collated_masks_pred

    def __call__(
        self, batch: list
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        batch_size = len(batch)
        collated_batch = torch.utils.data.default_collate(batch)

        collated_masks_enc = []
        collated_masks_pred = []
        for ratio in self.ratios:
            mask_enc, mask_pred = self.generate_masks(batch_size, ratio)
            collated_masks_enc.append(mask_enc)
            collated_masks_pred.append(mask_pred)

        return collated_batch, collated_masks_enc, collated_masks_pred


class MultiblockMaskCollator(Collator):
    """
    Multiblock masking collator that takes union of multiple smaller masks.

    Can generalize to the video case by extending masks temporally to apply to all frames.
    """

    # TODO Will write a more thorough docstring later.
    # Strategy seen in V-JEPA. Supports both the video and the image case (i.e. tubelet_size=1, num_frames=16)
    def __init__(
        self,
        temporal_scale: list | tuple[float, float] = (1.0, 1.0),
        spatial_scale: list | tuple[float, float] = (0.2, 0.8),
        aspect_ratio: list | tuple[float, float] = (0.3, 3.0),
        num_blocks: list | int = 1,
        max_temporal_keep: list | float = 1.0,
        max_keep: list | int | None = None,
        img_dimensions_hw: tuple[int, int] = (224, 224),
        num_frames: int = 16,
        patch_size: tuple[int, int] = (16, 16),
        tubelet_size: int = 2,
    ):
        super().__init__()

        # they should all be lists to simulate passing in multiple cfg dicts
        self.temporal_scales = (
            temporal_scale if isinstance(temporal_scale, list) else [temporal_scale]
        )
        self.spatial_scales = (
            spatial_scale if isinstance(spatial_scale, list) else [spatial_scale]
        )
        self.aspect_ratios = (
            aspect_ratio if isinstance(aspect_ratio, list) else [aspect_ratio]
        )
        self.num_blocks_list = (
            num_blocks if isinstance(num_blocks, list) else [num_blocks]
        )
        self.max_temporal_keeps = (
            max_temporal_keep
            if isinstance(max_temporal_keep, list)
            else [max_temporal_keep]
        )
        self.max_keeps = max_keep if isinstance(max_keep, list) else [max_keep]

        lengths = [
            len(p)
            for p in (
                self.temporal_scales,
                self.spatial_scales,
                self.aspect_ratios,
                self.num_blocks_list,
                self.max_temporal_keeps,
                self.max_keeps,
            )
        ]

        assert len(set(lengths)) == 1, (
            f"All multiblock mask configurations must have equal length, received {set(lengths)}"
        )

        self.num_masks = len(self.temporal_scales)

        self.img_dimensions = img_dimensions_hw
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

        self.duration = num_frames // tubelet_size
        img_height, img_width = img_dimensions_hw
        assert img_height % patch_size[0] == 0 and img_width % patch_size[1] == 0, (
            f"Image dimensions must be cleanly divisible by patch size! "
            f"h: {img_height}/{patch_size[0]} = {img_height / patch_size[0]} "
            f"w: {img_width}/{patch_size[1]}  = {img_width / patch_size[1]} "
        )

        self.height = img_dimensions_hw[0] // patch_size[0]
        self.width = img_dimensions_hw[1] // patch_size[1]

    def _sample_block_size(
        self, generator: torch.Generator, mask_idx: int
    ) -> tuple[int, int, int]:
        i = mask_idx

        # sample temporal scale
        min_t, max_t = self.temporal_scales[i]
        temporal_mask_scale = min_t + torch.rand(1, generator=generator).item() * (
            max_t - min_t
        )
        t = max(1, int(self.duration * temporal_mask_scale))

        # sample spatial scale
        min_s, max_s = self.spatial_scales[i]
        spatial_mask_scale = min_s + torch.rand(1, generator=generator).item() * (
            max_s - min_s
        )
        spatial_num_keep = int(self.height * self.width * spatial_mask_scale)

        # sample aspect ratio
        min_ar, max_ar = self.aspect_ratios[i]
        ar = min_ar + torch.rand(1, generator=generator).item() * (max_ar - min_ar)

        # Compute block height and width from target area and aspect ratio.
        h = int(round(math.sqrt(spatial_num_keep * ar)))
        w = int(round(math.sqrt(spatial_num_keep / ar)))
        h = min(h, self.height)
        w = min(w, self.width)
        return t, h, w

    def _sample_block_mask(
        self, block_size: tuple[int, int, int], max_context_duration: int
    ) -> torch.Tensor:
        t, h, w = block_size
        start = torch.randint(0, self.duration - t + 1, (1,)).item()
        top = torch.randint(0, self.height - h + 1, (1,)).item()
        left = torch.randint(0, self.width - w + 1, (1,)).item()

        # set the block region to 0 (i.e. mask it).
        mask = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
        mask[start : start + t, top : top + h, left : left + w] = 0

        # optionally limit the context to the first max_context_duration tubelets.
        if max_context_duration < self.duration:
            mask[max_context_duration:, :, :] = 0

        return mask

    def generate_masks(
        self, batch_size: int, mask_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO will seeding here be different or will seed_everything take care of it?
        i = mask_idx
        generator = torch.Generator()

        block_size = self._sample_block_size(generator, mask_idx)
        max_context_duration = max(1, int(self.duration * self.max_temporal_keeps[i]))
        npred = self.num_blocks_list[i]

        masks_enc_list = []
        masks_pred_list = []

        min_keep_enc = self.duration * self.height * self.width
        min_keep_pred = self.duration * self.height * self.width

        for _ in range(batch_size):
            while True:
                mask = torch.ones(
                    (self.duration, self.height, self.width), dtype=torch.int32
                )
                for _ in range(npred):
                    mask *= self._sample_block_mask(block_size, max_context_duration)

                mask_flat = mask.flatten()
                mask_pred = torch.nonzero(mask_flat == 0).squeeze()
                mask_enc = torch.nonzero(mask_flat != 0).squeeze()

                # break on non-empty mask
                if mask_enc.numel() > 0:
                    min_keep_enc = min(min_keep_enc, mask_enc.numel())
                    min_keep_pred = min(min_keep_pred, mask_pred.numel())
                    masks_enc_list.append(mask_enc)
                    masks_pred_list.append(mask_pred)
                    break

        if self.max_keeps[i] is not None:
            min_keep_enc = min(min_keep_enc, self.max_keeps[i])

        # trim each mask to the minimum size across the batch.
        masks_enc_list = [mask[:min_keep_enc] for mask in masks_enc_list]
        masks_pred_list = [mask[:min_keep_pred] for mask in masks_pred_list]

        collated_masks_enc = torch.utils.data.default_collate(masks_enc_list)
        collated_masks_pred = torch.utils.data.default_collate(masks_pred_list)
        return collated_masks_enc, collated_masks_pred

    def __call__(
        self, batch: list
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        batch_size = len(batch)
        collated_batch = torch.utils.data.default_collate(batch)
        collated_masks_enc, collated_masks_pred = [], []
        for i in range(self.num_masks):
            masks_enc, masks_pred = self.generate_masks(batch_size, mask_idx=i)
            collated_masks_enc.append(masks_enc)
            collated_masks_pred.append(masks_pred)

        return collated_batch, collated_masks_enc, collated_masks_pred


if __name__ == "__main__":
    # Testing the collators
    collator = TubeMaskCollator(
        ratio=[0.2, 0.5],
        img_dimensions_hw=(224, 224),
        num_frames=16,
        patch_size=(16, 16),
        tubelet_size=2,
    )
    batch = [torch.randn(3, 16, 224, 224) for _ in range(4)]
    collated_batch, collated_masks_enc, collated_masks_pred = collator(batch)
    print(
        collated_batch.shape, collated_masks_enc[0].shape, collated_masks_pred[0].shape
    )
    # > torch.Size([4, 3, 16, 224, 224]) torch.Size([4, 1248]) torch.Size([4, 320])
    collator = MultiblockMaskCollator(
        temporal_scale=[(0.5, 1.0), (0.5, 1.0)],
        spatial_scale=[(0.2, 0.8), (0.2, 0.8)],
        aspect_ratio=[(0.3, 3.0), (0.3, 3.0)],
        num_blocks=[1, 1],
        max_temporal_keep=[1.0, 1.0],
        max_keep=[None, None],
        img_dimensions_hw=(224, 224),
        num_frames=16,
        patch_size=(16, 16),
        tubelet_size=2,
    )
    batch = [torch.randn(3, 16, 224, 224) for _ in range(4)]
    collated_batch, collated_masks_enc, collated_masks_pred = collator(batch)
    print(
        collated_batch.shape, collated_masks_enc[0].shape, collated_masks_pred[0].shape
    )
    # > torch.Size([4, 3, 16, 224, 224]) torch.Size([4, 853]) torch.Size([4, 715])
