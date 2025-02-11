from typing import Union

import numpy as np
import torch
from einops import rearrange
from torch import nn


class Patchify2D(nn.Module):
    """Turn an image into patches of patch_size x patch_size.

    Parameters
    ----------
    patch_size : Union[int, tuple[int, int]], default=16
        Size of patches to extract. Can be either:
        - An integer for square patches (patch_size x patch_size)
        - A tuple of (height, width) for rectangular patches

    Returns
    -------
    torch.Tensor
        Tensor of patches with shape [N, P, D] where:
        - N is number of patches (H/patch_size_height * W/patch_size_width)
        - P is number of pixels per patch (patch_size_height * patch_size_width)
        - D is the input channel dimension: 3 for RGB images
    """

    def __init__(
        self,
        patch_size: Union[int, tuple[int, int]] = 16,
    ):
        super().__init__()
        if isinstance(patch_size, tuple):
            assert len(patch_size) == 2
            self.patch_size = patch_size
        if isinstance(patch_size, int):
            self.patch_size = (patch_size,) * 2

        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)

    def __call__(self, image_CHW: torch.Tensor) -> torch.Tensor:
        C, H, W = image_CHW.shape

        grid_height = H // self.patch_size[0]
        grid_width = W // self.patch_size[1]
        assert (grid_height * self.patch_size[0]) == H and (
            grid_width * self.patch_size[1]
        ) == W

        image_patched = self.unfold(image_CHW).T
        return image_patched


class Patchify3D(nn.Module):
    """Patchify a video tensor into tubelets with a certain patch size, similar to 3D convolutions.

    This module converts a video tensor into spatiotemporal patches (tubelets) by:
    1. Grouping frames into temporal chunks of size tubelet_size
    2. Within each chunk, extracting spatial patches of size patch_size x patch_size

    Parameters
    ----------
    patch_size : Union[int, tuple[int, int]], default=16
        Size of spatial patches to extract. Can be either:
        - An integer for square patches (patch_size x patch_size)
        - A tuple of (height, width) for rectangular patches
    tubelet_size : int, default=2
        Number of consecutive frames to group into each tubelet

    Returns
    -------
    torch.Tensor
        Tensor of tubelets with shape [T, H, W, C] where:
        - T is number of frames (original T/tubelet_size)
        - H,W are spatial grid dimensions (original H,W/patch_size)
        - C is channel dimension (original C * patch_size^2 * tubelet_size)
    """

    def __init__(
        self, patch_size: Union[int, tuple[int, int]] = 16, tubelet_size: int = 2
    ):
        super().__init__()
        if isinstance(patch_size, tuple):
            assert len(patch_size) == 2
            self.patch_size = patch_size
        if isinstance(patch_size, int):
            self.patch_size = (patch_size,) * 2

        self.tubelet_size = tubelet_size
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)

    def __call__(self, video_TCHW: torch.Tensor) -> torch.Tensor:
        T, C, H, W = video_TCHW.shape

        timesteps: int = T // self.tubelet_size
        assert (timesteps * self.tubelet_size) == T

        grid_height = H // self.patch_size[0]
        grid_width = W // self.patch_size[1]
        assert (grid_height * self.patch_size[0]) == H and (
            grid_width * self.patch_size[1]
        ) == W

        video_tubed = rearrange(
            video_TCHW, "(n t) c h w -> n (t c) h w", n=timesteps, t=self.tubelet_size
        )

        video_patched_flattened = self.unfold(video_tubed)
        video_patched = rearrange(
            video_patched_flattened,
            "n (t c ph pw) (gh gw) -> (n t) gh gw (c ph pw)",
            t=self.tubelet_size,
            c=C,
            gh=grid_height,
            gw=grid_width,
            ph=self.patch_size[0],
            pw=self.patch_size[1],
        )

        return video_patched


class TubeMask:
    """Apply tube masking to spatiotemporal video data by masking aligned spatial patches across time.

    This class implements tube masking as used in V-JEPA and similar architectures. It can handle:
    1. Raw video tensors [T, H, W, C]
    2. Pre-patchified tensors where H,W represent a grid of patches

    For example, given:
    - Raw video: [16, 224, 224, 3]
    - Patchified video: [16, 14, 14, 768] (using 16x16 patches)
    The masking pattern is consistent across the temporal dimension, creating "tubes".

    Parameters
    ----------
    ratio : float
        Ratio of patches to mask out (between 0 and 1)
    patch_size : Union[tuple[int, int], int]
        Size of patches for masking. For pre-patchified input, use (1,1)

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Two tensors containing:
        1. Kept patches with shape [T, N_kept, C]
        2. Masked patches with shape [T, N_masked, C]
        where N_kept + N_masked = H*W/patch_size^2
    """

    def __init__(
        self,
        ratio: float,
        patch_size: Union[tuple[int, int], int],
    ):
        super(TubeMask, self).__init__()
        self.ratio = ratio
        if isinstance(patch_size, int):
            self.patch_size = (patch_size,) * 2

        self.patch_size = patch_size

    def sample_spatial_mask(
        self, ratio: float, num_spatial_patches: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate spatial masking pattern to be applied across temporal dimension.

        Parameters
        ----------
        ratio : float
            Ratio of patches to mask (between 0 and 1)
        num_spatial_patches : int
            Total number of spatial patches (H*W/patch_size^2)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Indices of patches to keep and discard
        """
        num_keep_spatial = int(num_spatial_patches * (1.0 - ratio))
        mask = np.hstack(
            [
                np.zeros(num_spatial_patches - num_keep_spatial),
                np.ones(num_keep_spatial),
            ]
        )
        np.random.shuffle(mask)
        mask = torch.tensor(mask)
        mask_discard = torch.argwhere(mask == 0).squeeze()
        mask_keep = torch.nonzero(mask).squeeze()
        return mask_keep, mask_discard

    def __call__(self, video_thwc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply tube masking to input video.

        Parameters
        ----------
        video_thwc : torch.Tensor
            Input video tensor in [T, H, W, C] format
            Can be either raw video or pre-patchified

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Kept patches and masked patches
            Both have shape [T, N, C] where N varies based on ratio
        """
        T, H, W, C = video_thwc.shape
        num_patches_spatial: int = (H // self.patch_size[0]) * (W // self.patch_size[1])
        mask_keep, mask_discard = self.sample_spatial_mask(
            self.ratio, num_patches_spatial
        )
        mask_keep, mask_discard = (
            mask_keep.unsqueeze(-1).expand(T, -1, C),
            mask_discard.unsqueeze(-1).expand(T, -1, C),
        )
        # Flatten video across the H,W dimensions
        video_flattened_grid = rearrange(video_thwc, "t h w c -> t (h w) c")
        # since the masks contain indices to keep, we can use gather to apply the masking:
        masked_video_keep = torch.gather(video_flattened_grid, dim=1, index=mask_keep)
        masked_video_discard = torch.gather(
            video_flattened_grid, dim=1, index=mask_discard
        )
        return masked_video_keep, masked_video_discard


if __name__ == "__main__":
    tube = TubeMask(0.5, (1, 1), 2)
    patchify = Patchify3D(16, 2)
    randvid = torch.randn((16, 3, 224, 224))
    # 16, 14, 14, 768
    vidpatch = patchify(randvid)
    # since ratio is 0.5: 16, 98, 768
    vidtubemask = tube(vidpatch)
