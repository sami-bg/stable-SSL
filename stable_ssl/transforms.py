from typing import Union

import numpy as np
import torch
from einops import rearrange
from torch import nn


class Patchify2D(nn.Module):
    """Turns an image into patches of patch_size x patch_size."""

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
    """Patchifies a video tensor into tubelets with a certain patch size, similar to 3D convolutions."""

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
        # nn.Unfold expects batchdim x C x *, so we fold the tubelet size into channels and use the timesteps as batchdims
        # 8, 1536, 196 =  # [timesteps, tubelet*C*16*16, grid_h*grid_w]
        video_patched_flattened = self.unfold(video_tubed)
        # 16, 196, 768 = num frames, num patches, (c x patchsize_height x patchsize_width)
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
    """
    WIP: Takes a video tensor with dimensions T, H, W, C, generates a tube mask with a certain ratio of patches to keep, and masks H,W alongside the entirety of T and C.

    BIG NOTE: Currently, this is designed to be completely agnostic to whether the input is patchified. I'm not sure if it's desirable but it felt like it made sense to me, for example:
    - In V-JEPA, we tube-mask the patches, not the raw video.
    - For other approaches, we might want tube-mask the raw video itself.

    Because of this, our input dimensions T,H,W,C can be something like:
    - T=16, H=224, W=224, C=3 for a raw video
    - T=16, H=14, W=14, C=768 for a patchified video, where patch-size is 16x16 (so that 224/16 = 14, and 768 = 3*16*16)
    The output, with a ratio of 0.5, will be:
    - T=16, patches_kept=(0.5*14*14 = 98), C=768. However, the H/W dims have been flattened into one. Not sure if this is desirable or not, but I assume not.
    I suppose we could also reshape them but this gets tricky if the number of patches kept is not a multiple of the original H/W dimensions (e.g. some awkward ratios used)

    ---
    So, for the V-JEPA case, the patch-->tube-masking will be a patchify3D into a tube-mask with a patch-size of 1.
    This is because each of the H/W dimensions are already patchified.

    If we wanted to tube-mask the raw video, we can ignore the patchify3D and just use a patch-size of 16.
    Code is still not super clean and uses a bunch of einops (which could be slow) but is optimized for readability for now :D
    """

    def __init__(
        self,
        ratio: float,
        patch_size: Union[tuple[int, int], int],
        tubelet_size: int,
    ):
        super(TubeMask, self).__init__()
        self.ratio = ratio
        if isinstance(patch_size, int):
            self.patch_size = (patch_size,) * 2

        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

    def sample_mask(
        self, ratio: float, num_spatial_patches: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # NOTE number of patches to keep
        num_keep_spatial = int(num_spatial_patches * (1.0 - ratio))
        mask = np.hstack(
            [
                np.zeros(num_spatial_patches - num_keep_spatial),
                np.ones(num_keep_spatial),
            ]
        )
        np.random.shuffle(mask)
        mask = torch.tensor(mask)
        mask_p = torch.argwhere(mask == 0).squeeze()
        mask_e = torch.nonzero(mask).squeeze()
        return mask_e, mask_p

    def __call__(self, video_thwc: torch.Tensor) -> torch.Tensor:
        T, H, W, C = video_thwc.shape
        num_patches_spatial: int = (H // self.patch_size[0]) * (W // self.patch_size[1])
        # NOTE mask_pred is never used and is always the exact inverse of mask_enc but I kept it here for the time-being to be consistent-ish with the original jepa
        mask_enc, mask_pred = self.sample_mask(self.ratio, num_patches_spatial)
        # tile a patch across the temporal and channel dimensions
        mask_enc, mask_pred = (
            mask_enc.unsqueeze(-1).expand(T, -1, C),
            mask_pred.unsqueeze(-1).expand(T, -1, C),
        )
        # Flatten video across the H,W dimensions
        video_flattened_grid = rearrange(video_thwc, "t h w c -> t (h w) c")
        # since the masks contain indices to keep, we can use gather to apply the masking:
        masked_video_enc = torch.gather(video_flattened_grid, dim=1, index=mask_enc)
        return masked_video_enc


if __name__ == "__main__":
    tube = TubeMask(0.5, (1, 1), 2)
    patchify = Patchify3D(16, 2)
    randvid = torch.randn((16, 3, 224, 224))
    # 16, 14, 14, 768
    vidpatch = patchify(randvid)
    # since ratio is 0.5: 16, 98, 768
    vidtubemask = tube(vidpatch)
