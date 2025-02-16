import math
from functools import cache
from typing import Union

import torch
from einops import rearrange
from torch import nn


#### NOTE All of these are slightly modified versions of what's in the JEPA repo.
def get_3d_sincos_pos_embed(
    embed_dim,
    grid_height,
    grid_width,
    grid_depth,
    cls_token=False,
    uniform_power=False  # weights pos embeddings differently for temporal dimension vs spatial dimensions
):
    """
    grid_size: int of the grid height and width
    grid_depth: int of the grid depth
    returns:
        pos_embed: [grid_depth*grid_size*grid_size, embed_dim] (w/o cls_token)
                or [1+grid_depth*grid_size*grid_size, embed_dim] (w/ cls_token)
    """
    grid_d = torch.arange(grid_depth, dtype=torch.float)
    grid_h = torch.arange(grid_height, dtype=torch.float)
    grid_w = torch.arange(grid_width, dtype=torch.float)
    grid_h, grid_d, grid_w = torch.meshgrid(grid_h, grid_d, grid_w)  # order of meshgrid is very important for indexing as [d,h,w]

    if not uniform_power:
        h_embed_dim = embed_dim // 4
        w_embed_dim = embed_dim // 4
        d_embed_dim = embed_dim // 2
    else:
        h_embed_dim = w_embed_dim = d_embed_dim = int(torch.ceil(torch.tensor(embed_dim/6))*2)

    emb_h = get_1d_sincos_pos_embed_from_grid(h_embed_dim, grid_h)  # (T*H*W, D1)
    emb_w = get_1d_sincos_pos_embed_from_grid(w_embed_dim, grid_w)  # (T*H*W, D2)
    emb_d = get_1d_sincos_pos_embed_from_grid(d_embed_dim, grid_d)  # (T*H*W, D3)
    pos_embed = torch.cat([emb_d, emb_h, emb_w], dim=1)
    pos_embed = pos_embed[:, :embed_dim]
    if cls_token:
        pos_embed = torch.cat([torch.zeros(1, embed_dim), pos_embed], dim=0)
    return pos_embed


def get_2d_sincos_pos_embed(embed_dim: int, grid_H: int, grid_W: int, cls_token=False):
    """
    grid_size: int of the grid height and width
    returns:
        pos_embed: [grid_size*grid_size, embed_dim] (w/o cls_token)
                or [1+grid_size*grid_size, embed_dim] (w/ cls_token)
    """
    grid_h = torch.arange(grid_H, dtype=torch.float)
    grid_w = torch.arange(grid_W, dtype=torch.float)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h)  # order of meshgrid is very important for indexing as [h, w]

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h)  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w)  # (H*W, D/2)
    pos_embed = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)
    if cls_token:
        pos_embed = torch.cat([torch.zeros(1, embed_dim), pos_embed], dim=0)
    return pos_embed

def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    embed_dim: output dimension for each position
    grid_size: int of the grid length
    returns:
        pos_embed: [grid_size, embed_dim] (w/o cls_token)
                or [1+grid_size, embed_dim] (w/ cls_token)
    """
    grid = torch.arange(grid_size, dtype=torch.float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = torch.cat([torch.zeros(1, embed_dim), pos_embed], dim=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    returns: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega   # (D/2,)

    pos = pos.reshape(-1)   # (M,)
    out = torch.einsum('m,d->md', pos, omega)   # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


class Patchify2D(nn.Module):
    """Turn an image into patches of patch_size x patch_size.

    Parameters
    ----------
    patch_size : Union[int, tuple[int, int]], default=16
        Size of patches to extract. Can be either:
        - An integer for square patches (patch_size x patch_size)
        - A tuple of (height, width) for rectangular patches
    positional_encoding : nn.Module, default=PositionalEncoding()
        Positional encoding to apply to the patches

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
        use_pos_embed: bool = True,
    ):
        super().__init__()
        if isinstance(patch_size, tuple):
            assert len(patch_size) == 2
            self.patch_size = patch_size
        if isinstance(patch_size, int):
            self.patch_size = (patch_size,) * 2

        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        self.use_pos_embed = use_pos_embed

    @cache
    def get_pos_embed(self, embed_dim: int, grid_H: int, grid_W: int) -> torch.Tensor:
        # NOTE Converting this to and from numpy could be slow but this is just a PoC for now
        sincos = get_2d_sincos_pos_embed(embed_dim, grid_H, grid_W, cls_token=False)
        return sincos


    # TODO Change this to hwc if unfold allows for it to make it similar to 3d patchify
    def __call__(self, image_CHW: torch.Tensor) -> torch.Tensor:
        C, H, W = image_CHW.shape

        grid_height = H // self.patch_size[0]
        grid_width = W // self.patch_size[1]
        assert (grid_height * self.patch_size[0]) == H and (
            grid_width * self.patch_size[1]
        ) == W

        image_patched_flat: torch.Tensor = self.unfold(image_CHW).T
        mean = image_patched_flat.mean(dim=1, keepdim=True)
        if self.use_pos_embed:
            # NOTE Since positional embeddings are lazily created based on the patched shape
            # and not the input shape, we don't ever need to interpolate them
            D = image_patched_flat.shape[-1]
            pos_embed = self.get_pos_embed(embed_dim=D, grid_H=grid_height, grid_W=grid_width)
            image_patched_flat += pos_embed
        
        mean_after_pos_embed = image_patched_flat.mean(dim=1, keepdim=True)

        # NOTE Unflatten patches to recover original shape
        image_patched: torch.Tensor = rearrange(
            image_patched_flat, "(gh gw) d -> gh gw d", gh=grid_height, gw=grid_width
        )

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
    positional_encoding : nn.Module, default=PositionalEncoding()
        Positional encoding to apply to the patches

    Returns
    -------
    torch.Tensor
        Tensor of tubelets with shape [T, H, W, C] where:
        - T is number of frames (original T/tubelet_size)
        - H,W are spatial grid dimensions (original H,W/patch_size)
        - C is channel dimension (original C * patch_size^2 * tubelet_size)
    """

    def __init__(
        self,
        patch_size: Union[int, tuple[int, int]] = 16,
        tubelet_size: int = 2,
        use_pos_embed: bool = True,
    ):
        super().__init__()
        if isinstance(patch_size, tuple):
            assert len(patch_size) == 2
            self.patch_size = patch_size
        if isinstance(patch_size, int):
            self.patch_size = (patch_size,) * 2

        self.tubelet_size = tubelet_size
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        self.use_pos_embed = use_pos_embed


    @cache
    def get_pos_embed(self, embed_dim: int, grid_H: int, grid_W: int, grid_T: int) -> torch.Tensor:
        # NOTE Converting this to and from numpy could be slow but this is just a PoC for now
        sincos = get_3d_sincos_pos_embed(embed_dim, grid_H, grid_W, grid_T, cls_token=False)
        # sincos = torch.from_numpy(sincos).float()
        # NOTE Will this preserve the values of the positional embeddings
        # NOTE How can I check this? Means over th
        sincos = rearrange(sincos, "(gh gw t) d -> t d (gh gw)", gh=grid_H, gw=grid_W)
        return sincos


    # TODO Change this to thwc
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

        video_patched_flattened: torch.Tensor = self.unfold(video_tubed)
        if self.use_pos_embed:
            D = video_patched_flattened.shape[1]
            pos_embed = self.get_pos_embed(embed_dim=D, grid_H=grid_height, grid_W=grid_width, grid_T=timesteps)
            video_patched_flattened += pos_embed
        
        
        video_patched: torch.Tensor = rearrange(
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
        mask = torch.cat(
            [
                torch.zeros(num_spatial_patches - num_keep_spatial),
                torch.ones(num_keep_spatial),
            ]
        )
        # NOTE Equivalent to np.random.shuffle(mask)
        mask = mask.view(-1)[torch.randperm(mask.nelement())].view(mask.size())
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


class MultiBlock3DMask:
    """Apply multi-block 3D masking to spatiotemporal video data.

    This implements the masking strategy from JEPA, which generates multiple block masks
    with configurable spatial and temporal scales, aspect ratios, and number of blocks.

    Parameters
    ----------
    spatial_scale : tuple[float, float]
        Min and max scale for spatial masking (e.g. (0.2, 0.8))
    temporal_scale : tuple[float, float]
        Min and max scale for temporal masking (e.g. (1.0, 1.0))
    aspect_ratio : tuple[float, float]
        Min and max aspect ratios for blocks (e.g. (0.3, 3.0))
    num_blocks : int
        Number of mask blocks to generate
    max_temporal_keep : float
        Maximum ratio of temporal frames to keep (1.0 = all frames)
    patch_size : Union[tuple[int, int], int]
        Size of patches to mask
    """

    def __init__(
        self,
        spatial_scale: tuple[float, float] = (0.2, 0.8),
        temporal_scale: tuple[float, float] = (1.0, 1.0),
        aspect_ratio: tuple[float, float] = (0.3, 3.0),
        num_blocks: int = 1,
        max_temporal_keep: float = 1.0,
        patch_size: Union[tuple[int, int], int] = (16, 16),
    ):
        if isinstance(patch_size, int):
            self.patch_size = (patch_size,) * 2
        else:
            self.patch_size = patch_size

        self.spatial_scale = spatial_scale
        self.temporal_scale = temporal_scale
        self.aspect_ratio = aspect_ratio
        self.num_blocks = num_blocks
        self.max_temporal_keep = max_temporal_keep

    def _sample_block_size(
        self,
        height: int,
        width: int,
        duration: int,
        generator: torch.Generator,
    ) -> tuple[int, int, int]:
        """Sample a random block size given constraints."""
        min_t, max_t = self.temporal_scale
        t_scale = min_t + torch.rand(1, generator=generator).item() * (max_t - min_t)
        t = max(1, int(duration * t_scale))

        # Sample spatial block size
        min_s, max_s = self.spatial_scale
        s_scale = min_s + torch.rand(1, generator=generator).item() * (max_s - min_s)
        spatial_num_keep = int(height * width * s_scale)

        # Sample aspect ratio
        min_ar, max_ar = self.aspect_ratio
        ar = min_ar + torch.rand(1, generator=generator).item() * (max_ar - min_ar)

        # Calculate block height/width
        h = int(round(math.sqrt(spatial_num_keep * ar)))
        w = int(round(math.sqrt(spatial_num_keep / ar)))
        h = min(h, height)
        w = min(w, width)

        return (t, h, w)

    def _sample_block_mask(
        self,
        block_size: tuple[int, int, int],
        height: int,
        width: int,
        duration: int,
    ) -> torch.Tensor:
        """Generate a single block mask."""
        t, h, w = block_size
        top = torch.randint(0, height - h + 1, (1,))
        left = torch.randint(0, width - w + 1, (1,))
        start = torch.randint(0, duration - t + 1, (1,))

        mask = torch.ones((duration, height, width), dtype=torch.int32)
        mask[start : start + t, top : top + h, left : left + w] = 0

        max_context_duration = max(1, int(duration * self.max_temporal_keep))
        if max_context_duration < duration:
            mask[max_context_duration:, :, :] = 0

        return mask

    def __call__(self, video_thwc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply masking to input video. An image tensor can be passed as well, in which case the mask will interpret it as a single frame.

        Parameters
        ----------
        video_thwc : torch.Tensor
            Input video tensor [T, H, W, C]

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Kept patches [T, N_kept, C] and masked patches [T, N_masked, C]
        """
        # NOTE Add temporal dimension for image
        is_image = video_thwc.ndim == 3
        if video_thwc.ndim == 3:
            video_thwc = video_thwc.unsqueeze(0)

        T, H, W, C = video_thwc.shape
        grid_h = H // self.patch_size[0]
        grid_w = W // self.patch_size[1]

        block_size = self._sample_block_size(grid_h, grid_w, T, torch.Generator())

        mask = torch.ones((T, grid_h, grid_w), dtype=torch.int32)
        for _ in range(self.num_blocks):
            mask *= self._sample_block_mask(block_size, grid_h, grid_w, T)

        mask = mask.flatten()
        mask_discard = torch.argwhere(mask == 0).squeeze()
        mask_keep = torch.nonzero(mask).squeeze()

        video_flat = rearrange(video_thwc, "t h w c -> (t h w) c")

        kept = video_flat[mask_keep].reshape(T, -1, C)
        masked = video_flat[mask_discard].reshape(T, -1, C)

        # remove temporal dimension for image
        if is_image:
            kept = kept.squeeze(0)
            masked = masked.squeeze(0)

        return kept, masked


if __name__ == "__main__":
    tube = TubeMask(0.5, (1, 1))
    mb3d = MultiBlock3DMask((0.2, 0.8), (1.0, 1.0), (0.3, 3.0), 1, 1.0, (1, 1))
    patchify_noembed = Patchify2D(16)
    patchify_embed = Patchify2D(16, use_pos_embed=True)
    patchify_3d_noembed = Patchify3D(16, 2)
    patchify_3d_embed = Patchify3D(16, 2, use_pos_embed=True)

    randvid = torch.randn((16, 3, 224, 224))
    randimg = torch.randn((3, 224, 224))
    # 16, 14, 14, 768
    vidpatch = patchify_3d_noembed(randvid)
    imgpatch = patchify_noembed(randimg)
    # since ratio is 0.5: 16, 98, 768
    vidtubemask = tube(vidpatch)
    imgmb3dmask = mb3d(imgpatch)

    vidpatch_embed = patchify_3d_embed(randvid)
    imgpatch_embed = patchify_embed(randimg)
    x = 1
    posembed_2d = patchify_embed.get_pos_embed(embed_dim=768, grid_H=14, grid_W=14)
    posembed_3d = patchify_3d_embed.get_pos_embed(embed_dim=768, grid_H=14, grid_W=14, grid_T=8)
    import matplotlib.pyplot as plt
    # the 2d embeds should be one image, ignore the channel dim
    posembed_2d_vis = posembed_2d.reshape(14*14, 768).detach().numpy()
    plt.imshow(posembed_2d_vis)
    plt.colorbar()
    plt.savefig("torch_posembed_2d.png")
    for i in range(8):
        plt.figure(figsize=(8, 6))
        posembed_3d_vis = posembed_3d[i,:,:].reshape(14*14, 768).detach().numpy()
        plt.imshow(posembed_3d_vis)
        plt.axis("off")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f"torch_posembed_3d_frame{i}.png")
        plt.close()
