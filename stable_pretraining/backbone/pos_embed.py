"""Positional embedding utilities for vision transformers."""

import torch
from typing import Literal

__all__ = ["get_sincos_pos_embed", "get_1d_sincos_pos_embed", "get_2d_sincos_pos_embed"]


def get_1d_sincos_pos_embed(
    embed_dim: int, length: int, cls_token: bool = False
) -> torch.Tensor:
    """1D sinusoidal positional embeddings.

    :param embed_dim: Embedding dimension
    :param length: Sequence length
    :param cls_token: Prepend a zero embedding for CLS token
    :return: Shape (length, embed_dim) or (length+1, embed_dim) if cls_token
    """
    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    dim = torch.arange(0, embed_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (dim / embed_dim))

    pe = torch.zeros(length, embed_dim)
    pe[:, 0::2] = torch.sin(pos * inv_freq)
    pe[:, 1::2] = torch.cos(pos * inv_freq)

    if cls_token:
        pe = torch.cat([torch.zeros(1, embed_dim), pe], dim=0)
    return pe


def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: int, cls_token: bool = False
) -> torch.Tensor:
    """2D sinusoidal positional embeddings for image patches (used by MAE encoder).

    :param embed_dim: Embedding dimension (must be divisible by 4)
    :param grid_size: Grid height/width (assumes square)
    :param cls_token: Prepend a zero embedding for CLS token
    :return: Shape (grid_size^2, embed_dim) or (grid_size^2+1, embed_dim)
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"

    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
    grid = torch.stack(grid, dim=-1).reshape(-1, 2)  # (H*W, 2)

    dim = embed_dim // 4
    omega = torch.arange(dim, dtype=torch.float32) / dim
    omega = 1.0 / (10000**omega)  # (D/4,)

    # (H*W, 1) @ (1, D/4) -> (H*W, D/4)
    out_h = grid[:, 0:1] @ omega.unsqueeze(0)
    out_w = grid[:, 1:2] @ omega.unsqueeze(0)

    pe = torch.cat(
        [torch.sin(out_h), torch.cos(out_h), torch.sin(out_w), torch.cos(out_w)], dim=1
    )

    if cls_token:
        pe = torch.cat([torch.zeros(1, embed_dim), pe], dim=0)
    return pe


def get_sincos_pos_embed(
    embed_dim: int,
    num_patches: int,
    mode: Literal["1d", "2d"] = "1d",
    grid_size: int | None = None,
    cls_token: bool = False,
) -> torch.Tensor:
    """Unified interface for sinusoidal positional embeddings.

    :param embed_dim: Embedding dimension
    :param num_patches: Total number of patches (used for 1d)
    :param mode: '1d' or '2d'
    :param grid_size: Required for '2d' mode
    :param cls_token: Prepend CLS token position
    :return: Positional embeddings tensor
    """
    if mode == "2d":
        assert grid_size is not None, "grid_size required for 2d mode"
        return get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token)
    return get_1d_sincos_pos_embed(embed_dim, num_patches, cls_token)
