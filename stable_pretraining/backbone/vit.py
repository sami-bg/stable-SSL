import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Literal


class PositionalEncoding2D(nn.Module):
    """Flexible 2D positional encoding for vision transformers."""

    def __init__(
        self,
        embed_dim: int,
        grid_size: Tuple[int, int],
        pos_type: Literal["learnable", "sinusoidal", "rope", "none"] = "learnable",
        num_prefix_tokens: int = 1,
        learnable: Optional[
            bool
        ] = None,  # Override: force learnable even for sinusoidal
    ):
        """Positional encoding for 2d input.

        :param embed_dim: Embedding dimension
        :param grid_size: (H, W) grid size in patches
        :param pos_type: Type of positional encoding
        :param num_prefix_tokens: Number of prefix tokens (CLS + registers)
        :param learnable: If True, make sinusoidal learnable; if None, use default

        """
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_h, self.grid_w = grid_size
        self.num_patches = self.grid_h * self.grid_w
        self.pos_type = pos_type
        self.num_prefix_tokens = num_prefix_tokens

        # Override learnable if specified
        if learnable is not None:
            self.is_learnable = learnable
        else:
            self.is_learnable = pos_type == "learnable"

        if pos_type == "none":
            # No positional encoding
            self.pos_embed = None

        elif pos_type == "learnable":
            # Learnable absolute positional embeddings
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_prefix_tokens + self.num_patches, embed_dim)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        elif pos_type == "sinusoidal":
            # 2D sinusoidal positional embeddings
            pos_embed = self._build_sinusoidal_2d(embed_dim, self.grid_h, self.grid_w)

            # Add prefix token positions (zeros or learned separately)
            prefix_pos = torch.zeros(1, num_prefix_tokens, embed_dim)
            pos_embed = torch.cat([prefix_pos, pos_embed], dim=1)

            if self.is_learnable:
                self.pos_embed = nn.Parameter(pos_embed)
            else:
                self.register_buffer("pos_embed", pos_embed)

        elif pos_type == "rope":
            # RoPE doesn't use additive embeddings
            self.pos_embed = None
            # Precompute RoPE frequencies
            self.register_buffer(
                "freqs_h", self._build_rope_freqs(embed_dim // 4, self.grid_h)
            )
            self.register_buffer(
                "freqs_w", self._build_rope_freqs(embed_dim // 4, self.grid_w)
            )
        else:
            raise ValueError(f"Unknown pos_type: {pos_type}")

    def _build_sinusoidal_2d(
        self, embed_dim: int, grid_h: int, grid_w: int
    ) -> torch.Tensor:
        """Build 2D sinusoidal positional embeddings."""
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sinusoidal"

        dim_h = embed_dim // 2
        dim_w = embed_dim // 2

        # Height positions
        pos_h = torch.arange(grid_h).unsqueeze(1)  # [H, 1]
        dim_t_h = torch.arange(0, dim_h, 2).float()  # [dim_h/2]
        omega_h = 1.0 / (10000 ** (dim_t_h / dim_h))

        pos_embed_h = torch.zeros(grid_h, dim_h)
        pos_embed_h[:, 0::2] = torch.sin(pos_h * omega_h)
        pos_embed_h[:, 1::2] = torch.cos(pos_h * omega_h)

        # Width positions
        pos_w = torch.arange(grid_w).unsqueeze(1)  # [W, 1]
        dim_t_w = torch.arange(0, dim_w, 2).float()
        omega_w = 1.0 / (10000 ** (dim_t_w / dim_w))

        pos_embed_w = torch.zeros(grid_w, dim_w)
        pos_embed_w[:, 0::2] = torch.sin(pos_w * omega_w)
        pos_embed_w[:, 1::2] = torch.cos(pos_w * omega_w)

        # Combine: [H, W, D]
        pos_embed_h = pos_embed_h.unsqueeze(1).expand(-1, grid_w, -1)  # [H, W, dim_h]
        pos_embed_w = pos_embed_w.unsqueeze(0).expand(grid_h, -1, -1)  # [H, W, dim_w]

        pos_embed = torch.cat([pos_embed_h, pos_embed_w], dim=-1)  # [H, W, D]
        pos_embed = pos_embed.reshape(1, grid_h * grid_w, embed_dim)  # [1, H*W, D]

        return pos_embed

    def _build_rope_freqs(
        self, dim: int, max_seq_len: int, base: float = 10000.0
    ) -> torch.Tensor:
        """Build RoPE frequency tensor."""
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        pos = torch.arange(max_seq_len)
        freqs = torch.einsum("i,j->ij", pos, inv_freq)  # [seq_len, dim/2]
        freqs = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
        return freqs

    def _apply_rope_2d(self, x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        """Apply 2D RoPE to patch tokens."""
        B, N, D = x.shape

        # Separate prefix and patch tokens
        prefix = x[:, : self.num_prefix_tokens, :]
        patches = x[:, self.num_prefix_tokens :, :]  # [B, H*W, D]

        # Reshape to 2D grid
        patches = patches.reshape(B, grid_h, grid_w, D)

        # Split embedding into 4 parts for 2D RoPE
        d_quarter = D // 4
        x1, x2, x3, x4 = patches.split(d_quarter, dim=-1)

        # Get frequencies (interpolate if needed)
        freqs_h = self.freqs_h[:grid_h, :d_quarter]  # [H, d_quarter]
        freqs_w = self.freqs_w[:grid_w, :d_quarter]  # [W, d_quarter]

        # Apply rotation to height dimension (x1, x2)
        cos_h = torch.cos(freqs_h).unsqueeze(1)  # [H, 1, d_quarter]
        sin_h = torch.sin(freqs_h).unsqueeze(1)  # [H, 1, d_quarter]
        x1_rot = x1 * cos_h - x2 * sin_h
        x2_rot = x1 * sin_h + x2 * cos_h

        # Apply rotation to width dimension (x3, x4)
        cos_w = torch.cos(freqs_w).unsqueeze(0)  # [1, W, d_quarter]
        sin_w = torch.sin(freqs_w).unsqueeze(0)  # [1, W, d_quarter]
        x3_rot = x3 * cos_w - x4 * sin_w
        x4_rot = x3 * sin_w + x4 * cos_w

        # Combine
        patches = torch.cat([x1_rot, x2_rot, x3_rot, x4_rot], dim=-1)
        patches = patches.reshape(B, grid_h * grid_w, D)

        # Recombine with prefix (prefix tokens don't get RoPE)
        return torch.cat([prefix, patches], dim=1)

    def forward(
        self, x: torch.Tensor, grid_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Apply positional encoding.

        :param x: [B, num_prefix + num_patches, D]
        :param grid_size: (H, W) if different from default (for dynamic size)
        :return: x with positional encoding applied
        """
        if self.pos_type == "none":
            return x

        grid_h = grid_size[0] if grid_size else self.grid_h
        grid_w = grid_size[1] if grid_size else self.grid_w

        if self.pos_type == "rope":
            return self._apply_rope_2d(x, grid_h, grid_w)

        # Additive positional embeddings (learnable or sinusoidal)
        pos_embed = self.pos_embed

        # Interpolate if dynamic size
        if grid_h != self.grid_h or grid_w != self.grid_w:
            pos_embed = self._interpolate(pos_embed, grid_h, grid_w)

        return x + pos_embed

    def _interpolate(
        self, pos_embed: torch.Tensor, target_h: int, target_w: int
    ) -> torch.Tensor:
        """Interpolate positional embeddings to new grid size."""
        prefix_pos = pos_embed[:, : self.num_prefix_tokens, :]
        patch_pos = pos_embed[:, self.num_prefix_tokens :, :]

        D = patch_pos.shape[-1]
        patch_pos = patch_pos.reshape(1, self.grid_h, self.grid_w, D).permute(
            0, 3, 1, 2
        )
        patch_pos = F.interpolate(
            patch_pos, size=(target_h, target_w), mode="bicubic", align_corners=False
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, target_h * target_w, D)

        return torch.cat([prefix_pos, patch_pos], dim=1)
