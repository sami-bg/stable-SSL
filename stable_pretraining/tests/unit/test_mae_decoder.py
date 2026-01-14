"""Unit tests for MAE decoder."""

import pytest
import torch

from stable_pretraining.backbone import MAEDecoder


@pytest.mark.unit
class TestMAEDecoderInit:
    """Test MAE decoder."""

    def test_default_initialization(self):
        decoder = MAEDecoder(num_patches=196)
        assert decoder is not None

    def test_custom_dimensions(self):
        decoder = MAEDecoder(
            embed_dim=512,
            decoder_embed_dim=256,
            output_dim=768,
            num_patches=64,
            depth=4,
            num_heads=8,
        )
        assert decoder.decoder_embed.in_features == 512
        assert decoder.decoder_embed.out_features == 256
        assert decoder.decoder_pred.in_features == 256
        assert decoder.decoder_pred.out_features == 768
        assert len(decoder.blocks) == 4

    def test_mask_token_is_learnable(self):
        decoder = MAEDecoder(decoder_embed_dim=512, num_patches=196)
        assert isinstance(decoder.mask_token, torch.nn.Parameter)
        assert decoder.mask_token.requires_grad
        assert decoder.mask_token.shape == (1, 1, 512)

    def test_sincos_1d_pos_embed_is_buffer(self):
        decoder = MAEDecoder(num_patches=100, pos_embed_type="sincos_1d")
        assert "pos_embed" in dict(decoder.named_buffers())
        assert not decoder.pos_embed.requires_grad

    def test_sincos_2d_pos_embed_is_buffer(self):
        decoder = MAEDecoder(num_patches=49, pos_embed_type="sincos_2d", grid_size=7)
        assert "pos_embed" in dict(decoder.named_buffers())
        assert not decoder.pos_embed.requires_grad
        assert decoder.pos_embed.shape == (1, 49, 512)

    def test_learned_pos_embed_is_parameter(self):
        decoder = MAEDecoder(num_patches=100, pos_embed_type="learned")
        assert "pos_embed" in dict(decoder.named_parameters())
        assert decoder.pos_embed.requires_grad


@pytest.mark.unit
class TestMAEDecoderForward:
    """Test MAE decoder."""

    @pytest.fixture
    def decoder(self):
        return MAEDecoder(
            embed_dim=64,
            decoder_embed_dim=32,
            output_dim=64,  # Explicitly set to match expected output
            num_patches=16,
            depth=2,
            num_heads=4,
            pos_embed_type="sincos_1d",  # Use 1d to avoid grid_size requirement
        )

    def test_output_shape(self, decoder):
        N, T, D = 2, 16, 64
        T_visible = 4

        x = torch.randn(N, T_visible, D)
        mask = torch.ones(N, T)
        mask[:, :T_visible] = 0

        out = decoder(x, mask)
        assert out.shape == (N, T, D)  # D = output_dim

    def test_output_dim_differs_from_embed_dim(self):
        decoder = MAEDecoder(
            embed_dim=64,
            decoder_embed_dim=32,
            output_dim=128,  # Different from embed_dim!
            num_patches=16,
            depth=2,
            num_heads=4,
            pos_embed_type="sincos_1d",
        )

        x = torch.randn(2, 4, 64)  # Input: embed_dim=64
        mask = torch.zeros(2, 16)
        mask[:, 4:] = 1

        out = decoder(x, mask)
        assert out.shape == (2, 16, 128)  # Output: output_dim=128

    def test_output_dtype(self, decoder):
        x = torch.randn(2, 4, 64)
        mask = torch.zeros(2, 16)
        mask[:, 4:] = 1

        out = decoder(x, mask)
        assert out.dtype == torch.float32

    def test_different_mask_ratios(self, decoder):
        N, T, D = 2, 16, 64

        for num_visible in [1, 4, 8, 15]:
            x = torch.randn(N, num_visible, D)
            mask = torch.ones(N, T)
            mask[:, :num_visible] = 0

            out = decoder(x, mask)
            assert out.shape == (N, T, D)

    def test_batch_size_one(self, decoder):
        x = torch.randn(1, 4, 64)
        mask = torch.zeros(1, 16)
        mask[:, 4:] = 1

        out = decoder(x, mask)
        assert out.shape == (1, 16, 64)

    def test_no_masking(self, decoder):
        """All tokens visible (mask all zeros)."""
        N, T, D = 2, 16, 64
        x = torch.randn(N, T, D)
        mask = torch.zeros(N, T)

        out = decoder(x, mask)
        assert out.shape == (N, T, D)

    def test_gradient_flow(self, decoder):
        x = torch.randn(2, 4, 64, requires_grad=True)
        mask = torch.zeros(2, 16)
        mask[:, 4:] = 1

        out = decoder(x, mask)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert decoder.mask_token.grad is not None

    def test_mask_token_affects_output(self, decoder):
        x = torch.randn(1, 4, 64)
        mask = torch.zeros(1, 16)
        mask[:, 4:] = 1

        out1 = decoder(x, mask).detach().clone()

        with torch.no_grad():
            decoder.mask_token.add_(10.0)

        out2 = decoder(x, mask)

        assert not torch.allclose(out1, out2)


@pytest.mark.unit
class TestMAEDecoderPositionalEmbeddings:
    """Test MAE decoder."""

    def test_sincos_1d(self):
        decoder = MAEDecoder(num_patches=100, pos_embed_type="sincos_1d")
        assert decoder.pos_embed.shape == (1, 100, 512)

    def test_sincos_2d(self):
        decoder = MAEDecoder(num_patches=49, pos_embed_type="sincos_2d", grid_size=7)
        assert decoder.pos_embed.shape == (1, 49, 512)

    def test_learned(self):
        decoder = MAEDecoder(num_patches=100, pos_embed_type="learned")
        assert decoder.pos_embed.shape == (1, 100, 512)
        assert decoder.pos_embed.requires_grad


@pytest.mark.unit
class TestMAEDecoderKwargs:
    """Test MAE decoder."""

    def test_qkv_bias(self):
        decoder = MAEDecoder(
            num_patches=16, depth=1, qkv_bias=False, pos_embed_type="sincos_1d"
        )
        x = torch.randn(1, 4, 768)
        mask = torch.zeros(1, 16)
        mask[:, 4:] = 1
        decoder(x, mask)

    def test_drop_path(self):
        decoder = MAEDecoder(
            num_patches=16, depth=2, drop_path=0.1, pos_embed_type="sincos_1d"
        )
        x = torch.randn(1, 4, 768)
        mask = torch.zeros(1, 16)
        mask[:, 4:] = 1
        decoder(x, mask)
