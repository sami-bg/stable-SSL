"""Unit tests for BarlowTwinsLoss.

Focus on the ``feature_dim`` argument added for FSDP compatibility:
:class:`torch.nn.LazyBatchNorm1d` materializes parameters on first forward,
which is incompatible with FSDP's flat-param assumption established at wrap
time. Passing ``feature_dim`` switches to eager :class:`torch.nn.BatchNorm1d`.
"""

from __future__ import annotations

import pytest
import torch

from stable_pretraining.losses import BarlowTwinsLoss


pytestmark = pytest.mark.unit


def test_default_uses_lazy_bn():
    """Default constructor preserves the lazy BN for backwards compatibility."""
    loss = BarlowTwinsLoss()
    assert isinstance(loss.bn, torch.nn.LazyBatchNorm1d)
    # Lazy BN has no materialized weight before first forward.
    assert isinstance(loss.bn.weight, torch.nn.parameter.UninitializedParameter)


def test_feature_dim_uses_eager_bn():
    """Passing feature_dim materializes BN eagerly (FSDP-compatible)."""
    loss = BarlowTwinsLoss(feature_dim=32)
    assert isinstance(loss.bn, torch.nn.BatchNorm1d)
    assert not isinstance(loss.bn, torch.nn.LazyBatchNorm1d)
    # Eager BN has weights and buffers ready before any forward call.
    assert loss.bn.weight.shape == (32,)
    assert loss.bn.running_mean.shape == (32,)
    assert loss.bn.running_var.shape == (32,)


def test_eager_bn_matches_lazy_bn_after_first_forward():
    """Numerical equivalence: with the same init, eager and lazy BN produce
    identical losses on the first forward."""
    torch.manual_seed(0)
    z_i = torch.randn(8, 16, requires_grad=True)
    z_j = torch.randn(8, 16, requires_grad=True)

    loss_lazy = BarlowTwinsLoss()
    out_lazy = loss_lazy(z_i, z_j)

    # Force the eager BN to start with the same init as the freshly-materialized
    # lazy BN (both default-initialize weight=1, bias=0, running stats=0/1).
    loss_eager = BarlowTwinsLoss(feature_dim=16)
    out_eager = loss_eager(z_i, z_j)

    assert torch.allclose(out_lazy, out_eager, atol=1e-6, rtol=1e-6)


def test_eager_bn_finite_loss_and_grads():
    """Sanity: eager-BN BarlowTwins runs forward+backward and produces finite
    loss and gradients."""
    torch.manual_seed(0)
    z_i = torch.randn(8, 16, requires_grad=True)
    z_j = torch.randn(8, 16, requires_grad=True)

    loss = BarlowTwinsLoss(feature_dim=16)
    out = loss(z_i, z_j)
    assert out.ndim == 0
    assert torch.isfinite(out)

    out.backward()
    assert torch.isfinite(z_i.grad).all()
    assert torch.isfinite(z_j.grad).all()


def test_feature_dim_mismatch_raises_at_forward():
    """If feature_dim doesn't match input dim, BN raises (no silent corruption)."""
    loss = BarlowTwinsLoss(feature_dim=8)  # wrong: input is 16-dim below
    z = torch.randn(4, 16)
    with pytest.raises(RuntimeError):
        loss(z, z)
