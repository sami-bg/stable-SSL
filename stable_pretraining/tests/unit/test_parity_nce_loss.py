
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from stable_pretraining.losses import (
    NTXEntLoss as Old_NTXEntLoss,
    New_NTXEntLoss,
    SymmetricContrastiveLoss as New_InfoNCELoss,
    InfoNCELoss as Old_InfoNCELoss,
)

# ==============================================================================
# Parity Tests
# ==============================================================================

@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("dim", [128, 256])
def test_ntxent_parity(batch_size, dim):
    """Verify that the new NTXEntLoss gives the same result as the old one."""
    # ARRANGE
    z_i = torch.randn(batch_size, dim)
    z_j = torch.randn(batch_size, dim)
    
    old_loss_fn = Old_NTXEntLoss(temperature=0.5)
    new_loss_fn = New_NTXEntLoss(temperature=0.5)

    # ACT
    loss_old = old_loss_fn(z_i, z_j)
    loss_new = new_loss_fn(z_i, z_j)

    # ASSERT
    assert torch.allclose(loss_old, loss_new)

@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("dim", [128, 256])
def test_symmetric_contrastive_parity(batch_size, dim):
    """Verify that the new SymmetricContrastiveLoss gives the same result as the old one."""
    # ARRANGE
    feats_i = torch.randn(batch_size, dim)
    feats_j = torch.randn(batch_size, dim)

    old_loss_fn = Old_InfoNCELoss(temperature=0.07)
    new_loss_fn = New_InfoNCELoss(temperature=0.07)

    # ACT
    loss_old = old_loss_fn(feats_i, feats_j)
    loss_new = new_loss_fn(feats_i, feats_j)
    
    # ASSERT
    assert torch.allclose(loss_old, loss_new)
