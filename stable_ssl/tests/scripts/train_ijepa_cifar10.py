
import torch
from torch import nn
from typing import Optional
import torchvision
import torch.nn.functional as F

import stable_ssl as ssl
from stable_ssl.data import transforms
from stable_ssl.data.utils import Dataset
from stable_ssl.backbone.utils import TeacherStudentModule

from timm.models.vision_transformer import VisionTransformer


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
patch_size = 4

# Based on the in1k_vith14_ep300.yaml config in the ijepa repository
mask_transform_kwargs = dict(
    patch_size=patch_size,
    num_blocks=4,
    context_scale=(0.85, 1.0),
    target_scale=(0.15, 0.2),
    aspect_ratio=(0.75, 1.5),
    min_keep=20,
)


train_transform = transforms.Compose(
    transforms.RGB(),
    transforms.RandomResizedCrop((32, 32), scale=(0.3, 1.0)),  # CIFAR-10 is 32x32, but on INET, IJEPA uses 224
    transforms.MultiBlockMask(**mask_transform_kwargs),
    transforms.ToImage(mean=mean, std=std),
)
# Don't mask during validation 
val_transform = transforms.Compose(
    transforms.RGB(),
    transforms.Resize((32, 32)),
    transforms.CenterCrop((32, 32)),
    transforms.ToImage(mean=mean, std=std),
)

# Use torchvision CIFAR-10 wrapped in FromTorchDataset
cifar_train = torchvision.datasets.CIFAR10(
    root="/tmp/cifar10", train=True, download=True
)
cifar_val = torchvision.datasets.CIFAR10(
    root="/tmp/cifar10", train=False, download=True
)


class IndexedDataset(Dataset):
    """Custom dataset wrapper that adds sample_idx to each sample."""

    def __init__(self, dataset, transform=None):
        super().__init__(transform)
        self.dataset = dataset

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        sample = {"image": image, "label": label, "sample_idx": idx}
        return self.process_sample(sample)

    def __len__(self):
        return len(self.dataset)


def standardize_masks(batch: list[dict]):

    context_indices = [item.pop('mask_context') for item in batch]
    target_indices  = [item.pop('mask_target')  for item in batch]
    batch           = torch.utils.data.default_collate(batch)

    min_keep_enc    = min(len(ctx) for ctx in context_indices)
    min_keep_pred   = min(
        len(tgt)
        for multiblock  in target_indices
        for tgt         in multiblock
    )
    
    context_batch   = [ctx[:min_keep_enc] for ctx in context_indices]
    target_batch    = [
        [tgt[:min_keep_pred] for tgt in multiblock]
        for multiblock in target_indices
    ]

    collated_masks_context = torch.utils.data.default_collate(context_batch)
    collated_masks_target = torch.utils.data.default_collate(target_batch)

    batch['mask_context']   = collated_masks_context
    batch['mask_target']    = collated_masks_target
    return batch


train_dataset = IndexedDataset(cifar_train, transform=train_transform)
# IJEPA does not use multi-view sampling like SimCLR etc. because it processes
# single views and handles masking at the model level
train = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,  # Regular shuffling, no RepeatedRandomSampler
    num_workers=0,
    drop_last=True,
    collate_fn=standardize_masks
)

"""
{
  'image': torch.Size([64, 3, 32, 32]),
  'label': torch.Size([64]),
  'sample_idx': torch.Size([64]),
  'RandomResizedCrop': torch.Size([64, 4]),
  'mask_context': torch.Size([64, 8, 8]),
  'mask_target': torch.Size([64, 8, 8]),
  'MultiBlockMask': torch.Size([64])
}
"""


val_dataset = IndexedDataset(cifar_val, transform=val_transform)
val = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=128,
    num_workers=0,
    shuffle=True,
)

"""
{
'image': torch.Size([128, 3, 32, 32]),
'label': torch.Size([128]),
'sample_idx': torch.Size([128])
}
"""

data = ssl.data.DataModule(train=train, val=val)


def pos_embed(patches: torch.Tensor) -> torch.Tensor:
    return patches


def patchify(image: torch.Tensor, patch_size: int = patch_size):
    """Convert image tensor into patches.
    
    Args:
        image: Tensor of shape [B, C, H, W]
        
    Returns:
        patches: Tensor of shape [B, N, P*P*C] where:
            N = number of patches (H/P * W/P)
            P = patch size
    """
    B, C, H, W = image.shape
    P = patch_size
    
    # Unfold into patches
    patches = image.unfold(2, P, P).unfold(3, P, P)
    
    # Reshape to [B, num_patches, patch_dim]
    patches = patches.permute(0, 2, 3, 1, 4, 5)
    num_patch_h, num_patch_w = patches.shape[1], patches.shape[2]
    patches = patches.reshape(B, num_patch_h * num_patch_w, P*P*C)

    return patches


def apply_mask(patches: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    patch_dim = patches.shape[-1]
    mask_expanded = mask.unsqueeze(-1).expand(-1,-1,patch_dim)
    masked_patches = torch.gather(patches, dim=1, index=mask_expanded)
    return masked_patches


class IJEPA_Encoder(VisionTransformer):
    def forward_features(self, x, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.blocks(x)
        x = self.norm(x)
        return x


class IJEPA_Predictor(VisionTransformer):
    # TODO 
    def forward_features(self, x, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.blocks(x)
        x = self.norm(x)
        return x


encoder_kwargs      = dict(patch_size=4, embed_dim=768, depth=12, num_heads=12, qkv_bias=False)
context_encoder     = IJEPA_Encoder(**encoder_kwargs)
target_encoder      = TeacherStudentModule(context_encoder)
predictor_kwargs    = dict(patch_size=4, embed_dim=384, depth=6, num_heads=6, qkv_bias=False)
predictor           = IJEPA_Predictor(**predictor_kwargs)


def forward(self: ssl.Module, batch, stage):
    out = {}
    if self.training:
        mask_context, mask_target   = batch['mask_context'], batch['mask_target']
        image_patches               = patchify(batch['image'])
        pos_embedding               = pos_embed(image_patches)
        target_patches              = self.target_encoder.patch_project(image_patches)
        # target encoder is applied on full patches, then masked
        out['target_patches']       = apply_mask(
            self.target_encoder(target_patches) + pos_embedding,
            mask_target
        )
        # context encoder is applied on masked patches
        context_patches        = self.context_encoder.patch_project(apply_mask(image_patches, mask_context))
        context_pos            = apply_mask(pos_embedding, mask_context)
        out['context_patches'] = self.context_encoder(context_patches + context_pos)

        out['predicted_patches'] = self.predictor(
            out['context_patches'],
            mask_context,
            mask_target,
        )
        out['loss'] = self.ijepa_loss(
            apply_mask(out['predicted_patches'], mask_target),
            out['target_patches']
        )
    else:
        image_patches = patchify(batch['image'])
        patches       = self.target_encoder.patch_project(image_patches)
        return          self.target_encoder(patches)


module = ssl.Module(
    context_encoder=context_encoder,
    target_encoder=target_encoder,
    predictor=predictor,
    forward=forward,
    ijepa_loss=F.mse_loss,
)



if __name__ == "__main__":
    train_iter = iter(train)
    val_iter = iter(val)
    for batch in train_iter:
        print({k:v.shape for k,v in batch.items()})
        break
    for batch in val_iter:
        print({k:v.shape for k,v in batch.items()})
        break


# def ijepa_forward(self: ssl.Module, batch: dict) -> dict:
#   mask_keep, mask_discard = batch['mask'], 1-batch['mask']
#   batch['tgt_patches']    = self.tgt_enc(mask_discard & batch['images'])
#   batch['ctx_patches']    = self.ctx_enc(mask_keep    & batch['images'])
#   batch['pred_patches']   = self.pred   (mask_keep, batch['ctx_patches'])
#   # -- calculate loss
#   batch['loss']           = self.ijepa_loss(batch)
#   return batch


# def custom_ijepa_loss(self: ssl.Module, batch: dict) -> Tensor:
#   return F.mse(batch['pred_patches'], batch['tgt_patches']) + self.vicreg_loss(batch['pred_patches'])
