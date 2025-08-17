import math

import lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import torchvision
from lightning.pytorch.loggers import WandbLogger
from timm.models.vision_transformer import VisionTransformer
from torch import nn

import stable_ssl as ssl
from stable_ssl.data import transforms
from stable_ssl.data.utils import Dataset
from stable_ssl.utils.pos_embed import get_2d_sincos_pos_embed


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
height, width, patch_size = 32, 32, 4
crop_height, crop_width = 32, 32
num_patches = (height // patch_size) * (width // patch_size)
patch_channel_dim = 3 * patch_size * patch_size
mask_ratio = 0.75
num_visible_patches = int(num_patches * (1 - mask_ratio))
num_classes = 10
batch_size = 512
val_batch_size = 128

mask_transform_kwargs = dict(
    patch_size=patch_size,
    mask_ratio=mask_ratio,
    source="image",
    target_visible="mask_visible",
    target_masked="mask_masked",
)

train_transform = transforms.Compose(
    transforms.RandomResizedCrop((crop_height, crop_width), scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
    transforms.RandomHorizontalFlip(),
    transforms.RandomMask(**mask_transform_kwargs),
    transforms.ToImage(mean=mean, std=std),
)

val_transform = transforms.Compose(
    transforms.RGB(),
    transforms.Resize((height, width)),
    transforms.CenterCrop((height, width)),
    transforms.ToImage(mean=mean, std=std),
)


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
    """Simpler collate function for MAE - just handle visible/masked indices"""
    visible_indices = [item.pop("mask_visible") for item in batch]
    masked_indices = [item.pop("mask_masked") for item in batch]
    batch = torch.utils.data.default_collate(batch)
    
    # Standardize to minimum length
    min_visible = min(len(vis) for vis in visible_indices)
    min_masked = min(len(mask) for mask in masked_indices)
    
    visible_batch = [vis[:min_visible] for vis in visible_indices]
    masked_batch = [mask[:min_masked] for mask in masked_indices]
    
    batch["mask_visible"] = torch.utils.data.default_collate(visible_batch)
    batch["mask_masked"] = torch.utils.data.default_collate(masked_batch)
    return batch


train_dataset = IndexedDataset(cifar_train, transform=train_transform)
val_dataset = IndexedDataset(cifar_val, transform=val_transform)
train = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=True,
    collate_fn=standardize_masks,
    pin_memory=True,
)

val = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=0,
    shuffle=False,
    pin_memory=True,
)

data = ssl.data.DataModule(train=train, val=val)


def pos_embed(patches: torch.Tensor) -> torch.Tensor:
    return (
        torch.from_numpy(
            get_2d_sincos_pos_embed(patches.shape[-1], int(math.sqrt(patches.shape[1])))
        )
        .to(patches.device)
        .float()
        .repeat(patches.shape[0], 1, 1)
    )


def apply_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply single mask to tensor"""
    B, N, D = x.shape
    mask_expanded = mask.unsqueeze(-1).expand(-1, -1, D)
    return torch.gather(x, dim=1, index=mask_expanded)


class MAE_Encoder(VisionTransformer):
    """MAE encoder - processes only visible patches"""
    
    def __init__(self, *args, **kwargs):
        self.weight_init = kwargs.get("weight_init", "")
        self.fix_init = kwargs.get("fix_init", False)
        mae_in_dim = kwargs.pop("mae_in_dim")
        super().__init__(*args, **kwargs)
        
        self.mae_patch_project = nn.Linear(mae_in_dim, self.embed_dim)
        
        if self.weight_init != "skip":
            self.init_weights(self.weight_init)
        if self.fix_init:
            self.fix_init_weight()

    def patchify(self, image: torch.Tensor) -> torch.Tensor:
        B, C, H, W = image.shape
        P = patch_size
        
        patches = image.unfold(2, P, P).unfold(3, P, P)
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        num_patch_h, num_patch_w = patches.shape[1], patches.shape[2]
        patches = patches.reshape(B, num_patch_h * num_patch_w, P * P * C)
        return patches

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        B, N, P2C = patches.shape
        P = patch_size
        C = patch_channel_dim // (P * P)
        assert P2C == C * P * P
        H = W = (num_patches * P) // (P * P)
        patches = patches.reshape(B, H, W, P, P)
        patches = patches.permute(0, 3, 1, 4, 2) # [B, P, H, P, W]
        return patches.reshape(B, P * P * C, H, W) # [B, P * P * C, H, W]

    def project_patches(self, patches: torch.Tensor) -> torch.Tensor:
        return self.mae_patch_project(patches)

    def encode_patches(self, patches: torch.Tensor) -> torch.Tensor:
        x = self.blocks(patches)
        x = self.norm(x)
        return x

class MAE_Decoder(nn.Module):
    """MAE decoder - reconstructs full image from visible patches + mask tokens"""
    
    def __init__(self, encoder_dim=768, decoder_dim=512, decoder_depth=8, decoder_heads=16):
        super().__init__()
        self.decoder_dim = decoder_dim
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        
        # Decoder transformer blocks
        self.decoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=decoder_dim,
                nhead=decoder_heads,
                dim_feedforward=decoder_dim * 4,
                dropout=0.0,
                activation='gelu',
                batch_first=True
            )
            for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_channel_dim)  # Predict pixel values
        
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        
        # Fixed positional embeddings
        self.decoder_pos_embed = nn.Parameter(
            torch.from_numpy(
                get_2d_sincos_pos_embed(decoder_dim, int(math.sqrt(num_patches)))
            ).float(),
            requires_grad=False
        )

    def forward(self, x_visible: torch.Tensor, mask_visible: torch.Tensor, mask_masked: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_visible: [B, N_visible, encoder_dim] - encoded visible patches
            mask_visible: [B, N_visible] - indices of visible patches
            mask_masked: [B, N_masked] - indices of masked patches
        Returns:
            x_reconstructed: [B, N_masked, patch_pixels] - reconstructed masked patches
        """
        B = x_visible.shape[0]
        N_visible = mask_visible.shape[1]
        N_masked = mask_masked.shape[1]
        
        # Project encoder features to decoder dimension
        x_visible = self.decoder_embed(x_visible)  # [B, N_visible, decoder_dim]
        
        # Create mask tokens for masked positions
        mask_tokens = self.mask_token.expand(B, N_masked, -1)  # [B, N_masked, decoder_dim]
        
        # Get positional embeddings for all patches
        pos_embed_all = self.decoder_pos_embed.unsqueeze(0).expand(B, -1, -1)  # [B, N_total, decoder_dim]
        pos_visible = apply_mask(pos_embed_all, mask_visible)  # [B, N_visible, decoder_dim]
        pos_masked = apply_mask(pos_embed_all, mask_masked)    # [B, N_masked, decoder_dim]
        
        # Add positional embeddings
        x_visible = x_visible + pos_visible
        mask_tokens = mask_tokens + pos_masked
        
        # Combine visible and masked tokens
        # For simplicity, concatenate them (real MAE uses more sophisticated ordering)
        x_full = torch.cat([x_visible, mask_tokens], dim=1)  # [B, N_visible + N_masked, decoder_dim]
        
        # Apply decoder transformer blocks
        for block in self.decoder_blocks:
            x_full = block(x_full)
        
        x_full = self.decoder_norm(x_full)
        
        # Extract predictions for masked patches only
        x_masked_pred = x_full[:, N_visible:]  # [B, N_masked, decoder_dim]
        
        # Predict pixel values
        x_reconstructed = self.decoder_pred(x_masked_pred)  # [B, N_masked, patch_pixels]
        
        return x_reconstructed

encoder_kwargs = dict(
    patch_size=patch_size,
    embed_dim=768,
    depth=12,
    num_heads=12,
    qkv_bias=True,  # MAE typically uses bias
    mae_in_dim=patch_channel_dim,
)

decoder_kwargs = dict(
    encoder_dim=768,
    decoder_dim=512,
    decoder_depth=8,
    decoder_heads=16,
)

def forward(self: ssl.Module, batch, stage):
    out = {}
    encoder: MAE_Encoder = self.encoder
    decoder: MAE_Decoder = self.decoder
    mae_loss: nn.Module  = self.mae_loss
    
    # Patchify and get all patches with positions
    image_patches = encoder.patchify(batch["image"])  # [B, N, patch_pixels]
    all_patches = encoder.project_patches(image_patches)  # [B, N, embed_dim]
    pos_embedding = pos_embed(all_patches)
    all_patches = all_patches + pos_embedding
    
    if not self.training:
        # For validation, encode all patches for downstream tasks
        out["embedding"] = encoder.encode_patches(all_patches)
        out["sum_embedding"] = out["embedding"].sum(dim=1)
        out["flat_embedding"] = out["embedding"].reshape(out["embedding"].shape[0], -1)
        return out
    
    mask_visible, mask_masked = batch["mask_visible"], batch["mask_masked"]
    
    # Encode only visible patches
    visible_patches = apply_mask(all_patches, mask_visible)  # [B, N_visible, embed_dim]
    encoded_visible = encoder.encode_patches(visible_patches)  # [B, N_visible, embed_dim]
    
    # Decode to reconstruct masked patches
    reconstructed_masked = decoder(encoded_visible, mask_visible, mask_masked)  # [B, N_masked, patch_pixels]
    
    # Get ground truth masked patches
    gt_masked_patches = apply_mask(image_patches, mask_masked)  # [B, N_masked, patch_pixels]
    
    # Compute reconstruction loss (MSE in pixel space)
    out["loss"] = mae_loss(reconstructed_masked, gt_masked_patches)
    
    # For monitoring
    out["embedding"]        = encoded_visible
    out["reconstructed"]    = reconstructed_masked
    out["sum_embedding"]    = out["embedding"].sum(dim=1)
    out["flat_embedding"]   = out["embedding"].reshape(out["embedding"].shape[0], -1)
    out["ground_truth"]     = gt_masked_patches
    return out

module = ssl.Module(
    encoder=MAE_Encoder(**encoder_kwargs),
    decoder=MAE_Decoder(**decoder_kwargs),
    forward=forward,
    mae_loss=F.mse_loss,  # Pixel MSE loss
)

# Note: Linear probe uses visible patches only during training
linear_probe = ssl.callbacks.OnlineProbe(
    "linear_probe",
    "sum_embedding",
    "label", 
    probe=torch.nn.Linear(768, num_classes),
    loss_fn=torch.nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(num_classes),
        "top5": torchmetrics.classification.MulticlassAccuracy(num_classes, top_k=5),
    },
)

# RankMe on encoder outputs
rankme = ssl.callbacks.RankMe(
    name="rankme",
    target="flat_embedding", 
    queue_length=min(512, batch_size),
    target_shape=(num_visible_patches, 768), 
)

# Initialize W&B logger
wandb_logger = WandbLogger(
    project="mae-cifar10",
    entity="slightly-more-badass",
    name="mae-cifar10-run",
    log_model=False,
    offline=True,
)

trainer = pl.Trainer(
    max_epochs=6,
    num_sanity_val_steps=0,
    callbacks=[linear_probe, rankme],
    precision="16-mixed",
    logger=wandb_logger,
    enable_checkpointing=False,
    accelerator="gpu",
    devices=1,
    # strategy=DDPStrategy(
    #     find_unused_parameters=True, # this is because only teacher's params are used in the teacher-student module
    #     static_graph=True,
    #     gradient_as_bucket_view=True,
    # )
)

manager = ssl.Manager(trainer=trainer, module=module, data=data)
manager()