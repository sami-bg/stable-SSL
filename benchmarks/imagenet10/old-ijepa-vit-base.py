#old-ijepa-vit-base.py
"""I-JEPA (Image-based Joint Embedding Predictive Architecture) on ImageNet-10.

Trains a ViT-Base/16 encoder with an I-JEPA predictor on ImageNette (10-class
ImageNet subset). Hyperparameters follow the I-JEPA paper (Assran et al., 2023)
Table 9 for ViT-Base, adapted for single-GPU training on a smaller dataset.

Paper: https://arxiv.org/abs/2301.08243

Checkpoints are saved every `save_every_n_epochs` epochs for downstream probing.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import sys
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.data.masking import multi_block_mask

try:
    from stable_datasets.images.imagenette import Imagenette
except ImportError:
    print(f'stable_datasets not installed, install with \
        `uv pip install git+https://github.com/galilai-group/stable-datasets.git`')
    exit(1)

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir


# =============================================================================
# Hyperparameters (from I-JEPA paper Table 9 for ViT-Base)
# =============================================================================

IMAGE_SIZE = 224
PATCH_SIZE = 16
GRID_SIZE = IMAGE_SIZE // PATCH_SIZE  # 14
NUM_PATCHES = GRID_SIZE ** 2          # 196
EMBED_DIM = 768                       # ViT-Base hidden dimension

# Predictor (paper: depth=12, width=384 for ViT-Base)
PRED_DEPTH = 12
PRED_DIM = 384
PRED_HEADS = 12

# Optimizer (paper: AdamW, lr=1.5e-4, wd=0.05, betas=(0.9, 0.95))
# Paper uses batch_size=2048 with lr=1.5e-4. We use 256 and scale linearly.
BASE_LR = 1.5e-4
BATCH_SIZE = 256
SCALED_LR = BASE_LR * (BATCH_SIZE / 2048)
WEIGHT_DECAY = 0.05
BETAS = (0.9, 0.95)

# EMA schedule (paper: 0.996 → 1.0 cosine)
BASE_EMA = 0.996
FINAL_EMA = 1.0

# Training
MAX_EPOCHS = 300
WARMUP_EPOCHS = 15
NUM_CLASSES = 10  # ImageNette

# Masking (paper Table 9: 4 target blocks, scale 0.15-0.2, AR 0.75-1.5)
TARGET_SCALES = ((0.15, 0.2),) * 4
TARGET_ASPECT_RATIOS = ((0.75, 1.5),) * 4
CONTEXT_SCALE = (0.85, 1.0)
CONTEXT_ASPECT_RATIO = (1.0, 1.0)

# Checkpointing
SAVE_EVERY_N_EPOCHS = 25
CKPT_DIR = str(Path(__file__).parent / "checkpoints" / "ijepa-vitb")


# =============================================================================
# I-JEPA Forward Function
# =============================================================================

def ijepa_forward(self, batch, stage):
    """I-JEPA forward: predict target patch representations from context patches.

    During training:
        1. Encode full image through teacher (EMA) → target representations
        2. Encode full image through student → context representations
        3. Select context/target patches via multi-block masking
        4. Predictor maps context patches → predicted target patches
        5. Smooth L1 loss between predictions and teacher targets

    During validation:
        Returns mean-pooled patch embeddings from the teacher encoder.
    """
    images = batch["image"]
    B = images.shape[0]
    device = images.device
    out = {}

    # --- Validation: just produce embeddings for probes ---
    if not self.training:
        teacher_output = self.backbone(images)  # defaults to teacher
        if hasattr(teacher_output, "last_hidden_state"):
            patches = teacher_output.last_hidden_state[:, 1:, :]  # skip CLS
        else:
            patches = teacher_output[:, 1:, :] if teacher_output.ndim == 3 else teacher_output
        out["embedding"] = patches.mean(dim=1)
        if "label" in batch:
            out["label"] = batch["label"]
        return out

    # --- Training: I-JEPA ---
    # 1. Generate multi-block masks (shared across the batch for efficiency)
    scales = [CONTEXT_SCALE, *TARGET_SCALES]
    aspect_ratios = [CONTEXT_ASPECT_RATIO, *TARGET_ASPECT_RATIOS]
    context_mask_2d, *target_masks_2d = multi_block_mask(
        GRID_SIZE, GRID_SIZE,
        block_scales=scales,
        aspect_ratios=aspect_ratios,
        min_keep=10,
    )

    # Make context disjoint with all targets
    for tmask in target_masks_2d:
        context_mask_2d = context_mask_2d & (~tmask)

    # Convert 2D masks to flat indices
    context_idx_1d = torch.nonzero(context_mask_2d.flatten()).squeeze(-1).to(device)
    target_idx_list = [
        torch.nonzero(m.flatten()).squeeze(-1).to(device) for m in target_masks_2d
    ]
    target_idx_1d = torch.cat(target_idx_list, dim=0).unique()  # deduplicate overlaps

    N_ctx = context_idx_1d.size(0)
    N_tgt = target_idx_1d.size(0)

    # Expand indices to batch dimension
    context_idx = context_idx_1d.unsqueeze(0).expand(B, -1)  # [B, N_ctx]
    target_idx = target_idx_1d.unsqueeze(0).expand(B, -1)    # [B, N_tgt]

    # 2. Teacher forward (target representations, no gradient)
    teacher_output = self.backbone.forward_teacher(images)
    if hasattr(teacher_output, "last_hidden_state"):
        teacher_patches = teacher_output.last_hidden_state[:, 1:, :]
    else:
        teacher_patches = teacher_output[:, 1:, :]
    # Gather target patch representations
    target_repr = torch.gather(
        teacher_patches, 1,
        target_idx.unsqueeze(-1).expand(-1, -1, teacher_patches.size(-1)),
    )  # [B, N_tgt, D]

    # 3. Student forward (context representations, with gradient)
    student_output = self.backbone.forward_student(images)
    if hasattr(student_output, "last_hidden_state"):
        student_patches = student_output.last_hidden_state[:, 1:, :]
    else:
        student_patches = student_output[:, 1:, :]
    # Gather context patch representations
    context_repr = torch.gather(
        student_patches, 1,
        context_idx.unsqueeze(-1).expand(-1, -1, student_patches.size(-1)),
    )  # [B, N_ctx, D]

    # 4. Predictor: context → predicted targets
    # Create mask tokens as initial query embeddings for target positions
    mask_tokens = self.mask_token.expand(B, N_tgt, -1)

    pred_repr = self.predictor(
        context=context_repr,
        queries=mask_tokens,
        context_idx=context_idx,
        query_idx=target_idx,
    )  # [B, N_tgt, D]

    # 5. Loss: smooth L1 between predicted and teacher target (paper uses smooth L1)
    loss = F.smooth_l1_loss(pred_repr, target_repr.detach())

    out["loss"] = loss
    out["embedding"] = teacher_patches.mean(dim=1).detach()
    if "label" in batch:
        out["label"] = batch["label"]

    self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, sync_dist=True)
    return out


# =============================================================================
# Data
# =============================================================================

# I-JEPA uses only random resized crop + horizontal flip (no color jitter)
train_transform = transforms.Compose(
    transforms.RGB(),
    transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.3, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToImage(**spt.data.static.ImageNet),
)

val_transform = transforms.Compose(
    transforms.RGB(),
    transforms.Resize((256, 256)),
    transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToImage(**spt.data.static.ImageNet),
)

data_dir = get_data_dir("imagenet10")

class _HFDataset(spt.data.Dataset):
    """Per-sample transform wrapper for a pre-loaded HF dataset."""

    def __init__(self, hf_dataset, transform):
        super().__init__(transform)
        self.dataset = hf_dataset

    def __getitem__(self, idx):
        return self.process_sample(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)


_builder = Imagenette(config_name="imagenette", cache_dir=str(data_dir))
_builder.download_and_prepare()

train_dataset = _HFDataset(_builder.as_dataset(split="train"), train_transform)
val_dataset = _HFDataset(_builder.as_dataset(split="test"), val_transform)

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=(num_workers := 4),
    drop_last=True,
    persistent_workers=num_workers > 0,
    shuffle=True,
)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=(num_workers := 4),
    persistent_workers=num_workers > 0,
)

data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)


# =============================================================================
# Model
# =============================================================================

# ViT-Base/16 encoder (student + EMA teacher)
backbone = spt.backbone.vit_hf(
    size="base",
    patch_size=PATCH_SIZE,
    image_size=IMAGE_SIZE,
    pretrained=False,
)

wrapped_backbone = spt.TeacherStudentWrapper(
    backbone,
    warm_init=True,
    base_ema_coefficient=BASE_EMA,
    final_ema_coefficient=FINAL_EMA,
)

# I-JEPA predictor (paper: depth=12, hidden=384, heads=12 for ViT-Base)
predictor = spt.backbone.FlexibleTransformer(
    input_dim=EMBED_DIM,
    hidden_dim=PRED_DIM,
    output_dim=EMBED_DIM,
    num_patches=NUM_PATCHES,
    depth=PRED_DEPTH,
    num_heads=PRED_HEADS,
    mlp_ratio=4.0,
    self_attn=True,
    cross_attn=True,
    use_adaln=False,
    pos_embed_type="sincos_2d",
    grid_size=GRID_SIZE,
    num_prefix_tokens=0,  # no CLS token in context/queries
)

# Learnable mask token (initial representation for target positions)
mask_token = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
nn.init.trunc_normal_(mask_token, std=0.02)


# =============================================================================
# Module + Training
# =============================================================================

module = spt.Module(
    backbone=wrapped_backbone,
    predictor=predictor,
    mask_token=mask_token,
    forward=ijepa_forward,
    optim={
        "optimizer": {
            "type": "AdamW",
            "lr": SCALED_LR,
            "weight_decay": WEIGHT_DECAY,
            "betas": BETAS,
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
        },
        "interval": "epoch",
    },
)

# --- Callbacks ---

teacher_student_callback = spt.callbacks.TeacherStudentCallback(
    update_frequency=1,
    update_after_backward=False,
)

linear_probe = spt.callbacks.OnlineProbe(
    module,
    name="linear_probe",
    input="embedding",
    target="label",
    probe=nn.Linear(EMBED_DIM, NUM_CLASSES),
    loss=nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(NUM_CLASSES),
        "top5": torchmetrics.classification.MulticlassAccuracy(NUM_CLASSES, top_k=5),
    },
    optimizer={
        "type": "AdamW",
        "lr": 3e-3,
        "weight_decay": 1e-4,
    },
)

knn_probe = spt.callbacks.OnlineKNN(
    name="knn_probe",
    input="embedding",
    target="label",
    queue_length=10000,
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(NUM_CLASSES),
    },
    input_dim=EMBED_DIM,
    k=20,
)

checkpoint_callback = ModelCheckpoint(
    dirpath=CKPT_DIR,
    filename="ijepa-vitb-{epoch:03d}",
    save_top_k=-1,                     # save all checkpoints
    every_n_epochs=SAVE_EVERY_N_EPOCHS,
    save_last=True,
)

lr_monitor = LearningRateMonitor(logging_interval="step")

wandb_logger = WandbLogger(
    entity="stable-ssl",
    project="imagenet10-mae-ijepa",
    name=f"old-ijepa-vitb-inet10-{time.time():.0f}",
    log_model=False,
)

# --- Trainer ---

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    num_sanity_val_steps=0,
    callbacks=[
        teacher_student_callback,
        linear_probe,
        knn_probe,
        checkpoint_callback,
        lr_monitor,
    ],
    precision="16-mixed",
    logger=wandb_logger,
    devices=1,
    accelerator="gpu",
)

manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()
