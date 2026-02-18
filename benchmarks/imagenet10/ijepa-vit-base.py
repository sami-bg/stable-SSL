#new-ijepa-vit-base.py
"""I-JEPA (Image-based Joint Embedding Predictive Architecture) on ImageNet-10.

Trains a ViT-Base/16 encoder with an I-JEPA predictor on ImageNette (10-class
ImageNet subset). Hyperparameters follow the I-JEPA paper (Assran et al., 2023)
Table 9 for ViT-Base, adapted for multi-GPU training on a smaller dataset.

Uses stable_pretraining.methods.ijepa.IJEPA for model construction and forward.

Paper: https://arxiv.org/abs/2301.08243

Checkpoints are saved every `save_every_n_epochs` epochs for downstream probing.
"""
import time
import types
import sys
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.methods.ijepa import IJEPA

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir


# =============================================================================
# Hyperparameters (from I-JEPA paper Table 9 for ViT-Base)
# =============================================================================

IMAGE_SIZE = 224
PATCH_SIZE = 16
EMBED_DIM = 768                       # ViT-Base hidden dimension

# Predictor (paper: depth=12, width=384 for ViT-Base)
PRED_DEPTH = 12
PRED_DIM = 384

# Optimizer (paper: AdamW, lr=1.5e-4, wd=0.05, betas=(0.9, 0.95))
# Paper uses batch_size=2048 with lr=1.5e-4. Scale linearly with effective batch.
BASE_LR = 1.5e-4
BATCH_SIZE = 256
NUM_GPUS = torch.cuda.device_count() or 1
EFFECTIVE_BATCH = BATCH_SIZE * NUM_GPUS
SCALED_LR = BASE_LR * (EFFECTIVE_BATCH / 2048)
WEIGHT_DECAY = 0.05
BETAS = (0.9, 0.95)

# EMA schedule (paper: 0.996 → 1.0 cosine)
BASE_EMA = 0.996
FINAL_EMA = 1.0

# Training
MAX_EPOCHS = 300
WARMUP_EPOCHS = 15
NUM_CLASSES = 10  # ImageNette

# Checkpointing
SAVE_EVERY_N_EPOCHS = 25
CKPT_DIR = str(Path(__file__).parent / "checkpoints" / "ijepa-vitb")


# =============================================================================
# I-JEPA Forward Wrapper
# =============================================================================

def ijepa_forward(self, batch, stage):
    """Thin wrapper: unpacks batch dict, calls IJEPA.forward, returns probe-friendly dict."""
    output = IJEPA.forward(self, batch["image"])
    out = {
        "loss": output.loss,
        "embedding": output.embedding.mean(dim=1).detach() if self.training else output.embedding.mean(dim=1),
    }

    if "label" in batch:
        out["label"] = batch["label"].long()

    if self.training:
        self.log(f"{stage}/loss", output.loss, on_step=True, on_epoch=True, sync_dist=True)
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

from stable_datasets.images.imagenette import Imagenette


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
# Model (using stable_pretraining.methods.ijepa.IJEPA)
# =============================================================================

module = IJEPA(
    encoder_name="vit_base_patch16_224",
    predictor_embed_dim=PRED_DIM,
    predictor_depth=PRED_DEPTH,
    num_targets=4,
    target_scale=(0.15, 0.2),
    target_aspect_ratio=(0.75, 1.5),
    context_scale=(0.85, 1.0),
    ema_decay_start=BASE_EMA,
    ema_decay_end=FINAL_EMA,
    pretrained=False,
)

# Bind spt.Module-compatible forward and optimizer config
module.forward = types.MethodType(ijepa_forward, module)
module.optim = {
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
}


# =============================================================================
# Callbacks
# =============================================================================

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
    save_top_k=-1,
    every_n_epochs=SAVE_EVERY_N_EPOCHS,
    save_last=True,
)

lr_monitor = LearningRateMonitor(logging_interval="step")

wandb_logger = WandbLogger(
    entity="stable-ssl",
    project="imagenet10-mae-ijepa",
    name=f"new-ijepa-vitb-inet10-{time.time():.0f}",
    log_model=False,
)

# =============================================================================
# Trainer
# =============================================================================

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
    devices=NUM_GPUS,
    accelerator="gpu",
    strategy="ddp_find_unused_parameters_true" if NUM_GPUS > 1 else "auto",
)

manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()