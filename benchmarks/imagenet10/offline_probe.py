"""Linear probe evaluation for I-JEPA checkpoints on ImageNet-10.

Iterates over all I-JEPA pretraining checkpoints and trains a supervised
linear classifier on top of the frozen encoder to completion.

Hyperparameters follow the I-JEPA paper (Assran et al., 2023) Table 8
for linear evaluation on ImageNet.

Usage:
    python benchmarks/imagenet10/linear-probe-inet10.py

    # Override checkpoint directory:
    python benchmarks/imagenet10/linear-probe-inet10.py --ckpt-dir /path/to/ckpts

    # Evaluate a single checkpoint:
    python benchmarks/imagenet10/linear-probe-inet10.py --single /path/to/ckpt.ckpt
"""

import argparse
import json
import sys
import time
from glob import glob
from pathlib import Path
from typing import Dict

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

import stable_pretraining as spt
from stable_pretraining.data import transforms

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir


# =============================================================================
# Linear Probe Hyperparameters (I-JEPA paper Table 8)
# =============================================================================

IMAGE_SIZE = 224
PATCH_SIZE = 16
EMBED_DIM = 768       # ViT-Base
NUM_CLASSES = 10       # ImageNette
PROBE_EPOCHS = 100     # Paper: 100 epochs for linear eval
BATCH_SIZE = 256
NUM_GPUS = torch.cuda.device_count() or 1
EFFECTIVE_BATCH = BATCH_SIZE * NUM_GPUS
BASE_LR = 0.001        # Paper: lr tuned per dataset, 0.001 is standard for probing
LR = BASE_LR * NUM_GPUS  # linear scaling with GPU count

DEFAULT_CKPT_DIR = str(Path(__file__).parent / "checkpoints" / "ijepa-vitb")


# =============================================================================
# Data
# =============================================================================

def get_data():
    """Create data module for linear probing with standard ImageNet transforms."""
    train_transform = transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.08, 1.0)),
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

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        drop_last=True,
        persistent_workers=True,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        persistent_workers=True,
    )

    return spt.data.DataModule(train=train_loader, val=val_loader)


# =============================================================================
# Checkpoint Loading
# =============================================================================

def load_encoder_from_checkpoint(ckpt_path: str) -> nn.Module:
    """Load the ViT-Base encoder from an I-JEPA checkpoint.

    The checkpoint contains the full spt.Module state (backbone, predictor, etc).
    We extract only the student encoder from the TeacherStudentWrapper backbone.
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]

    # Create a fresh ViT-Base
    backbone = spt.backbone.vit_hf(
        size="base",
        patch_size=PATCH_SIZE,
        image_size=IMAGE_SIZE,
        pretrained=False,
    )

    # Extract student encoder weights from the checkpoint.
    # In the checkpoint, the backbone is wrapped in TeacherStudentWrapper,
    # so weights are stored under "backbone.student.*"
    prefix = "backbone.student."
    encoder_state = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            encoder_state[key[len(prefix):]] = value

    if not encoder_state:
        # Fallback: try "backbone.teacher." (teacher has same architecture)
        prefix = "backbone.teacher."
        for key, value in state_dict.items():
            if key.startswith(prefix):
                encoder_state[key[len(prefix):]] = value

    if not encoder_state:
        raise RuntimeError(
            f"Could not find encoder weights in checkpoint {ckpt_path}. "
            f"Available keys start with: {set(k.split('.')[0] for k in state_dict.keys())}"
        )

    missing, unexpected = backbone.load_state_dict(encoder_state, strict=False)
    if missing:
        print(f"  Warning: missing keys when loading encoder: {missing[:5]}...")
    if unexpected:
        print(f"  Warning: unexpected keys when loading encoder: {unexpected[:5]}...")

    return backbone


def extract_epoch_from_path(ckpt_path: str) -> int:
    """Extract epoch number from checkpoint filename like 'ijepa-vitb-epoch=025.ckpt'."""
    name = Path(ckpt_path).stem
    for part in name.split("-"):
        if part.startswith("epoch="):
            return int(part.split("=")[1])
    for part in name.split("-"):
        if part.isdigit():
            return int(part)
    return -1


# =============================================================================
# Module Forward (frozen encoder → embeddings only)
# =============================================================================

def frozen_embedding_forward(self, batch: Dict, stage: str):
    """Forward that extracts frozen embeddings for probe evaluation.

    Like the multi-layer probe pattern: the forward only produces embeddings,
    and OnlineProbe handles the classification head, loss, and metrics.
    """
    with torch.no_grad():
        patches = self.backbone(batch["image"]).last_hidden_state[:, 1:, :]
    return {"embedding": patches.mean(dim=1)}


# =============================================================================
# Linear Probe Training
# =============================================================================

def train_linear_probe(
    encoder: nn.Module,
    data: spt.data.DataModule,
    ckpt_path: str,
    epoch_num: int,
    use_wandb: bool = True,
) -> dict:
    """Train a linear classifier on top of a frozen encoder."""
    # Freeze encoder
    for param in encoder.parameters():
        param.requires_grad = False

    # Module only produces embeddings — no optimizer needed
    module = spt.Module(
        backbone=encoder,
        forward=frozen_embedding_forward,
        optim=None,
    )

    # OnlineProbe handles the linear head, loss, optimizer, and metrics
    linear_probe = spt.callbacks.OnlineProbe(
        module,
        name="linear_probe",
        input="embedding",
        target="label",
        probe=nn.Sequential(
            nn.BatchNorm1d(EMBED_DIM),
            nn.Linear(EMBED_DIM, NUM_CLASSES),
        ),
        loss=nn.CrossEntropyLoss(),
        optimizer={"type": "SGD", "lr": LR, "momentum": 0.9},
        scheduler={"type": "CosineAnnealingLR", "T_max": PROBE_EPOCHS},
        metrics={
            "top1": torchmetrics.classification.MulticlassAccuracy(NUM_CLASSES),
            "top5": torchmetrics.classification.MulticlassAccuracy(NUM_CLASSES, top_k=5),
        },
    )

    callbacks = [linear_probe, LearningRateMonitor(logging_interval="step")]

    if use_wandb:
        logger = WandbLogger(
            entity="stable-ssl",
            project="imagenet10-ijepa-probe",
            name=f"probe-epoch{epoch_num:03d}-{time.time():.0f}",
            log_model=False,
        )
    else:
        logger = False

    trainer = pl.Trainer(
        max_epochs=PROBE_EPOCHS,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        precision="16-mixed",
        logger=logger,
        devices=NUM_GPUS,
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true" if NUM_GPUS > 1 else "auto",
        enable_checkpointing=False,
    )

    manager = spt.Manager(trainer=trainer, module=module, data=data)
    manager()

    # Collect final validation metrics
    val_results = trainer.validate(module, datamodule=data, verbose=False)
    result = {
        "pretrain_epoch": epoch_num,
        "checkpoint": str(ckpt_path),
    }
    if val_results:
        result.update(val_results[0])

    if use_wandb and logger:
        logger.experiment.finish()

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Linear probe evaluation of I-JEPA checkpoints on ImageNet-10"
    )
    parser.add_argument(
        "--ckpt-dir", type=str, default=DEFAULT_CKPT_DIR,
        help="Directory containing I-JEPA pretraining checkpoints",
    )
    parser.add_argument(
        "--single", type=str, default=None,
        help="Evaluate a single checkpoint instead of iterating over all",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save results JSON (default: <ckpt_dir>/probe_results.json)",
    )
    args = parser.parse_args()

    use_wandb = not args.no_wandb

    # Discover checkpoints
    if args.single:
        ckpt_paths = [args.single]
    else:
        ckpt_dir = Path(args.ckpt_dir)
        if not ckpt_dir.exists():
            print(f"Error: checkpoint directory {ckpt_dir} does not exist.")
            print("Run ijepa-vitb.py first to generate pretraining checkpoints.")
            sys.exit(1)
        ckpt_paths = sorted(glob(str(ckpt_dir / "*.ckpt")))
        # Exclude 'last.ckpt' if a versioned one at the same epoch exists
        ckpt_paths = [p for p in ckpt_paths if "last" not in Path(p).stem]
        if not ckpt_paths:
            ckpt_paths = sorted(glob(str(ckpt_dir / "*.ckpt")))

    if not ckpt_paths:
        print(f"No checkpoints found in {args.ckpt_dir}")
        sys.exit(1)

    print(f"Found {len(ckpt_paths)} checkpoint(s) to evaluate:")
    for p in ckpt_paths:
        print(f"  {p}")
    print()

    # Create data (shared across all probes)
    data = get_data()

    # Evaluate each checkpoint
    all_results = []
    for i, ckpt_path in enumerate(ckpt_paths):
        epoch_num = extract_epoch_from_path(ckpt_path)
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(ckpt_paths)}] Probing checkpoint: {Path(ckpt_path).name}")
        print(f"  Pretraining epoch: {epoch_num}")
        print(f"{'='*70}")

        encoder = load_encoder_from_checkpoint(ckpt_path)
        result = train_linear_probe(
            encoder=encoder,
            data=data,
            ckpt_path=ckpt_path,
            epoch_num=epoch_num,
            use_wandb=use_wandb,
        )
        all_results.append(result)
        print(f"  Result: {result}")

    # Save results
    output_path = args.output or str(Path(args.ckpt_dir) / "probe_results.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to: {output_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY: Linear Probe Top-1 Accuracy by Pretraining Epoch")
    print(f"{'='*70}")
    print(f"{'Epoch':>8}  {'Top-1':>8}  {'Checkpoint'}")
    print(f"{'-'*8}  {'-'*8}  {'-'*40}")
    for r in all_results:
        epoch = r.get("pretrain_epoch", "?")
        top1 = None
        for key in r:
            if "top1" in key.lower() or "accuracy" in key.lower():
                top1 = r[key]
                break
        top1_str = f"{top1:.4f}" if top1 is not None else "N/A"
        ckpt_name = Path(r.get("checkpoint", "")).name
        print(f"{epoch:>8}  {top1_str:>8}  {ckpt_name}")


if __name__ == "__main__":
    main()