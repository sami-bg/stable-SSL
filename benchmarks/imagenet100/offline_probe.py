"""Offline linear probe evaluation for MAE/IJEPA checkpoints on ImageNet-100.

Iterates over pretraining checkpoints and trains a supervised linear classifier
on top of the frozen encoder. Supports both MAE and IJEPA pretrained ViT-Base/16.

The model type is inferred automatically from the checkpoint filename or its
parent directory name (must contain 'mae' or 'ijepa').

Usage:
    python benchmarks/imagenet100/offline_probe.py --ckpt-dir checkpoints/mae-vitb
    python benchmarks/imagenet100/offline_probe.py --ckpt-dir checkpoints/ijepa-vitb

    # Evaluate a single checkpoint:
    python benchmarks/imagenet100/offline_probe.py --single /path/to/mae-vitb-epoch=050.ckpt
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
from stable_pretraining.methods.ijepa import IJEPA
from stable_pretraining.methods.mae import MAE

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir


IMAGE_SIZE = 224
EMBED_DIM = 768  # ViT-Base hidden dimension
NUM_CLASSES = 100  # ImageNet-100
PROBE_EPOCHS = 100
BATCH_SIZE = 256
# NUM_GPUS = torch.cuda.device_count() or 1
NUM_GPUS = 1
LR = 0.01


def get_data() -> spt.data.DataModule:
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

    data_dir = get_data_dir("imagenet100")

    train_dataset = spt.data.HFDataset(
        "clane9/imagenet-100",
        split="train",
        cache_dir=str(data_dir),
        transform=train_transform,
    )
    val_dataset = spt.data.HFDataset(
        "clane9/imagenet-100",
        split="validation",
        cache_dir=str(data_dir),
        transform=val_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=16,
        drop_last=True,
        persistent_workers=True,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=16,
        persistent_workers=True,
    )

    return spt.data.DataModule(train=train_loader, val=val_loader)


def load_ijepa_encoder(ckpt_path: str) -> nn.Module:
    module = IJEPA(
        encoder_name="vit_base_patch16_224",
        predictor_embed_dim=384,
        predictor_depth=12,
        pretrained=False,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    module.load_state_dict(ckpt["state_dict"], strict=False)
    # encoder is a TeacherStudentWrapper; use the student (context encoder)
    return module.encoder.student


def load_mae_encoder(ckpt_path: str) -> nn.Module:
    module = MAE(
        encoder_name="vit_base_patch16_224",
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mask_ratio=0.75,
        pretrained=False,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    module.load_state_dict(ckpt["state_dict"], strict=False)
    return module.encoder


LOADERS = {
    "ijepa": load_ijepa_encoder,
    "mae": load_mae_encoder,
}


def infer_model_from_path(path: str) -> str:
    """Infer 'mae' or 'ijepa' from a checkpoint path (filename, then parent dir)."""
    for candidate in [Path(path).stem, Path(path).parent.name]:
        lower = candidate.lower()
        if "mae" in lower:
            return "mae"
        if "ijepa" in lower:
            return "ijepa"
    raise ValueError(
        f"Cannot infer model type from '{path}'. "
        "Checkpoint filename or parent directory must contain 'mae' or 'ijepa'."
    )


def frozen_embedding_forward(self, batch: Dict, _stage: str):
    """Pool frozen MaskedEncoder features over patch tokens (skip CLS)."""
    self.backbone.eval()
    features = self.backbone.forward_features(batch["image"])
    embedding = features[:, self.backbone.num_prefix_tokens :].mean(dim=1)
    return {
        "embedding": embedding,
        "label": batch["label"].long(),
    }


def extract_epoch_from_path(ckpt_path: str) -> int:
    """Extract epoch number from checkpoint filename (e.g. mae-vitb-epoch=025.ckpt)."""
    name = Path(ckpt_path).stem
    for part in name.split("-"):
        if part.startswith("epoch="):
            return int(part.split("=")[1])
    for part in name.split("-"):
        if part.isdigit():
            return int(part)
    return -1


def train_linear_probe(
    encoder: nn.Module,
    data: spt.data.DataModule,
    ckpt_path: str,
    epoch_num: int,
    model_type: str,
    use_wandb: bool,
) -> dict:
    for param in encoder.parameters():
        param.requires_grad = False

    module = spt.Module(
        backbone=encoder,
        forward=frozen_embedding_forward,
        optim=None,
    )

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
            "top5": torchmetrics.classification.MulticlassAccuracy(
                NUM_CLASSES, top_k=5
            ),
        },
    )

    if use_wandb:
        logger = WandbLogger(
            entity="stable-ssl",
            project="imagenet100-offline-probe",
            name=f"{model_type}-probe-epoch{epoch_num:03d}-{time.time():.0f}",
            log_model=False,
        )
    else:
        logger = False

    trainer = pl.Trainer(
        max_epochs=PROBE_EPOCHS,
        num_sanity_val_steps=0,
        callbacks=[linear_probe, LearningRateMonitor(logging_interval="step")],
        precision="16-mixed",
        logger=logger,
        devices=NUM_GPUS,
        accelerator="gpu",
        strategy="auto",
        enable_checkpointing=False,
    )

    trainer.fit(module, datamodule=data)
    val_results = trainer.validate(module, datamodule=data, verbose=False)

    result = {"pretrain_epoch": epoch_num, "checkpoint": str(ckpt_path)}
    if val_results:
        result.update(val_results[0])

    if use_wandb and logger:
        logger.experiment.finish()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Offline linear probe for MAE/IJEPA checkpoints on ImageNet-100"
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="Directory of checkpoints to evaluate (filenames must contain 'mae' or 'ijepa')",
    )
    parser.add_argument(
        "--single",
        type=str,
        default=None,
        help="Evaluate a single checkpoint instead of iterating over all",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON (default: <ckpt_dir>/probe_results.json)",
    )
    args = parser.parse_args()

    if not args.single and not args.ckpt_dir:
        parser.error("Provide --ckpt-dir or --single.")

    use_wandb = not args.no_wandb

    if args.single:
        ckpt_paths = [args.single]
        ckpt_dir = Path(args.single).parent
    else:
        ckpt_dir = Path(args.ckpt_dir)
        if not ckpt_dir.exists():
            print(f"Error: checkpoint directory {ckpt_dir} does not exist.")
            sys.exit(1)
        ckpt_paths = sorted(glob(str(ckpt_dir / "*.ckpt")))
        ckpt_paths = [p for p in ckpt_paths if "last" not in Path(p).stem]
        if not ckpt_paths:
            ckpt_paths = sorted(glob(str(ckpt_dir / "*.ckpt")))

    if not ckpt_paths:
        print(f"No checkpoints found in {ckpt_dir}")
        sys.exit(1)

    print(f"Found {len(ckpt_paths)} checkpoint(s) to evaluate:")
    for p in ckpt_paths:
        print(f"  {p}")
    print()

    data = get_data()

    all_results = []
    for i, ckpt_path in enumerate(ckpt_paths):
        model_type = infer_model_from_path(ckpt_path)
        epoch_num = extract_epoch_from_path(ckpt_path)
        print(f"\n{'=' * 70}")
        print(
            f"[{i + 1}/{len(ckpt_paths)}] {Path(ckpt_path).name}  (model={model_type}, epoch={epoch_num})"
        )
        print(f"{'=' * 70}")

        encoder = LOADERS[model_type](ckpt_path)
        encoder.eval()

        result = train_linear_probe(
            encoder=encoder,
            data=data,
            ckpt_path=ckpt_path,
            epoch_num=epoch_num,
            model_type=model_type,
            use_wandb=use_wandb,
        )
        result["model"] = model_type
        all_results.append(result)
        print(f"  Result: {result}")

    output_path = args.output or str(ckpt_dir / "probe_results.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to: {output_path}")

    print(f"\n{'=' * 70}")
    print("SUMMARY — Linear Probe on ImageNet-100")
    print(f"{'=' * 70}")
    print(f"{'Model':>8}  {'Epoch':>8}  {'Top-1':>8}  {'Top-5':>8}  {'Checkpoint'}")
    print(f"{'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 40}")
    for r in all_results:
        model = r.get("model", "?")
        epoch = r.get("pretrain_epoch", "?")
        top1 = next((v for k, v in r.items() if "top1" in k), None)
        top5 = next((v for k, v in r.items() if "top5" in k), None)
        top1_s = f"{top1:.4f}" if top1 is not None else "N/A"
        top5_s = f"{top5:.4f}" if top5 is not None else "N/A"
        print(
            f"{model:>8}  {epoch:>8}  {top1_s:>8}  {top5_s:>8}  {Path(r.get('checkpoint', '')).name}"
        )


if __name__ == "__main__":
    main()
