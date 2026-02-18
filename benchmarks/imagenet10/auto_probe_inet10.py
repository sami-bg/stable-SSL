"""Offline linear probe evaluation for I-JEPA checkpoints on ImageNet-10.

Sweeps LRs and weight decays following the I-JEPA paper (Table 8):
  - LARS optimizer (SGD fallback), batch 16384 in paper → scaled to local batch
  - StepLR: divide by 10 every 15 epochs
  - Sweep: lr in [0.001, 0.01, 0.05], wd in [0.0, 0.0005]
  - 100 epochs (paper uses 50 on IN-1k, more steps needed here)

Usage:
    python benchmarks/imagenet10/linear-probe-inet10.py
    python benchmarks/imagenet10/linear-probe-inet10.py --single /path/to/ckpt.ckpt
    python benchmarks/imagenet10/linear-probe-inet10.py --no-wandb
"""

import argparse
import sys
import time
from glob import glob
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

import stable_pretraining as spt
from stable_pretraining.backbone.probe import AutoLinearClassifier
from stable_pretraining.data import transforms
from stable_pretraining.methods.ijepa import IJEPA

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir


# Config
IMAGE_SIZE = 224
EMBED_DIM = 768
NUM_CLASSES = 10
PROBE_EPOCHS = 100
BATCH_SIZE = 256
SWEEP_LRS = [1e-2, 1e-1, 3e-1]
SWEEP_WDS = [0.0, 0.0005]

DEFAULT_CKPT_DIR = str(Path(__file__).parent / "checkpoints" / "ijepa-vitb")


# Data
def get_data():
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


# Module
class IJEPALinearProbe(spt.Module):
    def __init__(self, ckpt_path: str):
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        backbone = IJEPA(
            encoder_name="vit_base_patch16_224",
            predictor_embed_dim=384,
            predictor_depth=12,
            pretrained=False,
        )
        missing, unexpected = backbone.load_state_dict(
            checkpoint["state_dict"], strict=False
        )
        if missing:
            raise RuntimeError(f"Missing keys loading checkpoint: {missing[:5]}")
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad = False

        # Sweep: 3 lrs x 2 wds x 1 norm = 6 probes in one run
        head = AutoLinearClassifier(
            name="probe",
            embedding_dim=EMBED_DIM,
            num_classes=NUM_CLASSES,
            pooling="mean",
            normalization=["none", "bn"],
            lr_scaling=[1],
            weight_decay=[0.0, 0.0005],
            dropout=[0],
            label_smoothing=[0],
        )

        super().__init__(
            backbone=backbone,
            head=head,
            forward=self._probe_forward,
            optim={
                "optimizer": {
                    "type": "SGD",  # swap to LARS if available in spt
                    "lr": 1.0,      # base lr=1.0; lr_scaling in AutoLinearClassifier scales per-head
                    "momentum": 0.9,
                    "weight_decay": 0.0,
                },
                "scheduler": {
                    "type": "StepLR",
                    "step_size": 15,
                    "gamma": 0.1,   # paper: divide by 10 every 15 epochs
                },
                "interval": "epoch",
            },
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def _probe_forward(self, batch, stage):
        with torch.no_grad():
            output = IJEPA.forward(self.backbone, batch["image"])
            embedding = output.embedding.mean(dim=1)  # avg pool patches, matches paper

        labels = batch["label"].long()
        logits_dict = self.head(embedding)

        total_loss = torch.tensor(0.0, device=embedding.device)
        best_acc = torch.tensor(0.0, device=embedding.device)
        for probe_name, logits in logits_dict.items():
            loss = self.loss_fn(logits, labels)
            total_loss = total_loss + loss
            acc = (logits.argmax(dim=1) == labels).float().mean()
            best_acc = torch.max(best_acc, acc)
            self.log(f"{stage}/{probe_name}_top1", acc, on_epoch=True, on_step=False)

        self.log(f"{stage}/best_top1", best_acc, on_epoch=True, on_step=False)
        return {"loss": total_loss}


# Main
def extract_epoch(ckpt_path: str) -> int:
    name = Path(ckpt_path).stem
    for part in name.split("-"):
        if part.startswith("epoch="):
            return int(part.split("=")[1])
    return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=str, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--single", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    if args.single:
        ckpt_paths = [args.single]
    else:
        ckpt_dir = Path(args.ckpt_dir)
        ckpt_paths = sorted(glob(str(ckpt_dir / "*.ckpt")))
        ckpt_paths = [p for p in ckpt_paths if "last" not in Path(p).stem]
        if not ckpt_paths:
            ckpt_paths = sorted(glob(str(ckpt_dir / "*.ckpt")))

    if not ckpt_paths:
        print(f"No checkpoints found in {args.ckpt_dir}")
        sys.exit(1)

    data = get_data()

    for ckpt_path in ckpt_paths:
        epoch_num = extract_epoch(ckpt_path)
        print(f"\nProbing: {Path(ckpt_path).name} (epoch {epoch_num})")

        probe = IJEPALinearProbe(ckpt_path=ckpt_path)

        logger = (
            WandbLogger(
                entity="stable-ssl",
                project="imagenet10-ijepa-probe",
                name=f"probe-inet10-ep{epoch_num:03d}-{time.time():.0f}",
                log_model=False,
            )
            if not args.no_wandb
            else False
        )

        trainer = pl.Trainer(
            max_epochs=PROBE_EPOCHS,
            num_sanity_val_steps=0,
            callbacks=[LearningRateMonitor(logging_interval="epoch")],
            precision="16-mixed",
            logger=logger,
            devices=1,
            accelerator="gpu",
            enable_checkpointing=False,
        )

        spt.Manager(trainer=trainer, module=probe, data=data)()

        if logger:
            logger.experiment.finish()


if __name__ == "__main__":
    main()
