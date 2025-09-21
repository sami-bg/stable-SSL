import lightning as pl
import torch
import torchmetrics
from lightning.pytorch.loggers import WandbLogger
from torch import nn

import stable_pretraining as spt
from stable_pretraining.data import transforms
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from benchmarks.utils import get_data_dir

# SimCLR augmentations, as described in the paper
simclr_transform = transforms.MultiViewTransform(
    [
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((224, 224), scale=(0.08, 1.0)),
            transforms.ColorJitter(
                brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.PILGaussianBlur(p=1.0),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(**spt.data.static.ImageNet),
        ),
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((224, 224), scale=(0.08, 1.0)),
            transforms.ColorJitter(
                brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.PILGaussianBlur(p=0.1),
            transforms.RandomSolarize(threshold=0.5, p=0.0), # Note: Solarize was not in original SimCLR
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(**spt.data.static.ImageNet),
        ),
    ]
)

val_transform = transforms.Compose(
    transforms.RGB(),
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToImage(**spt.data.static.ImageNet),
)

# Using a standard ImageNet-1k dataset
train_dataset = spt.data.HFDataset(
    "imagenet-1k",
    split="train",
    cache_dir=str(get_data_dir()),
    transform=simclr_transform,
)
val_dataset = spt.data.HFDataset(
    "imagenet-1k",
    split="validation",
    cache_dir=str(get_data_dir()),
    transform=val_transform,
)

# Batch size from the paper (adjust if necessary)
total_batch_size, world_size = 4096, 4
local_batch_size = total_batch_size // world_size

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    sampler=spt.data.sampler.RepeatedRandomSampler(train_dataset, n_views=2),
    batch_size=local_batch_size,
    num_workers=16,
    drop_last=True,
    persistent_workers=True,
)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=local_batch_size,
    num_workers=8,
    persistent_workers=True,
)

data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)


def forward(self, batch, stage):
    out = {}
    out["embedding"] = self.backbone(batch["image"])
    if self.training:
        proj = self.projector(out["embedding"])
        views = spt.data.fold_views(proj, batch["sample_idx"])
        out["loss"] = self.simclr_loss(views[0], views[1])
    return out


# Using ResNet-50 as in the paper
backbone = spt.backbone.from_torchvision(
    "resnet50",
    low_resolution=False,
)
backbone.fc = torch.nn.Identity()

# Projector network as described in SimCLR paper
projector = nn.Sequential(
    nn.Linear(2048, 2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, 128),
)

module = spt.Module(
    backbone=backbone,
    projector=projector,
    forward=forward,
    # Temperature can be tuned, 0.1 is a common value
    simclr_loss=spt.losses.NTXEntLoss(temperature=0.1),
    optim={
        "optimizer": {
            "type": "LARS",
            "lr": 0.3 * (total_batch_size / local_batch_size),
            "weight_decay": 1e-6,
            "clip_lr": True,
            "eta": 0.001,
            "exclude_bias_n_norm": True,
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
        },
        "interval": "epoch",
    },
)

# Update probes for 1000 classes
linear_probe = spt.callbacks.OnlineProbe(
    name="linear_probe",
    input="embedding",
    target="label",
    probe=torch.nn.Linear(2048, 1000),
    loss_fn=torch.nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(1000),
        "top5": torchmetrics.classification.MulticlassAccuracy(1000, top_k=5),
    },
)

knn_probe = spt.callbacks.OnlineKNN(
    name="knn_probe",
    input="embedding",
    target="label",
    queue_length=20000,
    metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(1000)},
    input_dim=2048,
    k=20,
)

wandb_logger = WandbLogger(
    entity="stable-ssl",
    project="imagenet1k-simclr",
    name="simclr-resnet50-4gpu",
    log_model=False,
)

trainer = pl.Trainer(
    max_epochs=800, # From the paper
    num_sanity_val_steps=0,
    callbacks=[linear_probe, knn_probe],
    precision="16-mixed",
    logger=wandb_logger,
    enable_checkpointing=True,
    accelerator="gpu",
    sync_batchnorm=world_size > 1,
)

manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()