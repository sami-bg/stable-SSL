"""Deterministic smoke tests for MAE and IJEPA benchmark configs (imagenet10)."""

import types

import lightning as pl
import pytest
import torch

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.methods.ijepa import IJEPA
from stable_pretraining.methods.mae import MAE

BATCH_SIZE = 16
MAX_STEPS = 10
SEED = 42


def _imagenette_data(batch_size, train_transform, val_transform):
    """Build DataModule from frgfm/imagenette (parquet)."""
    common = dict(revision="refs/convert/parquet")
    return spt.data.DataModule(
        train=torch.utils.data.DataLoader(
            dataset=spt.data.HFDataset(
                "frgfm/imagenette",
                split="train",
                transform=train_transform,
                **common,
            ),
            batch_size=batch_size,
            num_workers=0,
            drop_last=True,
            shuffle=True,
        ),
        val=torch.utils.data.DataLoader(
            dataset=spt.data.HFDataset(
                "frgfm/imagenette",
                split="validation",
                transform=val_transform,
                **common,
            ),
            batch_size=batch_size,
            num_workers=0,
        ),
    )


def _train_transform():
    return transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((224, 224), scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToImage(**spt.data.static.ImageNet),
    )


def _val_transform():
    return transforms.Compose(
        transforms.RGB(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToImage(**spt.data.static.ImageNet),
    )


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.download
class TestBenchmarkDeterminism:
    """Run inet10 benchmarks for 10 steps and assert loss is close to the expected value in the deterministic case."""

    def test_mae_10_steps(self):
        pl.seed_everything(SEED, workers=True)
        data = _imagenette_data(BATCH_SIZE, _train_transform(), _val_transform())

        def mae_forward(self, batch, stage):
            output = MAE.forward(self, batch["image"])
            with torch.no_grad():
                features = self.encoder.forward_features(batch["image"])

            self.log(
                f"{stage}/loss",
                output.loss,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

            return {
                "loss": output.loss,
                "embedding": features[:, 1:].mean(dim=1).detach(),
                **({"label": batch["label"].long()} if "label" in batch else {}),
            }

        module = MAE(
            encoder_name="vit_base_patch16_224",
            decoder_embed_dim=512,
            decoder_depth=8,
            decoder_num_heads=16,
            mask_ratio=0.75,
            block_size=1,
            norm_pix_loss=True,
            loss_type="mse",
            pretrained=False,
        )

        module.forward = types.MethodType(mae_forward, module)
        module.optim = {
            "optimizer": {
                "type": "AdamW",
                "lr": 5e-4,
                "weight_decay": 0.05,
                "betas": (0.9, 0.95),
            },
            "scheduler": {"type": "LinearWarmupCosineAnnealing"},
            "interval": "epoch",
        }

        trainer = pl.Trainer(
            max_steps=MAX_STEPS,
            num_sanity_val_steps=0,
            precision="16-mixed",
            logger=False,
            enable_checkpointing=False,
            devices=1,
            accelerator="gpu",
            enable_progress_bar=False,
        )

        manager = spt.Manager(trainer=trainer, module=module, data=data, seed=SEED)
        manager()

        final_loss = trainer.callback_metrics.get("fit/loss_step")
        assert final_loss is not None, "No loss logged"
        print(f"\nMAE final loss after {MAX_STEPS} steps: {final_loss.item():.6f}")
        expected = torch.tensor(0.950017)
        assert torch.isclose(final_loss.cpu(), expected, atol=1e-4), (
            f"MAE loss {final_loss.item():.6f} != expected {expected.item():.6f}"
        )

    def test_ijepa_10_steps(self):
        pl.seed_everything(SEED, workers=True)
        data = _imagenette_data(
            BATCH_SIZE,
            # IJEPA uses scale=(0.3, 1.0) and no horizontal flip
            transforms.Compose(
                transforms.RGB(),
                transforms.RandomResizedCrop((224, 224), scale=(0.3, 1.0)),
                transforms.ToImage(**spt.data.static.ImageNet),
            ),
            _val_transform(),
        )

        def ijepa_forward(self, batch, stage):
            output = IJEPA.forward(self, batch["image"], embedding_source="student")
            embedding = output.embedding.mean(dim=1)
            if self.training:
                embedding = embedding.detach()

            self.log(
                f"{stage}/loss",
                output.loss,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

            return {
                "loss": output.loss,
                "embedding": embedding,
                **({"label": batch["label"].long()} if "label" in batch else {}),
            }

        module = IJEPA(
            encoder_name="vit_base_patch16_224",
            predictor_embed_dim=384,
            predictor_depth=12,
            num_targets=4,
            target_scale=(0.15, 0.2),
            target_aspect_ratio=(0.75, 1.5),
            context_scale=(0.85, 1.0),
            ema_decay_start=0.996,
            ema_decay_end=1.0,
            pretrained=False,
        )

        module.forward = types.MethodType(ijepa_forward, module)
        module.optim = {
            "optimizer": {
                "type": "AdamW",
                "lr": 6e-4,
                "weight_decay": 0.05,
                "betas": (0.9, 0.95),
            },
            "scheduler": {"type": "LinearWarmupCosineAnnealing"},
            "interval": "epoch",
        }

        trainer = pl.Trainer(
            max_steps=MAX_STEPS,
            num_sanity_val_steps=0,
            callbacks=[
                spt.callbacks.TeacherStudentCallback(
                    update_frequency=1,
                    update_after_backward=True,
                ),
            ],
            precision="16-mixed",
            logger=False,
            enable_checkpointing=False,
            devices=1,
            accelerator="gpu",
            enable_progress_bar=False,
        )

        manager = spt.Manager(trainer=trainer, module=module, data=data, seed=SEED)
        manager()

        final_loss = trainer.callback_metrics.get("fit/loss_step")
        assert final_loss is not None, "No loss logged"
        print(f"\nIJEPA final loss after {MAX_STEPS} steps: {final_loss.item():.6f}")
        expected = torch.tensor(0.404552)
        assert torch.isclose(final_loss.cpu(), expected, atol=1e-4), (
            f"IJEPA loss {final_loss.item():.6f} != expected {expected.item():.6f}"
        )
