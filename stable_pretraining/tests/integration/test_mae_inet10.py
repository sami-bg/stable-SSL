"""Deterministic smoke test for the MAE imagenet10 benchmark config."""

import types

import lightning as pl
import pytest
import torch

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.methods.mae import MAE


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.download
@pytest.mark.filterwarnings("ignore:Precision 16-mixed is not supported")
@pytest.mark.filterwarnings("ignore:`isinstance.treespec, LeafSpec.` is deprecated")
@pytest.mark.filterwarnings("ignore:.*does not have many workers")
@pytest.mark.filterwarnings("ignore:Trying to infer the `batch_size`")
class TestMAEImagenet10:
    """Run the inet10 MAE benchmark for 10 steps and check determinism."""

    def test_mae_10_steps(self):
        """Train MAE for 10 steps and assert loss matches expected value."""
        pl.seed_everything(42, workers=True)

        # Build data from frgfm/imagenette (same dataset as benchmarks)
        data = spt.data.DataModule(
            train=torch.utils.data.DataLoader(
                dataset=spt.data.HFDataset(
                    "frgfm/imagenette",
                    split="train",
                    revision="refs/convert/parquet",
                    transform=transforms.Compose(
                        transforms.RGB(),
                        transforms.RandomResizedCrop((224, 224), scale=(0.2, 1.0)),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ToImage(**spt.data.static.ImageNet),
                    ),
                ),
                batch_size=16,
                num_workers=0,
                drop_last=True,
                shuffle=True,
            ),
            val=torch.utils.data.DataLoader(
                dataset=spt.data.HFDataset(
                    "frgfm/imagenette",
                    split="validation",
                    revision="refs/convert/parquet",
                    transform=transforms.Compose(
                        transforms.RGB(),
                        transforms.Resize((256, 256)),
                        transforms.CenterCrop((224, 224)),
                        transforms.ToImage(**spt.data.static.ImageNet),
                    ),
                ),
                batch_size=16,
                num_workers=0,
            ),
        )

        # Forward function matching benchmarks/imagenet10/mae-vit-base.py
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

        # Create MAE module (same hyperparams as benchmark)
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

        # Create trainer (stripped down for testing)
        trainer = pl.Trainer(
            max_steps=10,
            num_sanity_val_steps=0,
            precision="16-mixed",
            logger=False,
            enable_checkpointing=False,
            devices=1,
            accelerator="gpu",
            enable_progress_bar=False,
        )

        # Run training
        manager = spt.Manager(trainer=trainer, module=module, data=data, seed=42)
        manager()

        # Verify deterministic loss
        final_loss = trainer.callback_metrics.get("fit/loss_step")
        assert final_loss is not None, "No loss logged"

        print(f"\nMAE final loss after 10 steps: {final_loss.item():.6f}")
        expected = torch.tensor(0.950017)
        assert torch.isclose(final_loss.cpu(), expected, atol=1e-4), (
            f"MAE loss {final_loss.item():.6f} != expected {expected.item():.6f}"
        )
