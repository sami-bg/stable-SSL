import hydra
import torch
import stable_pretraining as spt
from stable_pretraining.data import transforms as T


def get_transform(mean, std, level: int = 2, image_size: int = 224):
    """Get ImageNet training augmentation at specified severity level.

    Args:
        level: Augmentation severity (1=minimal, 2=standard, 3=aggressive).
        image_size: Output image size.
        interpolation: Interpolation mode for resizing.

    Returns:
        Composed transform pipeline.
    """

    if level not in {0, 1, 2, 3}:
        raise ValueError(f"level must be 1, 2, or 3, got {level}")

    # Level 0: RRC only
    transforms = [T.RGB()]
    if level == 0:
        transforms.append(T.Resize(image_size))
        transforms.append(T.CenterCrop((image_size, image_size)))
    # Level 2: + RRC
    if level >= 1:
        transforms.append(T.RandomHorizontalFlip(p=0.5))
        transforms.append(T.RandomResizedCrop(image_size))

    # Level 2: + flip, color, blur, grayscale, solarize
    if level >= 2:
        transforms.append(T.ColorJitter(0.8, 0.8, 0.8, 0.2, p=0.8))
        transforms.append(T.RandomGrayscale(p=0.2))
        transforms.append(T.RandomSolarize(threshold=128, p=0.2))

    # Level 3: + geometry, posterize, equalize, sharpness, erasing
    if level >= 3:
        raise NotImplementedError
    transforms.append(T.ToImage(mean=mean, std=std))
    return T.Compose(*transforms)


def get_loaders(cfg):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = get_transform(
        mean, std, level=cfg.get("da_level"), image_size=cfg.get("resolution")
    )
    train_dataset = spt.data.HFDataset(
        path="frgfm/imagenette",
        name="full_size",
        split="train",
        transform=train_transform,
    )

    val_transform = get_transform(mean, std, level=0, image_size=cfg.get("resolution"))
    val_dataset = spt.data.HFDataset(
        path="frgfm/imagenette",
        name="full_size",
        split="validation",
        transform=val_transform,
    )
    num_classes = 10

    kwargs = dict(
        batch_size=cfg.get("batch_size"),
        num_workers=cfg.get("num_workers"),
        pin_memory=True,
        persistent_workers=True,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, **kwargs, drop_last=True, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, **kwargs)
    return train_loader, val_loader, num_classes


def plot_masked_images(image, masks):
    """
    :param image: (C, H, W) tensor (square)
    :param masks: (n_masks, T) boolean tensor
    """
    import matplotlib.pyplot as plt

    grid = int(masks.shape[1] ** 0.5)
    patch = image.shape[1] // grid

    img = image.permute(1, 2, 0).float().cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())

    n = len(masks)
    fig, axes = plt.subplots(2, max(1, n // 2), figsize=(3 * n // 2, 6))

    for ax, mask in zip(axes.flatten(), masks.cpu().numpy()):
        mask_px = mask.reshape(grid, grid).repeat(patch, 0).repeat(patch, 1)
        ax.imshow(img * (1 - mask_px[..., None]))
        ax.axis("off")

    return fig


def plot_mae_reconstruction(original, reconstruction, masks):
    """
    :param original: (N, C, H, W) tensor (square)
    :param reconstruction: (N, C, H, W) tensor
    :param masks: (N, T) boolean tensor, True=masked
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np

    grid = int(masks.shape[1] ** 0.5)
    patch = original.shape[2] // grid

    def normalize(x):
        x = x.permute(1, 2, 0).float().cpu().numpy()
        return np.clip((x - x.min()) / (x.max() - x.min() + 1e-8), 0, 1)

    n = len(masks)
    fig, axes = plt.subplots(n, 3, figsize=(10, 3 * n), squeeze=False)

    for idx, mask in enumerate(masks.cpu().numpy()):
        mask_2d = mask.reshape(grid, grid)
        orig, recon = normalize(original[idx]), normalize(reconstruction[idx])

        mask_px = mask_2d.repeat(patch, 0).repeat(patch, 1)
        masked_view = np.where(mask_px[..., None], 0.5, orig)

        axes[idx, 0].imshow(orig)
        axes[idx, 1].imshow(masked_view)
        axes[idx, 2].imshow(recon)

        for i, j in zip(*np.where(1 - mask_2d)):
            axes[idx, 1].add_patch(
                Rectangle(
                    (j * patch, i * patch), patch, patch, lw=2, ec="black", fc="none"
                )
            )

        for ax in axes[idx]:
            ax.axis("off")

    axes[0, 0].set_title("Original")
    axes[0, 1].set_title("Visible patches")
    axes[0, 2].set_title("Reconstruction")

    plt.tight_layout()
    return fig


@hydra.main(version_base="1.3")
def main(cfg: dict):
    import stable_pretraining as spt
    import torch
    import lightning as pl
    from lightning.pytorch.loggers import WandbLogger, CSVLogger
    from functools import partial
    from lightning.pytorch.plugins.io import AsyncCheckpointIO
    from omegaconf import open_dict
    import time
    from torchvision.ops import MLP
    import wandb
    import matplotlib.pyplot as plt

    torch.set_float32_matmul_precision("medium")

    with open_dict(cfg):
        cfg.setdefault("decode", False)
        cfg.setdefault("sigreg_lambda", 10)
        cfg.setdefault("sigreg_recon", 1)
        cfg.setdefault("sigreg_token", 0.1)
        cfg.setdefault("sigreg_n_quad", 1)
        cfg.setdefault("sigreg_sigma", 0)
        cfg.setdefault("batch_size", 32)
        cfg.setdefault("da_level", 0)
        cfg.setdefault("embedding_dim", 384)
        cfg.setdefault("resolution", 224)
        cfg.setdefault("lr", 1e-3)
        cfg.setdefault("weight_decay", 1e-5)
        cfg.setdefault("backbone", "vit_small_patch16_224")
        cfg.setdefault("max_epochs", 100)
        cfg.setdefault("n_views", 1)
        cfg.setdefault("num_nodes", 1)
        cfg.setdefault("num_workers", 10)
        cfg.setdefault("patch_size", 16)
        cfg.setdefault("wandb_project", None)
        cfg.setdefault("seed", 42)
        cfg.setdefault("projector_width", [2048, 2048])

    torch.manual_seed(cfg.seed)

    # without transform
    train_loader, val_loader, num_classes = get_loaders(cfg)
    data_module = spt.data.DataModule(train=train_loader, val=val_loader)

    def forward(self, batch, stage):
        N = batch["image"].size(0)
        V = cfg.get("n_views") if self.training else 1
        D = cfg.get("embedding_dim")
        num_prefix = self.encoder.num_prefix_tokens  # typically 1
        # Expand views
        if V > 1:
            exp_images = batch["image"].unsqueeze(1).expand(-1, V, -1, -1, -1)
            images = exp_images.flatten(0, 1)
            batch["label"] = torch.repeat_interleave(batch["label"], V)
        else:
            images = batch["image"]
        # Encode and project
        enc_out = self.encoder(images)
        batch["embedding"] = enc_out.encoded
        proj = self.projector(enc_out.encoded.flatten(0, 1))
        batch["proj"] = proj.reshape(*enc_out.encoded.shape)
        if not self.training:
            return batch
        num_patches = enc_out.mask.shape[1]
        num_visible = enc_out.ids_keep.shape[1]
        # Get masked indices  (N*V, num_masked)
        ids_mask = torch.argsort(enc_out.mask, dim=1)[:, num_visible:]
        # Reshape for cross-view target computation (patches only, no CLS)
        cls_by_view = batch["proj"][:, 0].view(N, V, D)
        patches_only = batch["proj"][:, num_prefix:]  # (N*V, num_visible, D)
        # Decoder
        rec = self.decoder(
            batch["embedding"][:, num_prefix:], enc_out.mask, output_masked_only=False
        )
        rec_loss = self.rec_loss(rec.float(), batch["image"], enc_out.mask)
        if batch["batch_idx"] == 0:
            # save samples with masks
            fig = plot_masked_images(batch["image"][0], enc_out.mask[:V])
            self.logger.experiment.log({"samples": wandb.Image(fig)})
            plt.close(fig)
            grid_size = (
                1,
                cfg.resolution // cfg.patch_size,
                cfg.resolution // cfg.patch_size,
            )
            rec_imag = spt.utils.unpatchify(
                rec[:V],
                patch_size=(3, cfg.patch_size, cfg.patch_size),
                grid_size=grid_size,
            ).detach()
            fig = plot_mae_reconstruction(
                batch["image"][:V], rec_imag, enc_out.mask[:V]
            )
            self.logger.experiment.log({"rec": wandb.Image(fig)})
            plt.close(fig)

        loss = rec_loss

        log_dict = {
            "time_stamp": time.time(),
            "loss/mae": rec_loss.item(),
            "loss/total": loss.item(),
        }

        batch["loss"] = loss
        self.log_dict(log_dict)
        return batch

    encoder = spt.backbone.MaskedEncoder(
        cfg.setdefault("backbone"),
        masking=spt.backbone.PatchMasking(
            mask_ratio=cfg.get("mask_ratio", 0.5),
            block_size=cfg.get("block_size", 1),
            crop_aspect_ratio=(0.6, 1.5),
            crop_ratio=cfg.get("crop_ratio", 0.9),
        ),
        patch_size=cfg.get("patch_size", None),
        img_size=(cfg.get("resolution"), cfg.get("resolution")),
        pretrained=False,
    )
    projector = MLP(
        cfg.get("embedding_dim"),
        cfg.get("projector_width") + [cfg.get("embedding_dim")],
        norm_layer=torch.nn.BatchNorm1d,
        inplace=True,
        bias=False,
    )

    grid = cfg.get("resolution") // cfg.get("patch_size")
    decoder = spt.backbone.MAEDecoder(
        embed_dim=cfg.get("embedding_dim"),  # Match your encoder
        decoder_embed_dim=512,  # Always 512 in MAE
        num_patches=grid**2,  # 14x14 for ViT
        output_dim=3 * cfg.get("patch_size") ** 2,
        depth=4,
        num_heads=16,
        mlp_ratio=4.0,
        pos_embed_type="sincos_2d",
    )

    # whitening GP
    sched = lambda opt, module: spt.optim.LinearWarmupCosineAnnealing(
        opt,
        peak_step=int(0.02 * module.trainer.estimated_stepping_batches),
        total_steps=module.trainer.estimated_stepping_batches,
        end_lr=cfg.get("lr") / 1000,
    )

    module = spt.Module(
        forward=forward,
        encoder=encoder,
        projector=projector,
        decoder=decoder,
        hparams=cfg,
        rec_loss=spt.utils.MAELoss(
            patch_size=cfg.get("patch_size"),
            loss_type="cosine",
            mask_only=False,
            patch_normalize=False,
        ),
        optim={
            "optimizer": partial(
                torch.optim.AdamW,
                lr=cfg.get("lr"),
                weight_decay=cfg.get("weight_decay"),
            ),
            "scheduler": sched,
        },
    )

    callbacks = []
    callbacks.append(
        spt.callbacks.OnlineProbe(
            module=module,
            name="embedding_probe",
            input="embedding",
            target="label",
            probe=spt.backbone.MultiHeadAttentiveProbe(
                embedding_dim=cfg.get("embedding_dim"), num_classes=num_classes
            ),
            loss=torch.nn.CrossEntropyLoss(),
        )
    )

    callbacks.append(
        spt.callbacks.OnlineProbe(
            module=module,
            name="proj_probe",
            input="proj",
            target="label",
            probe=spt.backbone.MultiHeadAttentiveProbe(
                embedding_dim=cfg.get("embedding_dim"), num_classes=num_classes
            ),
            loss=torch.nn.CrossEntropyLoss(),
        )
    )
    callbacks.append(
        pl.pytorch.callbacks.LearningRateMonitor(
            logging_interval="step", log_momentum=True, log_weight_decay=True
        )
    )

    logger = WandbLogger(project=cfg.get("wandb_project", "mila_1"))

    async_ckpt_io = AsyncCheckpointIO()

    trainer = pl.Trainer(
        max_epochs=cfg.get("max_epochs"),
        plugins=[async_ckpt_io],
        callbacks=callbacks,
        precision=cfg.get("precision"),
        logger=logger,
        sync_batchnorm=True,
        enable_checkpointing=False,
        check_val_every_n_epoch=cfg.get("check_val_every_n_epoch"),
        num_nodes=cfg.get("num_nodes"),
    )
    manager = spt.Manager(
        trainer=trainer, module=module, data=data_module, compile=False
    )
    manager()


if __name__ == "__main__":
    main()
