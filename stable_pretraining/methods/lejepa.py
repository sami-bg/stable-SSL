"""LeJEPA: Latent Embedding Joint-Embedding Predictive Architecture.

Self-supervised learning via multi-view invariance combined with a
sliced goodness-of-fit test (SIGReg) that pushes embeddings toward
an isotropic Gaussian.

References:
    Balestriero & LeCun. "LeJEPA: Provable and Scalable
    Self-Supervised Learning Without the Heuristics." 2025.
    https://arxiv.org/abs/2511.08544

Example::

    from stable_pretraining.methods.lejepa import LeJEPA

    model = LeJEPA("vit_base_patch16_224")

    global_images = [torch.randn(4, 3, 224, 224)] * 2
    all_images = [torch.randn(4, 3, 224, 224)] * 4
    model.train()
    output = model(global_images, all_images)
    output.loss.backward()

    model.eval()
    output = model(images=torch.randn(4, 3, 224, 224))
    features = output.embedding  # [N, D]
"""

from dataclasses import dataclass
from typing import List, Optional

import timm
import torch
import torch.nn as nn

from stable_pretraining import Module
from stable_pretraining.backbone import MLP


class EppsPulley(nn.Module):
    """Epps-Pulley goodness-of-fit test for univariate normality.

    Uses Simpson-rule quadrature to compare the empirical characteristic
    function of sorted 1-D samples against the standard-normal
    characteristic function.

    :param t_max: Upper integration bound.
    :param n_points: Number of quadrature nodes (must be odd).
    """

    def __init__(self, t_max: float = 3.0, n_points: int = 17):
        super().__init__()
        assert n_points % 2 == 1, "n_points must be odd for Simpson's rule"
        ts = torch.linspace(0, t_max, n_points)
        w = torch.ones(n_points)
        w[1:-1:2] = 4.0
        w[2:-1:2] = 2.0
        w *= t_max / (n_points - 1) / 3.0
        self.register_buffer("ts", ts)
        self.register_buffer("weights", w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Sorted samples [N, S] (N samples, S slices).
        :return: Per-slice statistic [S].
        """
        tx = self.ts[:, None, None] * x[None, :, :]  # [T, N, S]
        cos_tx = tx.cos().mean(dim=1)
        sin_tx = tx.sin().mean(dim=1)
        gauss_cf = (-0.5 * self.ts ** 2).exp()
        diff_sq = (cos_tx - gauss_cf[:, None]) ** 2 + sin_tx ** 2
        return (self.weights[:, None] * diff_sq).sum(dim=0)


class SlicedEppsPulley(nn.Module):
    """Sliced Epps-Pulley goodness-of-fit test for multivariate normality.

    Projects data onto random 1-D directions and averages the univariate
    Epps-Pulley statistics.  A synchronised step counter seeds the random
    projections so all DDP ranks sample identical directions.

    :param num_slices: Number of random 1-D projections.
    :param t_max: EP integration upper bound.
    :param n_points: EP quadrature nodes.
    """

    def __init__(self, num_slices: int = 256, t_max: float = 3.0, n_points: int = 17):
        super().__init__()
        self.num_slices = num_slices
        self.ep = EppsPulley(t_max=t_max, n_points=n_points)
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Embeddings [N, D].
        :return: Scalar mean EP statistic.
        """
        with torch.no_grad():
            step = self.global_step.clone()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(step, op=torch.distributed.ReduceOp.MAX)
            g = torch.Generator(device=x.device).manual_seed(step.item())
            A = torch.randn(x.size(-1), self.num_slices, device=x.device, generator=g)
            A = A / A.norm(p=2, dim=0)
            self.global_step.add_(1)

        proj = (x @ A).sort(dim=0).values
        return self.ep(proj).mean()



@dataclass
class LeJEPAOutput:
    """Output from LeJEPA forward pass.

    :ivar loss: Combined invariance + SIGReg loss (0 in eval mode).
    :ivar embedding: Backbone embeddings [V*N, D] (train) or [N, D] (eval).
    :ivar inv_loss: Invariance component.
    :ivar sigreg_loss: Epps-Pulley goodness-of-fit component.
    """

    loss: torch.Tensor
    embedding: torch.Tensor
    inv_loss: torch.Tensor
    sigreg_loss: torch.Tensor


class LeJEPA(Module):
    """LeJEPA: multi-view invariance + sliced Epps-Pulley SIGReg.

    Architecture:
        - **Backbone**: timm ViT (CLS-pooled, ``num_classes=0``)
        - **Projector**: MLP projection head
        - **Loss**: ``(1 - λ) * invariance + λ * SIGReg``

    Centers are computed from global-view projections only.  The invariance
    term penalises the MSE between each view's projection and the center.
    The SIGReg term is a sliced goodness-of-fit test that pushes
    projected embeddings toward an isotropic Gaussian, averaged over views.

    :param encoder_name: timm model name (e.g., ``"vit_base_patch16_224"``)
    :param projector: Optional projection head.  When ``None``, a 3-layer
        BN+ReLU MLP (``embed_dim → 2048 → 2048 → 128``) is created.
    :param num_slices: Random projection directions for the goodness-of-fit test (default: 256)
    :param t_max: EP integration upper bound (default: 3.0)
    :param n_points: EP quadrature nodes (default: 17)
    :param lamb: SIGReg weight λ (default: 0.02)
    :param pretrained: Load pretrained timm weights

    Example::

        model = LeJEPA("vit_base_patch16_224")
        images = torch.randn(4, 3, 224, 224)

        model.train()
        output = model(
            global_views=[images, images],
            all_views=[images, images, images, images],
        )
        output.loss.backward()

        model.eval()
        output = model(images=images)
        features = output.embedding  # [4, 768]

    Example with Lightning::

        import lightning as pl
        from stable_pretraining.methods.lejepa import LeJEPA

        class LeJEPALightning(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.model = LeJEPA("vit_base_patch16_224")

            def training_step(self, batch, batch_idx):
                views = [v["image"] for v in batch["views"]]
                output = self.model(global_views=views, all_views=views)
                self.log("loss", output.loss)
                return output.loss

            def configure_optimizers(self):
                return torch.optim.AdamW(self.parameters(), lr=1e-3)
    """

    def __init__(
        self,
        encoder_name: str = "vit_base_patch16_224",
        projector: Optional[nn.Module] = None,
        num_slices: int = 256,
        t_max: float = 3.0,
        n_points: int = 17,
        lamb: float = 0.02,
        pretrained: bool = False,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            encoder_name, pretrained=pretrained, num_classes=0
        )
        embed_dim = self.backbone.embed_dim

        if projector is None:
            projector = MLP(
                in_channels=embed_dim,
                hidden_channels=[2048, 2048, 128],
                norm_layer="batch_norm",
                activation_layer=nn.ReLU,
                inplace=True,
                dropout=0.0,
            )
        self.projector = projector

        self.sigreg = SlicedEppsPulley(
            num_slices=num_slices, t_max=t_max, n_points=n_points
        )
        self.lamb = lamb
        self.embed_dim = embed_dim

    @staticmethod
    def _compute_loss(
        global_projections: List[torch.Tensor],
        all_projections: List[torch.Tensor],
        sigreg: SlicedEppsPulley,
        lamb: float = 0.02,
    ):
        """Compute the LeJEPA loss.

        :param global_projections: List of V_g projected tensors [N, K] (global views).
        :param all_projections: List of V projected tensors [N, K] (all views).
        :param sigreg: SlicedEppsPulley module.
        :param lamb: SIGReg weight λ.
        :return: Tuple of (total_loss, inv_loss, sigreg_loss).
        """
        N = global_projections[0].shape[0]

        # Centers from global views
        centers = torch.stack(global_projections).mean(0)  # [N, K]

        # Invariance: MSE between each view projection and the center
        stacked = torch.stack(all_projections)  # [V, N, K]
        inv_loss = (centers - stacked).square().mean()

        # SIGReg: per-view Epps-Pulley, averaged
        sigreg_loss = torch.stack(
            [sigreg(all_projections[v]) for v in range(len(all_projections))]
        ).mean()

        loss = (1 - lamb) * inv_loss + lamb * sigreg_loss
        return loss, inv_loss, sigreg_loss

    def forward(
        self,
        global_views: Optional[List[torch.Tensor]] = None,
        all_views: Optional[List[torch.Tensor]] = None,
        images: Optional[torch.Tensor] = None,
    ) -> LeJEPAOutput:
        """Forward pass.

        In training mode:
            - Encodes and projects global views → mean projection = center
            - Encodes and projects all views
            - Invariance loss: MSE between each view projection and center
            - SIGReg loss: per-view Epps-Pulley, averaged

        In eval mode:
            - Encodes a single image tensor, returns embeddings with zero loss

        :param global_views: List of V_g image tensors [N, C, H, W].
        :param all_views: List of V image tensors [N, C, H, W] (superset of global).
        :param images: Single image tensor [N, C, H, W] for eval.
        :return: :class:`LeJEPAOutput` with loss and embeddings.
        """
        if self.training:
            N = global_views[0].shape[0]

            # Embed + project global views → centers
            g_emb = self.projector(self.backbone(torch.cat(global_views, dim=0)))
            g_projections = list(g_emb.reshape(len(global_views), N, -1))

            # Embed + project all views
            a_cat = torch.cat(all_views, dim=0)
            a_backbone = self.backbone(a_cat)
            a_proj = self.projector(a_backbone)
            a_projections = list(a_proj.reshape(len(all_views), N, -1))

            loss, inv_loss, sigreg_loss = self._compute_loss(
                g_projections, a_projections, self.sigreg, self.lamb
            )

            return LeJEPAOutput(
                loss=loss,
                embedding=a_backbone,
                inv_loss=inv_loss,
                sigreg_loss=sigreg_loss,
            )
        else:
            embedding = self.backbone(images)
            zero = torch.tensor(0.0, device=images.device)

            return LeJEPAOutput(
                loss=zero,
                embedding=embedding,
                inv_loss=zero,
                sigreg_loss=zero,
            )