"""I-JEPA: Image-based Joint-Embedding Predictive Architecture.

Self-supervised learning via predicting target patch representations
from context patch representations using a lightweight predictor.

References:
    Assran et al. "Self-Supervised Learning from Images with a
    Joint-Embedding Predictive Architecture." CVPR 2023.
    https://arxiv.org/abs/2301.08243

Example::

    from stable_pretraining.backbone import IJEPA
    from stable_pretraining.callbacks import TeacherStudentCallback
    import lightning as pl

    # Create model
    model = IJEPA(
        encoder_name="vit_base_patch16_224",
        predictor_embed_dim=384,
        predictor_depth=6,
        num_targets=4,
    )

    # Training with PyTorch Lightning
    trainer = pl.Trainer(
        max_epochs=300,
        callbacks=[TeacherStudentCallback()],  # Handles EMA updates
    )
    trainer.fit(model, dataloader)

    # Get encoder for downstream tasks
    encoder = model.encoder.student
"""

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_pretraining.backbone import (
    FlexibleTransformer,
    IJEPAMasking,
    MaskedEncoder,
    TeacherStudentWrapper,
)
from stable_pretraining import Module


@dataclass
class IJEPAOutput:
    """Output from IJEPA forward pass.

    :ivar loss: Prediction loss (0 in eval mode)
    :ivar predictions: Predicted representations [B, N_tgt, D] (or context in eval)
    :ivar targets: Target representations [B, N_tgt, D] (or context in eval)
    :ivar num_targets: Number of target patches (0 in eval)
    :ivar num_context: Number of context patches (all patches in eval)
    """

    loss: torch.Tensor
    predictions: torch.Tensor
    targets: torch.Tensor
    num_targets: int
    num_context: int


class IJEPA(Module):
    """I-JEPA: Image-based Joint-Embedding Predictive Architecture.

    Architecture:
        - **Context Encoder** (student): Encodes visible/context patches
        - **Target Encoder** (teacher): EMA copy, encodes target patches
        - **Predictor**: Lightweight transformer predicting targets from context

    The context encoder is wrapped with :class:`TeacherStudentWrapper`, enabling
    automatic EMA updates via :class:`TeacherStudentCallback`.

    :param encoder_name: timm model name (e.g., "vit_base_patch16_224")
    :param predictor_embed_dim: Predictor hidden dimension (default: 384)
    :param predictor_depth: Number of predictor blocks (default: 6)
    :param num_targets: Number of target blocks to sample (default: 4)
    :param target_scale: (min, max) fraction of patches per target block
    :param target_aspect_ratio: (min, max) aspect ratio of target blocks
    :param context_scale: (min, max) fraction of non-target patches as context
    :param ema_decay_start: Initial EMA decay (default: 0.996)
    :param ema_decay_end: Final EMA decay (default: 1.0)
    :param pretrained: Load pretrained encoder weights

    Example::

        # Basic usage
        model = IJEPA("vit_base_patch16_224")
        images = torch.randn(4, 3, 224, 224)

        # Training mode: predicts masked targets
        model.train()
        output = model(images)
        output.loss.backward()

        # Eval mode: encodes all patches (no masking)
        model.eval()
        output = model(images)
        features = output.predictions  # [B, N, D]

    Example with Lightning::

        import lightning as pl
        from stable_pretraining.callbacks import TeacherStudentCallback


        class IJEPALightning(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.model = IJEPA("vit_base_patch16_224")

            def training_step(self, batch, batch_idx):
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                output = self.model(images)
                self.log("loss", output.loss)
                return output.loss

            def configure_optimizers(self):
                return torch.optim.AdamW(self.parameters(), lr=1.5e-4)


        trainer = pl.Trainer(callbacks=[TeacherStudentCallback()])
        trainer.fit(IJEPALightning(), dataloader)

    Note:
        - Use :class:`TeacherStudentCallback` to handle EMA updates automatically
        - In eval mode, ``num_targets=0`` and all patches are returned as context
        - Access trained encoder via ``model.encoder.student``
    """

    def __init__(
        self,
        encoder_name: str = "vit_base_patch16_224",
        predictor_embed_dim: int = 384,
        predictor_depth: int = 6,
        num_targets: int = 4,
        target_scale: Tuple[float, float] = (0.15, 0.2),
        target_aspect_ratio: Tuple[float, float] = (0.75, 1.5),
        context_scale: Tuple[float, float] = (0.85, 1.0),
        ema_decay_start: float = 0.996,
        ema_decay_end: float = 1.0,
        pretrained: bool = False,
    ):
        super().__init__()

        # Encoder with EMA wrapper (enables TeacherStudentCallback)
        base_encoder = MaskedEncoder(encoder_name, masking=None, pretrained=pretrained)
        self.encoder = TeacherStudentWrapper(
            base_encoder,
            ema_decay_start=ema_decay_start,
            ema_decay_end=ema_decay_end,
        )

        embed_dim = base_encoder.embed_dim
        num_patches = base_encoder.default_grid_h * base_encoder.default_grid_w

        # Lightweight predictor: cross-attention from target queries to context
        self.predictor = FlexibleTransformer(
            input_dim=embed_dim,
            hidden_dim=predictor_embed_dim,
            output_dim=embed_dim,
            num_patches=num_patches,
            depth=predictor_depth,
            num_heads=max(1, predictor_embed_dim // 64),
            self_attn=True,
            cross_attn=True,
            use_adaln=False,
            num_prefix_tokens=0,
            zero_init_output=True,
        )

        # Learnable query token for target positions
        self.target_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.target_query, std=0.02)

        # I-JEPA multi-block masking
        self.masking = IJEPAMasking(
            num_targets=num_targets,
            target_scale=target_scale,
            target_aspect_ratio=target_aspect_ratio,
            context_scale=context_scale,
        )

        self.embed_dim = embed_dim

    def _encode(
        self,
        patches: torch.Tensor,
        indices: torch.Tensor,
        grid_h: int,
        grid_w: int,
        encoder: MaskedEncoder,
    ) -> torch.Tensor:
        """Encode patches at specified indices.

        :param patches: All patch embeddings [B, N, D]
        :param indices: Indices to encode [B, K]
        :param grid_h: Patch grid height
        :param grid_w: Patch grid width
        :param encoder: Encoder to use (student or teacher)
        :return: Encoded representations [B, K, D]
        """
        B, _, D = patches.shape
        x = torch.gather(patches, 1, indices.unsqueeze(-1).expand(-1, -1, D))

        # Add positional embeddings
        _, pos = self.encoder.student._get_pos_embed(grid_h, grid_w)
        x = x + torch.gather(
            pos.expand(B, -1, -1), 1, indices.unsqueeze(-1).expand(-1, -1, D)
        )

        # Forward through transformer
        x = encoder.vit.pos_drop(x)
        x = encoder.vit.blocks(x)
        return encoder.vit.norm(x)

    def forward(self, images: torch.Tensor) -> IJEPAOutput:
        """Forward pass.

        In training mode:
            - Samples target blocks and context region via :class:`IJEPAMasking`
            - Encodes context through student, targets through teacher (EMA)
            - Predicts target representations from context
            - Returns smooth L1 loss between predictions and targets

        In eval mode:
            - No masking, all patches treated as context
            - Returns encoded features with zero loss

        :param images: Input images [B, C, H, W]
        :return: :class:`IJEPAOutput` with loss and representations
        """
        B = images.shape[0]
        grid_h, grid_w = self.encoder.student._get_grid_size(images)
        patches = self.encoder.student.patch_embed(images)

        # Apply masking (returns all patches as context in eval mode)
        mask_out = self.masking(patches, grid_h, grid_w)

        if self.training:
            # Encode context (student) and targets (teacher, no grad)
            context = self._encode(
                patches, mask_out.context_idx, grid_h, grid_w, self.encoder.student
            )

            with torch.no_grad():
                targets = self._encode(
                    patches, mask_out.target_idx, grid_h, grid_w, self.encoder.teacher
                )

            # Predict target representations via cross-attention
            N_tgt = mask_out.target_idx.shape[1]
            queries = self.target_query.expand(B, N_tgt, -1)
            predictions = self.predictor(
                context=context,
                queries=queries,
                context_idx=mask_out.context_idx,
                query_idx=mask_out.target_idx,
            )

            loss = F.smooth_l1_loss(predictions, targets, beta=2.0)
        else:
            # Eval: encode all patches through student
            context = self._encode(
                patches, mask_out.context_idx, grid_h, grid_w, self.encoder.student
            )
            predictions = context
            targets = context
            loss = torch.tensor(0.0, device=images.device)

        return IJEPAOutput(
            loss=loss,
            predictions=predictions,
            targets=targets,
            num_targets=mask_out.target_idx.shape[1],
            num_context=mask_out.context_idx.shape[1],
        )
