import torch
import torch.nn.functional as F
from typing import Dict, List, Literal, Optional, Tuple, Union, Callable

from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging
import torchmetrics

from ..utils import get_data_from_batch_or_outputs
from .utils import EarlyStopping, format_metrics_as_dict


class CLIPZeroShot(Callback):
    """Zero-shot classification evaluator for CLIP-style models.

    This callback computes zero-shot predictions by:
    1) Encoding images with the provided image encoder (each validation batch)
    2) Computing cosine-similarity logits and taking argmax as predictions
    3) Updating provided metrics on predictions vs targets

    Args:
        name: Unique identifier for this callback instance (used as log prefix and registry key).
        image_key: Key in batch or outputs containing input images or precomputed image features.
        tokens_key: Key in batch containing tokenized text.
        class_key: Key in batch containing ground-truth class indices (0..C-1, aligned with class_names order).
        class_names: List of class names in index order.
        image_backbone: Module/callable to encode images into embeddings.
        text_backbone: Module/callable to encode tokenized text into embeddings.
        tokenizer_fn: Callable that maps List[str] -> tensor of shape (N, T) and 'attention_mask' tensor of shape (N, T).
        metrics: Dict of torchmetrics to compute on validation (e.g., {"top1": MulticlassAccuracy(...)}).
    """
    def __init__(
        self,
        name: str,
        image_key: str,
        class_key: str,
        class_names: list[str],
        image_backbone: torch.nn.Module,
        text_backbone: torch.nn.Module,
        tokenizer_fn: Callable[[List[str]], Tuple[torch.Tensor, torch.Tensor]],
        metrics: Optional[Union[dict, tuple, list, torchmetrics.Metric]] = None,
        early_stopping: Optional[EarlyStopping] = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.image_key = image_key
        self.class_key = class_key
        self.class_names = class_names
        self.class_map = {c: i for i, c in enumerate(class_names)}
        self.image_backbone = image_backbone
        self.text_backbone = text_backbone
        self.tokenizer_fn = tokenizer_fn
        self.early_stopping = early_stopping

        self._train_metrics = None
        self._val_metrics = None

        # Format metrics
        self.metrics_config = metrics

        logging.info(f"Initialized CLIPZeroShot callback: {name}")
        logging.info(f"  - Image key: {image_key}")
        logging.info(f"  - Number of classes: {len(class_names)}")
        logging.info(f"  - Class names: [{', '.join(class_names[:5])}...]")
        logging.info(f"  - Image backbone: {image_backbone.__class__.__name__}")
        logging.info(f"  - Text backbone: {text_backbone.__class__.__name__}")


    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Initialize optimizer, scheduler, and metrics."""
        # Call parent setup for module/optimizer/scheduler
        super().setup(trainer, pl_module, stage)

        # Setup metrics
        logging.info(f"{self.name}: Setting up metrics")
        if not hasattr(pl_module, "_callbacks_metrics"):
            pl_module._callbacks_metrics = {}
        pl_module._callbacks_metrics[self.name] = format_metrics_as_dict(
            self.metrics_config
        )

        self._train_metrics = pl_module._callbacks_metrics[self.name]["_train"]
        self._val_metrics = pl_module._callbacks_metrics[self.name]["_val"]
        # TODO Doesn't seem too right. e.g. for INET-22k this would take forever cause it's not batched?
        self.class_tokens = [self.tokenizer_fn(c)[0] for c in self.class_names]


    def _compute_metrics(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict,
        batch: Dict,
        stage: Literal["train", "val"],
    ) -> None:
        image = get_data_from_batch_or_outputs(
            self.image_key, batch, outputs, caller_name=self.name
        )
        classes = get_data_from_batch_or_outputs(
            self.class_key, batch, outputs, caller_name=self.name
        ) # eg dog, cat
        
        if image is None:
            return
        
        image = image.to(device=pl_module.device)
        # if classes are ["cat", "dog", ...] map "dog" -> index 1 -> class_tokens[1]
        class_idxs = [self.class_map[c] for c in classes]
        class_tokens = torch.stack([
            self.class_tokens[i] for i in class_idxs
        ], dim=0).to(device=pl_module.device)
        
        with torch.no_grad():
            image_features = self.image_backbone(image)
            image_features = F.normalize(image_features, dim=-1)
            class_features = self.text_backbone(class_tokens)
            class_features = F.normalize(class_features, dim=-1)

            logits = (image_features @ class_features.T)
            preds = torch.argmax(logits, dim=1)
        
        prediction_key = f"{self.name}_preds"
        if prediction_key not in batch:
            batch[prediction_key] = preds.detach()
        
        logs = {}
        for metric_name, metric in pl_module._callbacks_metrics[self.name][
            f"_{stage}"
        ].items():
            metric(preds.detach(), torch.tensor(class_idxs))
            logs[f"{stage}/{self.name}_{metric_name}"] = metric

        pl_module.log_dict(logs, on_step=True, on_epoch=True)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict,
        batch: Dict,
        batch_idx: int,
    ) -> None:
        return self._compute_metrics(trainer, pl_module, outputs, batch, "train")

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict,
        batch: Dict,
        batch_idx: int,
    ) -> None:
        return self._compute_metrics(trainer, pl_module, outputs, batch, "val")

