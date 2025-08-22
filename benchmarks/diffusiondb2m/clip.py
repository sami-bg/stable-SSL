import os
import torch
import torch.nn as nn
import lightning as pl

from lightning.pytorch.loggers import WandbLogger
from transformers import AutoTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection
import stable_pretraining as spt
from functools import partial
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from stable_pretraining.losses import InfoNCELoss
import torch.nn.functional as F

# Batch size per GPU

num_devices = 8
global_batch = 4096
batch_size = global_batch // num_devices
lr = 5e-4

tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

def tokenize(text: str, tokenizer: AutoTokenizer):
    data = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True
    )
    return data["input_ids"].squeeze(0), data["attention_mask"].squeeze(0)

image_transform = spt.data.transforms.Compose(
    spt.data.transforms.Resize((224, 224)),
    spt.data.transforms.ToImage(
        mean=[0.48145466, 0.4578275, 0.40821073], # TODO
        std=[0.26862954, 0.26130258, 0.27577711]  # TODO
    ),
    spt.data.transforms.LambdaTransform(
        fn=partial(tokenize, tokenizer=tokenizer),
        source="prompt",
        targets=("tokenized_prompt", "attention_mask")
    )
)

# Load DiffusionDB dataset
train_dataset = spt.data.HFDataset(
    "poloclub/diffusiondb",
    "2m_all",  # Change to "2m_all" for full 2M dataset
    split="train",
    trust_remote_code=True,
    transform=image_transform,
    remove_columns=["timestamp"],
    num_proc=64
)
train_dataset = spt.data.Subset(train_dataset, range(100, len(train_dataset)))
val_dataset = spt.data.Subset(train_dataset, range(100))

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=16,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=16,
)

data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)
vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32", trust_remote_code=True)
text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32", trust_remote_code=True)

def forward(self: spt.Module, batch: dict, stage: str) -> dict:
    out = {}
    vision_model: CLIPVisionModelWithProjection = self.vision_model
    text_model: CLIPTextModelWithProjection = self.text_model
    clip_loss: InfoNCELoss = self.clip_loss
    
    # Get image embeddings
    vision_outputs = vision_model(pixel_values=batch["image"])
    image_embeds = vision_outputs.image_embeds
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    
    # Get text embeddings
    text_outputs = text_model(
        input_ids=batch["tokenized_prompt"],
        attention_mask=batch["attention_mask"]
    )
    text_embeds = text_outputs.text_embeds
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    
    out["image_embeds"] = image_embeds
    out["text_embeds"] = text_embeds
    
    if self.training:
        out["loss"] = clip_loss(image_embeds, text_embeds)
    
    return out


class CLIPMonitor(pl.Callback):
    """Fixed-temp CLIP monitor with global (DDP) metrics via repo all_gather/all_reduce."""
    def __init__(self, log_every_n_steps: int = 50):
        super().__init__()
        self.every = log_every_n_steps
        self.scale = None  # 1 / temperature (constant)

    @torch.no_grad()
    def on_fit_start(self, trainer: pl.Trainer, pl_module):
        T = pl_module.clip_loss.temperature
        T = float(T.item()) if torch.is_tensor(T) else float(T)
        self.scale = 1.0 / T
        if trainer.is_global_zero:
            trainer.logger.log_metrics(
                {"config/temperature": T, "config/logit_scale": self.scale},
                step=trainer.global_step,
            )

    @torch.no_grad()
    def _log(self, trainer: pl.Trainer, outputs: dict, stage: str):
        # Assumes outputs contain normalized or raw embeds; we normalize here.
        img = F.normalize(outputs["image_embeds"], dim=-1)  # [B, D]
        txt = F.normalize(outputs["text_embeds"], dim=-1)   # [B, D]

        logits = self.scale * (img @ txt.T)                  # [N, N]
        N = logits.size(0)
        diag = torch.arange(N, device=logits.device)

        probs    = logits.softmax(dim=1)                   # [N, N]
        pos_prob = probs[diag, diag]                       # [N]
        entropy  = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)

        # margin vs top negative per row
        neg = logits.masked_fill(torch.eye(N, dtype=torch.bool, device=logits.device), float("-inf"))
        top_neg = neg.max(dim=1).values
        margin  = logits[diag, diag] - top_neg

        # in-batch Recall@1 both directions
        r1_i2t = (logits.argmax(dim=1) == diag).float().mean()
        r1_t2i = (logits.argmax(dim=0) == diag).float().mean()

        # quick health on local batch
        cos_pos  = F.cosine_similarity(img, txt, dim=-1).mean()
        img_norm = img.norm(dim=-1).mean()
        txt_norm = txt.norm(dim=-1).mean()

        metrics = {
            f"{stage}/retrieval/R@1_i2t":   float(r1_i2t.detach().cpu()),
            f"{stage}/retrieval/R@1_t2i":   float(r1_t2i.detach().cpu()),
            f"{stage}/contrast/pos_prob":   float(pos_prob.mean().detach().cpu()),
            f"{stage}/contrast/margin":     float(margin.mean().detach().cpu()),
            f"{stage}/contrast/entropy":    float(entropy.mean().detach().cpu()),
            f"{stage}/align/cos_pos":       float(cos_pos.detach().cpu()),
            f"{stage}/embed/img_norm":      float(img_norm.detach().cpu()),
            f"{stage}/embed/txt_norm":      float(txt_norm.detach().cpu()),
        }
        if trainer.is_global_zero:
            trainer.logger.log_metrics(metrics, step=trainer.global_step)

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx):
        if trainer.global_step % self.every == 0:
            self._log(trainer, outputs, "train")

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx, dataloader_idx=0):
        self._log(trainer, outputs, "val")


module = spt.Module(
    vision_model=vision_model,
    text_model=text_model,
    forward=forward,
    clip_loss=spt.losses.InfoNCELoss(temperature=0.07),  # CLIP uses InfoNCE loss
    optim={
        "optimizer": {
            "type": "AdamW",
            "lr": lr,
            "weight_decay": 0.1,
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
            "total_steps": (total_step := len(train_dataloader)),
            "peak_step": 0.1,
        },
        "interval": "step",
    },
)

# linear_probe = spt.callbacks.OnlineProbe(
#     name="zero_shot_classification",
#     input="image_embeds",
#     target="text_embeds",  
#     probe=torch.nn.Linear(768, 10),
#     loss_fn=torch.nn.CosineEmbeddingLoss(),
#     metrics={
#         "cosine_sim": torchmetrics.CosineSimilarity(),
#     },
# )

# WandB logger
wandb_logger = WandbLogger(
    entity="samibg",
    project="ijepa-cifar10", # TODO
    name="clip-vit-b32",
    log_model=False,
)

trainer = pl.Trainer(
    max_epochs=8,
    num_sanity_val_steps=0,
    callbacks = [ 
        ModelCheckpoint(monitor="train/loss", mode="min", every_n_epochs=1, dirpath='/mnt/data/sami/stable-pretraining/checkpoints', save_top_k=1),
        LearningRateMonitor(logging_interval="step"),
        CLIPMonitor(log_every_n_steps=10)
    ],
    precision="bf16-mixed",
    logger=wandb_logger,
    enable_checkpointing=True,
    devices=num_devices,  # TODO 8 
    accelerator="gpu",
    strategy="ddp",
)

# Run training
manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()