from __future__ import annotations

from pathlib import Path

import clip
import torch
from torch import nn

from .lora import LoRAConfig, apply_lora_to_clip_visual, mark_trainable_parameters


def load_clip_visual(
    backbone_name: str,
    device: torch.device,
    download_root: str | Path,
    lora_rank: int | None = None,
    lora_alpha: float = 1.0,
    lora_dropout: float = 0.0,
    lora_targets: str = "qv",
    train_ln: bool = False,
    train_visual_proj: bool = False,
) -> nn.Module:
    backbone, _ = clip.load(backbone_name, device=device, jit=False, download_root=str(download_root))
    for param in backbone.parameters():
        if param.dtype == torch.float16:
            param.data = param.data.float()

    visual = backbone.visual
    if lora_rank is not None and lora_rank > 0:
        apply_lora_to_clip_visual(
            visual,
            LoRAConfig(rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout, targets=lora_targets),
        )
    mark_trainable_parameters(visual, train_ln=train_ln, train_visual_proj=train_visual_proj)
    return visual


@torch.no_grad()
def embed_images(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    embeddings = model(images)
    return torch.nn.functional.normalize(embeddings.float(), dim=1)
