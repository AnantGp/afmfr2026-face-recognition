from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class ArcFaceHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, margin: float, scale: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings.float(), dim=1)
        weight = F.normalize(self.weight.float(), dim=1)
        cosine = F.linear(embeddings, weight).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cosine)
        target_logits = torch.cos(theta + self.margin)
        one_hot = F.one_hot(labels, num_classes=weight.shape[0]).to(dtype=cosine.dtype)
        logits = cosine * (1.0 - one_hot) + target_logits * one_hot
        return logits * self.scale


def clip_preservation_loss(current: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    current = F.normalize(current.float(), dim=1)
    original = F.normalize(original.float(), dim=1)
    return (1.0 - (current * original).sum(dim=1)).mean()
