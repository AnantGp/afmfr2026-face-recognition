from __future__ import annotations

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


_MHA_FORWARD_PARAMS = set(inspect.signature(F.multi_head_attention_forward).parameters)


@dataclass
class LoRAConfig:
    rank: int
    alpha: float
    dropout: float
    targets: str = "qv"


class LoRAMultiheadAttention(nn.Module):
    def __init__(self, base: nn.MultiheadAttention, cfg: LoRAConfig) -> None:
        super().__init__()
        self.base = base
        self.cfg = cfg
        embed_dim = base.embed_dim
        self.scaling = cfg.alpha / cfg.rank
        self.dropout = nn.Dropout(cfg.dropout)
        self.use_q = "q" in cfg.targets
        self.use_v = "v" in cfg.targets

        if self.use_q:
            self.q_lora_a = nn.Parameter(torch.empty(cfg.rank, embed_dim))
            self.q_lora_b = nn.Parameter(torch.zeros(embed_dim, cfg.rank))
            nn.init.kaiming_uniform_(self.q_lora_a, a=math.sqrt(5))
        if self.use_v:
            self.v_lora_a = nn.Parameter(torch.empty(cfg.rank, embed_dim))
            self.v_lora_b = nn.Parameter(torch.zeros(embed_dim, cfg.rank))
            nn.init.kaiming_uniform_(self.v_lora_a, a=math.sqrt(5))

        for param in self.base.parameters():
            param.requires_grad = False

    def _delta(self, which: str) -> torch.Tensor:
        if which == "q":
            return (self.q_lora_b @ self.q_lora_a) * self.scaling
        if which == "v":
            return (self.v_lora_b @ self.v_lora_a) * self.scaling
        raise ValueError(which)

    def merged_in_proj_weight(self) -> torch.Tensor:
        weight = self.base.in_proj_weight
        q_weight, k_weight, v_weight = weight.chunk(3, dim=0)
        if self.use_q:
            q_weight = q_weight + self._delta("q").to(dtype=q_weight.dtype)
        if self.use_v:
            v_weight = v_weight + self._delta("v").to(dtype=v_weight.dtype)
        return torch.cat([q_weight, k_weight, v_weight], dim=0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        kwargs = {
            "query": query,
            "key": key,
            "value": value,
            "embed_dim_to_check": self.base.embed_dim,
            "num_heads": self.base.num_heads,
            "in_proj_weight": self.merged_in_proj_weight(),
            "in_proj_bias": self.base.in_proj_bias,
            "bias_k": self.base.bias_k,
            "bias_v": self.base.bias_v,
            "add_zero_attn": self.base.add_zero_attn,
            "dropout_p": self.base.dropout if self.training else 0.0,
            "out_proj_weight": self.base.out_proj.weight,
            "out_proj_bias": self.base.out_proj.bias,
            "training": self.training,
            "key_padding_mask": key_padding_mask,
            "need_weights": need_weights,
            "attn_mask": attn_mask,
        }
        if "average_attn_weights" in _MHA_FORWARD_PARAMS:
            kwargs["average_attn_weights"] = average_attn_weights
        if "is_causal" in _MHA_FORWARD_PARAMS:
            kwargs["is_causal"] = is_causal
        return F.multi_head_attention_forward(**kwargs)

    def merge_into_base(self) -> nn.MultiheadAttention:
        with torch.no_grad():
            self.base.in_proj_weight.copy_(self.merged_in_proj_weight())
        return self.base


def apply_lora_to_clip_visual(visual: nn.Module, cfg: LoRAConfig) -> list[str]:
    replaced = []
    for block_idx, block in enumerate(visual.transformer.resblocks):
        if not isinstance(block.attn, nn.MultiheadAttention):
            raise TypeError(f"Expected MultiheadAttention at block {block_idx}, found {type(block.attn)}")
        device = block.attn.in_proj_weight.device
        block.attn = LoRAMultiheadAttention(block.attn, cfg).to(device)
        replaced.append(f"visual.transformer.resblocks.{block_idx}.attn")
    return replaced


def merge_lora_in_clip_visual(visual: nn.Module) -> int:
    count = 0
    for block in visual.transformer.resblocks:
        if isinstance(block.attn, LoRAMultiheadAttention):
            block.attn = block.attn.merge_into_base()
            count += 1
    return count


def mark_trainable_parameters(visual: nn.Module, train_ln: bool, train_visual_proj: bool) -> None:
    for _, param in visual.named_parameters():
        param.requires_grad = False

    for module in visual.modules():
        if isinstance(module, LoRAMultiheadAttention):
            for name, param in module.named_parameters():
                param.requires_grad = "lora" in name

    if train_ln:
        for name, param in visual.named_parameters():
            if "ln_" in name or "ln_pre" in name or "ln_post" in name:
                param.requires_grad = True

    if train_visual_proj and hasattr(visual, "proj") and isinstance(visual.proj, nn.Parameter):
        visual.proj.requires_grad = True


def trainable_parameter_groups(visual: nn.Module, lr_lora: float, weight_decay: float) -> list[dict]:
    lora_params = []
    other_params = []
    for name, param in visual.named_parameters():
        if not param.requires_grad:
            continue
        if "lora" in name:
            lora_params.append(param)
        else:
            other_params.append(param)
    groups = []
    if lora_params:
        groups.append({"params": lora_params, "lr": lr_lora, "weight_decay": weight_decay})
    if other_params:
        groups.append({"params": other_params, "lr": lr_lora * 0.5, "weight_decay": weight_decay})
    return groups
