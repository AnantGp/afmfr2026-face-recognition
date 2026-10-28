from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ensure_dir, load_config
from .data import (
    FaceManifestDataset,
    IdentityBatchSampler,
    build_eval_transform,
    build_train_transform,
    read_manifest,
)
from .losses import ArcFaceHead, clip_preservation_loss
from .lora import trainable_parameter_groups
from .metrics import evaluate_split
from .modeling import load_clip_visual


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: Path, model, head, optimizer, cfg, epoch: int, step: int, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "head_state": head.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "cfg": cfg,
            "epoch": epoch,
            "step": step,
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(cfg["paths"]["output_dir"])
    with (out_dir / "config.resolved.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    image_size = int(cfg["data"]["image_size"])
    train_rows = read_manifest(cfg["paths"]["manifest"], cfg["data"]["split_train"])
    val_same_rows = read_manifest(cfg["paths"]["manifest"], cfg["data"]["split_val_same"])
    val_id_rows = read_manifest(cfg["paths"]["manifest"], cfg["data"]["split_val_identity"])
    num_classes = len({row.identity for row in train_rows})
    if not train_rows:
        raise RuntimeError("No training rows found")

    train_dataset = FaceManifestDataset(train_rows, build_train_transform(image_size))
    eval_same_dataset = FaceManifestDataset(val_same_rows, build_eval_transform(image_size))
    eval_id_dataset = FaceManifestDataset(val_id_rows, build_eval_transform(image_size))

    max_steps_per_epoch = int(cfg["train"].get("max_steps_per_epoch", 0))
    sampler = IdentityBatchSampler(
        train_rows,
        identities_per_batch=int(cfg["train"]["identities_per_batch"]),
        images_per_identity=int(cfg["train"]["images_per_identity"]),
        seed=int(cfg["seed"]),
        max_batches=max_steps_per_epoch,
    )
    loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
    )

    model_cfg = cfg["model"]
    model = load_clip_visual(
        backbone_name=model_cfg["backbone_name"],
        device=device,
        download_root=cfg["paths"]["clip_download_root"],
        lora_rank=int(model_cfg["lora_rank"]),
        lora_alpha=float(model_cfg["lora_alpha"]),
        lora_dropout=float(model_cfg["lora_dropout"]),
        lora_targets=model_cfg["lora_targets"],
        train_ln=bool(model_cfg["train_ln"]),
        train_visual_proj=bool(model_cfg["train_visual_proj"]),
    ).to(device)

    frozen_original = None
    preservation_weight = float(cfg["loss"].get("clip_preservation_weight", 0.0))
    if preservation_weight > 0:
        frozen_original = load_clip_visual(
            backbone_name=model_cfg["backbone_name"],
            device=device,
            download_root=cfg["paths"]["clip_download_root"],
            lora_rank=None,
        ).to(device)
        frozen_original.eval()
        for param in frozen_original.parameters():
            param.requires_grad = False

    embedding_dim = int(getattr(model, "output_dim", 768))
    head = ArcFaceHead(
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        margin=float(cfg["loss"]["arcface_margin"]),
        scale=float(cfg["loss"]["arcface_scale"]),
    ).to(device)

    param_groups = trainable_parameter_groups(
        model,
        lr_lora=float(cfg["train"]["lr_lora"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    param_groups.append(
        {
            "params": head.parameters(),
            "lr": float(cfg["train"]["lr_head"]),
            "weight_decay": float(cfg["train"]["weight_decay"]),
        }
    )
    optimizer = torch.optim.AdamW(param_groups)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"]["amp"]) and device.type == "cuda")

    grad_accum = int(cfg["train"]["grad_accum_steps"])
    global_step = 0
    best_eer = float("inf")

    for epoch in range(int(cfg["train"]["epochs"])):
        model.train()
        head.train()
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}")
        for batch_idx, (images, labels, _, _) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=bool(cfg["train"]["amp"]) and device.type == "cuda"):
                embeddings = model(images)
                logits = head(embeddings, labels)
                loss = F.cross_entropy(logits, labels)
                if frozen_original is not None and preservation_weight > 0:
                    with torch.no_grad():
                        original = frozen_original(images)
                    loss = loss + preservation_weight * clip_preservation_loss(embeddings, original)
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            if (batch_idx + 1) % grad_accum == 0:
                if float(cfg["train"]["grad_clip_norm"]) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if global_step and global_step % int(cfg["train"]["log_every"]) == 0:
                pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}", step=global_step)

        same_metrics = evaluate_split(
            model,
            eval_same_dataset,
            val_same_rows,
            device,
            batch_size=64,
            max_images_per_identity=int(cfg["data"]["max_val_images_per_identity"]),
            max_identities=int(cfg["data"]["max_val_identities"]),
            seed=int(cfg["seed"]) + epoch,
        )
        id_metrics = evaluate_split(
            model,
            eval_id_dataset,
            val_id_rows,
            device,
            batch_size=64,
            max_images_per_identity=int(cfg["data"]["max_val_images_per_identity"]),
            max_identities=int(cfg["data"]["max_val_identities"]),
            seed=int(cfg["seed"]) + epoch,
        )
        metrics = {"val_same": same_metrics, "val_identity": id_metrics}
        print(json.dumps(metrics, indent=2))
        save_checkpoint(out_dir / "last.pt", model, head, optimizer, cfg, epoch, global_step, metrics)
        identity_eer = id_metrics.get("eer", float("inf"))
        if identity_eer < best_eer:
            best_eer = identity_eer
            save_checkpoint(out_dir / "best_identity.pt", model, head, optimizer, cfg, epoch, global_step, metrics)


if __name__ == "__main__":
    main()
