from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image

from .config import load_config
from .data import build_eval_transform, read_manifest
from .export_onnx import force_onnx_friendly_attention
from .inspect_onnx import FORBIDDEN_MARKERS
from .lora import merge_lora_in_clip_visual
from .modeling import load_clip_visual


def inspect_graph(onnx_path: Path) -> dict:
    model = onnx.load(str(onnx_path))
    names = []
    for node in model.graph.node:
        names.append(node.name.lower())
        names.extend(inp.lower() for inp in node.input)
        names.extend(out.lower() for out in node.output)

    offenders = sorted({marker for marker in FORBIDDEN_MARKERS if any(marker in name for name in names)})
    return {
        "ir_version": model.ir_version,
        "opsets": {op.domain: op.version for op in model.opset_import},
        "inputs": [i.name for i in model.graph.input],
        "outputs": [o.name for o in model.graph.output],
        "nodes": len(model.graph.node),
        "forbidden_markers": offenders,
    }


def load_reference_model(cfg: dict, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"State mismatch. missing={missing}, unexpected={unexpected}")
    merge_lora_in_clip_visual(model)
    force_onnx_friendly_attention(model)
    model.eval()
    return model


def load_sample_batch(cfg: dict, samples: int, split: str, seed: int) -> torch.Tensor:
    transform = build_eval_transform(int(cfg["data"]["image_size"]))
    rows = read_manifest(cfg["paths"]["manifest"], split)
    if len(rows) < samples:
        raise RuntimeError(f"Requested {samples} samples from {split}, but only found {len(rows)}")
    rng = random.Random(seed)
    selected = rng.sample(rows, samples)
    images = []
    for row in selected:
        image = Image.open(row.image_path).convert("RGB")
        images.append(transform(image))
    return torch.stack(images, dim=0)


def compare_outputs(
    model: torch.nn.Module,
    onnx_path: Path,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int,
    provider: str,
) -> dict:
    if provider == "cpu":
        providers = ["CPUExecutionProvider"]
    elif provider == "cuda":
        providers = ["CUDAExecutionProvider"]
    elif device.type == "cuda":
        providers = ["CUDAExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    torch_chunks = []
    ort_chunks = []
    with torch.no_grad():
        for start in range(0, images.shape[0], batch_size):
            batch = images[start : start + batch_size].to(device)
            torch_out = model(batch).detach().cpu().numpy().astype(np.float64)
            ort_out = session.run([output_name], {input_name: batch.cpu().numpy()})[0].astype(np.float64)
            torch_chunks.append(torch_out)
            ort_chunks.append(ort_out)

    torch_out = np.concatenate(torch_chunks, axis=0)
    ort_out = np.concatenate(ort_chunks, axis=0)
    if not np.isfinite(ort_out).all():
        raise RuntimeError("ONNX Runtime produced non-finite outputs")

    diff = np.abs(torch_out - ort_out)
    torch_norm = np.linalg.norm(torch_out, axis=1)
    ort_norm = np.linalg.norm(ort_out, axis=1)
    relative_l2 = np.linalg.norm(torch_out - ort_out, axis=1) / np.maximum(torch_norm, 1e-12)
    cosine = np.sum(torch_out * ort_out, axis=1) / np.maximum(torch_norm * ort_norm, 1e-12)
    return {
        "samples": int(images.shape[0]),
        "provider": session.get_providers()[0],
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_relative_l2": float(relative_l2.max()),
        "mean_relative_l2": float(relative_l2.mean()),
        "min_cosine": float(cosine.min()),
        "mean_cosine": float(cosine.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--split", default="val_identity")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-abs-threshold", type=float, default=1e-2)
    parser.add_argument("--mean-abs-threshold", type=float, default=5e-4)
    parser.add_argument("--relative-l2-threshold", type=float, default=3e-3)
    parser.add_argument("--cosine-threshold", type=float, default=0.99999)
    parser.add_argument("--provider", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.provider == "cpu":
        device = torch.device("cpu")
    elif args.provider == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = inspect_graph(args.onnx)
    if graph["forbidden_markers"]:
        raise RuntimeError(f"Forbidden training-only markers found in ONNX graph: {graph['forbidden_markers']}")

    model = load_reference_model(cfg, args.checkpoint, device)
    images = load_sample_batch(cfg, args.samples, args.split, int(cfg["seed"]))
    comparison = compare_outputs(model, args.onnx, images, device, args.batch_size, args.provider)
    report = {"graph": graph, "comparison": comparison}
    print(json.dumps(report, indent=2, sort_keys=True))

    if comparison["max_abs"] > args.max_abs_threshold:
        raise RuntimeError("ONNX max_abs exceeds threshold")
    if comparison["mean_abs"] > args.mean_abs_threshold:
        raise RuntimeError("ONNX mean_abs exceeds threshold")
    if comparison["max_relative_l2"] > args.relative_l2_threshold:
        raise RuntimeError("ONNX relative L2 exceeds threshold")
    if comparison["min_cosine"] < args.cosine_threshold:
        raise RuntimeError("ONNX cosine agreement below threshold")


if __name__ == "__main__":
    main()
