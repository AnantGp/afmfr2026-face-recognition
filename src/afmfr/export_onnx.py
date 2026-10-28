from __future__ import annotations

import argparse
from dataclasses import dataclass
import types
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .config import load_config
from .lora import merge_lora_in_clip_visual
from .modeling import load_clip_visual


@dataclass(frozen=True)
class OnnxCompareReport:
    max_abs: float
    mean_abs: float
    max_relative_l2: float
    min_cosine: float


def force_onnx_friendly_attention(model: torch.nn.Module) -> None:
    """Avoid fused scaled_dot_product_attention during legacy ONNX export."""

    def attention(self, x):
        attn_mask = self.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=x.dtype, device=x.device)
        return self.attn(x, x, x, need_weights=True, attn_mask=attn_mask)[0]

    for block in model.transformer.resblocks:
        block.attention = types.MethodType(attention, block)


def export_onnx(model: torch.nn.Module, output_path: Path, device: torch.device, image_size: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, image_size, image_size, device=device, dtype=torch.float32)
    torch.onnx.export(
        model=model,
        args=dummy_input,
        f=str(output_path),
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


def compare_onnx(
    model: torch.nn.Module,
    onnx_path: Path,
    device: torch.device,
    image_size: int,
    batches: int = 3,
    batch_size: int = 2,
) -> OnnxCompareReport:
    model.eval()
    providers = ["CUDAExecutionProvider"] if device.type == "cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    max_abs = 0.0
    abs_sum = 0.0
    abs_count = 0
    max_relative_l2 = 0.0
    min_cosine = 1.0
    for _ in range(batches):
        x = torch.randn(batch_size, 3, image_size, image_size, device=device, dtype=torch.float32)
        with torch.no_grad():
            torch_out = model(x).detach().cpu().numpy().astype(np.float64)
        ort_out = session.run([output_name], {input_name: x.cpu().numpy()})[0].astype(np.float64)
        if not np.isfinite(ort_out).all():
            raise RuntimeError("ONNX Runtime produced non-finite outputs")

        diff = np.abs(torch_out - ort_out)
        max_abs = max(max_abs, float(diff.max()))
        abs_sum += float(diff.sum())
        abs_count += int(diff.size)

        torch_norm = np.linalg.norm(torch_out, axis=1)
        ort_norm = np.linalg.norm(ort_out, axis=1)
        relative_l2 = np.linalg.norm(torch_out - ort_out, axis=1) / np.maximum(torch_norm, 1e-12)
        cosine = np.sum(torch_out * ort_out, axis=1) / np.maximum(torch_norm * ort_norm, 1e-12)
        max_relative_l2 = max(max_relative_l2, float(relative_l2.max()))
        min_cosine = min(min_cosine, float(cosine.min()))

    return OnnxCompareReport(
        max_abs=max_abs,
        mean_abs=abs_sum / max(abs_count, 1),
        max_relative_l2=max_relative_l2,
        min_cosine=min_cosine,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    checkpoint = torch.load(args.checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"State mismatch. missing={missing}, unexpected={unexpected}")

    merged = merge_lora_in_clip_visual(model)
    print(f"Merged {merged} LoRA attention modules")
    force_onnx_friendly_attention(model)
    model.eval()

    export_onnx(model, args.output, device, int(cfg["data"]["image_size"]))
    report = compare_onnx(model, args.output, device, int(cfg["data"]["image_size"]))
    print(
        "PyTorch vs ONNX "
        f"max_abs={report.max_abs:.6f} "
        f"mean_abs={report.mean_abs:.8f} "
        f"max_relative_l2={report.max_relative_l2:.8f} "
        f"min_cosine={report.min_cosine:.8f}"
    )
    if report.max_abs > 1e-2 or report.mean_abs > 5e-4 or report.max_relative_l2 > 3e-3 or report.min_cosine < 0.99999:
        raise RuntimeError("ONNX export differs too much from PyTorch output")


if __name__ == "__main__":
    main()
