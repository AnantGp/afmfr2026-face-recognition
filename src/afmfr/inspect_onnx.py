from __future__ import annotations

import argparse
from pathlib import Path

import onnx


FORBIDDEN_MARKERS = ("lora", "arcface", "classifier", "head", "projection_mlp")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_path", type=Path)
    args = parser.parse_args()

    model = onnx.load(str(args.onnx_path))
    names = []
    for node in model.graph.node:
        names.append(node.name.lower())
        names.extend(inp.lower() for inp in node.input)
        names.extend(out.lower() for out in node.output)

    offenders = sorted({marker for marker in FORBIDDEN_MARKERS if any(marker in name for name in names)})
    print(f"inputs: {[i.name for i in model.graph.input]}")
    print(f"outputs: {[o.name for o in model.graph.output]}")
    print(f"nodes: {len(model.graph.node)}")
    if offenders:
        raise RuntimeError(f"Forbidden training-only markers found in ONNX graph: {offenders}")
    print("ONNX graph inspection passed")


if __name__ == "__main__":
    main()
