from __future__ import annotations

import argparse
import csv
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def infer_identity(path: Path, data_root: Path, identity_depth: int, prefix_depth: int) -> str:
    rel = path.relative_to(data_root)
    parts = rel.parts
    if len(parts) <= identity_depth:
        raise ValueError(f"Cannot infer identity from shallow path: {path}")
    identity = parts[identity_depth]
    if prefix_depth >= 0:
        if len(parts) <= prefix_depth:
            raise ValueError(f"Cannot infer identity prefix from shallow path: {path}")
        identity = f"{parts[prefix_depth]}:{identity}"
    return identity


def build_manifest(data_root: Path, track: str, output: Path, identity_depth: int, prefix_depth: int) -> None:
    rows = []
    for image_path in sorted(data_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        identity = infer_identity(image_path, data_root, identity_depth, prefix_depth)
        rows.append(
            {
                "image_path": str(image_path.resolve()),
                "identity": identity,
                "track": track,
                "split": "unsplit",
            }
        )

    if not rows:
        raise RuntimeError(f"No images found under {data_root}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "identity", "track", "split"])
        writer.writeheader()
        writer.writerows(rows)

    identities = {row["identity"] for row in rows}
    print(f"Wrote {len(rows)} images across {len(identities)} identities to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--track", required=True, choices=["track1", "track2"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--identity-depth",
        default=0,
        type=int,
        help="Which relative path component is the identity label. Default: first directory.",
    )
    parser.add_argument(
        "--prefix-depth",
        default=-1,
        type=int,
        help="Optional relative path component to prefix identity labels, preventing cross-dataset ID collisions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_manifest(args.data_root, args.track, args.output, args.identity_depth, args.prefix_depth)


if __name__ == "__main__":
    main()
