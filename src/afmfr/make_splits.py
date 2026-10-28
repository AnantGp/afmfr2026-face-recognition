from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_path", "identity", "track", "split"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_manifest(
    manifest: Path,
    output: Path,
    seed: int,
    identity_val_fraction: float,
    same_id_val_images: int,
    min_images_for_same_id_val: int,
) -> None:
    rows = read_rows(manifest)
    by_identity: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_identity[row["identity"]].append(idx)

    rng = random.Random(seed)
    identities = sorted(by_identity)
    rng.shuffle(identities)

    n_val_identity = max(1, int(round(len(identities) * identity_val_fraction)))
    val_identity_set = set(identities[:n_val_identity])

    for row in rows:
        row["split"] = "train"

    for identity in val_identity_set:
        for idx in by_identity[identity]:
            rows[idx]["split"] = "val_identity"

    for identity in identities[n_val_identity:]:
        idxs = by_identity[identity]
        if len(idxs) < min_images_for_same_id_val:
            continue
        shuffled = idxs[:]
        rng.shuffle(shuffled)
        for idx in shuffled[:same_id_val_images]:
            rows[idx]["split"] = "val_same"

    write_rows(output, rows)

    counts = defaultdict(int)
    identity_counts: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        counts[row["split"]] += 1
        identity_counts[row["split"]].add(row["identity"])
    for split in ["train", "val_same", "val_identity"]:
        print(f"{split}: {counts[split]} images, {len(identity_counts[split])} identities")
    print(f"Wrote {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", default=20260502, type=int)
    parser.add_argument("--identity-val-fraction", default=0.05, type=float)
    parser.add_argument("--same-id-val-images", default=2, type=int)
    parser.add_argument("--min-images-for-same-id-val", default=5, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_manifest(
        manifest=args.manifest,
        output=args.output,
        seed=args.seed,
        identity_val_fraction=args.identity_val_fraction,
        same_id_val_images=args.same_id_val_images,
        min_images_for_same_id_val=args.min_images_for_same_id_val,
    )


if __name__ == "__main__":
    main()
