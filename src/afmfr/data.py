from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class ManifestRow:
    image_path: str
    identity: str
    track: str
    split: str
    label: int


def read_manifest(path: str | Path, split: str | None = None) -> list[ManifestRow]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))

    if split is not None:
        raw_rows = [row for row in raw_rows if row["split"] == split]

    identities = sorted({row["identity"] for row in raw_rows})
    identity_to_label = {identity: idx for idx, identity in enumerate(identities)}

    rows = []
    for row in raw_rows:
        identity = row["identity"]
        rows.append(
            ManifestRow(
                image_path=row["image_path"],
                identity=identity,
                track=row.get("track", ""),
                split=row["split"],
                label=identity_to_label[identity],
            )
        )
    return rows


def build_train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.70, 1.0),
                ratio=(0.85, 1.15),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(0.20, 0.20, 0.15, 0.05)],
                p=0.4,
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                image_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


class FaceManifestDataset(Dataset):
    def __init__(self, rows: list[ManifestRow], transform: transforms.Compose) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str, str]:
        row = self.rows[index]
        image = Image.open(row.image_path).convert("RGB")
        return self.transform(image), row.label, row.identity, row.image_path


class IdentityBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        rows: list[ManifestRow],
        identities_per_batch: int,
        images_per_identity: int,
        seed: int,
        max_batches: int = 0,
    ) -> None:
        self.rows = rows
        self.identities_per_batch = identities_per_batch
        self.images_per_identity = images_per_identity
        self.seed = seed
        self.max_batches = max_batches
        self.by_identity: dict[str, list[int]] = defaultdict(list)
        for idx, row in enumerate(rows):
            self.by_identity[row.identity].append(idx)
        self.identities = sorted(self.by_identity)
        if len(self.identities) < identities_per_batch:
            raise ValueError("Not enough identities for requested batch sampler")

    def __iter__(self) -> Iterable[list[int]]:
        rng = random.Random(self.seed)
        for _ in range(len(self)):
            identities = rng.sample(self.identities, self.identities_per_batch)
            batch: list[int] = []
            for identity in identities:
                idxs = self.by_identity[identity]
                if len(idxs) >= self.images_per_identity:
                    batch.extend(rng.sample(idxs, self.images_per_identity))
                else:
                    batch.extend(rng.choices(idxs, k=self.images_per_identity))
            yield batch

    def __len__(self) -> int:
        if self.max_batches:
            return self.max_batches
        batch_size = self.identities_per_batch * self.images_per_identity
        return max(1, len(self.rows) // batch_size)
