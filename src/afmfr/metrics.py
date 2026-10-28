from __future__ import annotations

import math
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import FaceManifestDataset, ManifestRow


def verification_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    order = np.argsort(scores)[::-1]
    labels_sorted = labels[order].astype(bool)
    positives = labels_sorted.sum()
    negatives = len(labels_sorted) - positives
    if positives == 0 or negatives == 0:
        return {"eer": math.nan, "tar_far_1e-3": math.nan, "tar_far_1e-4": math.nan}

    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(~labels_sorted)
    tpr = tp / positives
    fpr = fp / negatives
    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))

    def tar_at_far(target: float) -> float:
        ok = np.where(fpr <= target)[0]
        if len(ok) == 0:
            return 0.0
        return float(tpr[ok[-1]])

    return {
        "eer": float((fpr[eer_idx] + fnr[eer_idx]) / 2.0),
        "tar_far_1e-3": tar_at_far(1e-3),
        "tar_far_1e-4": tar_at_far(1e-4),
    }


def make_pairs(rows: list[ManifestRow], max_images_per_identity: int, max_identities: int, seed: int):
    rng = random.Random(seed)
    by_identity: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_identity[row.identity].append(idx)

    identities = [identity for identity, idxs in by_identity.items() if len(idxs) >= 2]
    rng.shuffle(identities)
    identities = identities[:max_identities]

    keep_indices = []
    remap = {}
    for identity in identities:
        idxs = by_identity[identity][:]
        rng.shuffle(idxs)
        for idx in idxs[:max_images_per_identity]:
            remap[idx] = len(keep_indices)
            keep_indices.append(idx)

    pairs = []
    labels = []
    for identity in identities:
        idxs = [idx for idx in by_identity[identity] if idx in remap]
        if len(idxs) >= 2:
            pairs.append((remap[idxs[0]], remap[idxs[1]]))
            labels.append(1)

    for i in range(len(identities) - 1):
        a = by_identity[identities[i]][0]
        b = by_identity[identities[i + 1]][0]
        if a in remap and b in remap:
            pairs.append((remap[a], remap[b]))
            labels.append(0)

    subset_rows = [rows[idx] for idx in keep_indices]
    return subset_rows, np.asarray(pairs, dtype=np.int64), np.asarray(labels, dtype=np.int64)


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    dataset: FaceManifestDataset,
    rows: list[ManifestRow],
    device: torch.device,
    batch_size: int,
    max_images_per_identity: int,
    max_identities: int,
    seed: int,
) -> dict[str, float]:
    subset_rows, pairs, labels = make_pairs(rows, max_images_per_identity, max_identities, seed)
    if len(subset_rows) < 2 or len(pairs) == 0:
        return {"eer": math.nan, "tar_far_1e-3": math.nan, "tar_far_1e-4": math.nan}

    subset = FaceManifestDataset(subset_rows, dataset.transform)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)
    embeddings = []
    model.eval()
    for images, _, _, _ in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        emb = F.normalize(model(images).float(), dim=1)
        embeddings.append(emb.cpu())
    embeddings_tensor = torch.cat(embeddings, dim=0)
    a = embeddings_tensor[pairs[:, 0]]
    b = embeddings_tensor[pairs[:, 1]]
    scores = (a * b).sum(dim=1).numpy()
    return verification_metrics(scores, labels)
