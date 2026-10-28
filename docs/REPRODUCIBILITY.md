# Reproducing the Competition Experiments

## Release Status

This release recovers the original source and configurations used during the competition. Checkpoints and data are not included. A new end-to-end GPU training run has not been performed for this GitHub release, and bitwise reproduction is not claimed. Historical metrics are reported in [RESULTS.md](RESULTS.md).

Recorded environment: Python 3.9, PyTorch 2.8.0+cu128, ONNX Runtime 1.19.2, one NVIDIA A100-SXM4-40GB, and OpenAI CLIP commit `ded190a052fdf4585bd685cee5bc96e0310d2c93`. The dependency file is not a complete frozen environment. Install a matching torchvision build with PyTorch, then install requirements and the editable package. CPU users should substitute `onnxruntime` for `onnxruntime-gpu`.

## Data Access and Layout

Obtain access from the [organizers](https://github.com/TaharChettaoui/IJCB-AFMFR-2026). Credentials and archive passwords are intentionally excluded. The official Track 1 specification provides approximately 3 million images from 35,000 identities; Track 2 provides 250,000 images from 5,000 identities. The historical extracted Track 1 manifest actually contained 3,097,400 images across 35,974 folder-derived identities; inspect your archive version rather than silently enforcing nominal counts.

Use this layout (one dataset-directory level, then identity folders):

```text
data/raw/track1/dataset_a/identity/image.png
data/raw/track1/dataset_b/identity/image.png
data/raw/track2/dataset/identity/image.png
```

If authorized GCP access is still available, `bash scripts/download_data.sh "$PWD"` downloads official archives. Set `AFMFR_ZIP_PASSWORD` in your shell from the organizer-provided value and run `bash scripts/extract_data.sh "$PWD"`. Never commit that value. Helpers assume the original archive names; inspect them against your download first.

Build Track 1 labels with a dataset prefix to distinguish equal identity folder names in separate datasets:

```bash
python -m afmfr.build_manifest --data-root data/raw/track1 \
  --track track1 --output data/manifests/track1_all.csv \
  --identity-depth 1 --prefix-depth 0
python -m afmfr.make_splits --manifest data/manifests/track1_all.csv \
  --output data/manifests/track1_splits.csv --seed 20260502 \
  --identity-val-fraction 0.04 --same-id-val-images 2 \
  --min-images-for-same-id-val 6
```

For Track 2:

```bash
python -m afmfr.build_manifest --data-root data/raw/track2 \
  --track track2 --output data/manifests/track2_all.csv --identity-depth 1
python -m afmfr.make_splits --manifest data/manifests/track2_all.csv \
  --output data/manifests/track2_splits.csv --seed 20260502 \
  --identity-val-fraction 0.05 --same-id-val-images 2
```

If your root directly contains identity folders, use `--identity-depth 0` instead. Manifests store absolute paths; regenerate them after moving data. Verify labels and counts before training.

Historical split counts:

| Track | Train images / identities | Same-ID validation images / identities | Unseen-ID validation images / identities |
| --- | --- | --- | --- |
| Track 1 | 2,903,680 / 34,535 | 69,070 / 34,535 | 124,650 / 1,439 |
| Track 2 | 228,000 / 4,750 | 9,500 / 4,750 | 12,500 / 250 |

## Training

```bash
python -m afmfr.train --config configs/track2_baseline.yaml
# Historical full-data experiment, which did not yield the accepted fallback:
python -m afmfr.train --config configs/track1_rank1_a100.yaml
```

The first model load downloads original CLIP weights. Training uses AdamW, mixed precision, a fixed base seed, identity-balanced microbatches, gradient accumulation, and backbone gradient clipping. It saves `last.pt` and the lowest sampled unseen-ID EER checkpoint as `best_identity.pt`. The source has no resume option or learning-rate scheduler.

Track 2's selected checkpoint was zero-based epoch 4, optimizer step 8,905. Track 1's selected adapted checkpoint was zero-based epoch 2, step 34,026, and was later replaced by unadapted CLIP for submission.

Preprocessing: random resized crop to 224 (scale 0.70-1.0, ratio 0.85-1.15), flip p=0.5, color jitter p=0.4, grayscale p=0.05, Gaussian blur p=0.15, and erasing p=0.15. CLIP normalization uses mean `(0.48145466, 0.4578275, 0.40821073)` and std `(0.26862954, 0.26130258, 0.27577711)`. Validation resizes and center-crops. No external alignment or detection model is used.

## Export and Numerical Validation

```bash
python -m afmfr.export_onnx --config configs/track2_baseline.yaml \
  --checkpoint runs/track2_baseline/best_identity.pt --output submissions/track2.onnx
python -m afmfr.inspect_onnx submissions/track2.onnx
python -m afmfr.validate_onnx --config configs/track2_baseline.yaml \
  --checkpoint runs/track2_baseline/best_identity.pt --onnx submissions/track2.onnx \
  --split val_identity --samples 64 --batch-size 8 --provider cuda
python -m afmfr.validate_onnx --config configs/track2_baseline.yaml \
  --checkpoint runs/track2_baseline/best_identity.pt --onnx submissions/track2.onnx \
  --split val_identity --samples 16 --batch-size 4 --provider cpu
```

The export is opset 13, float32 input `input` with shape `(N,3,224,224)`, and output `output` with shape `(N,768)`. Input normalization is external to ONNX. Normalize features before cosine scoring. Graph name inspection is a heuristic, not a formal architecture proof; also check checkpoint keys and the original CLIP architecture.

## Historical Implementation Caveats

The core training code is preserved to make its behavior auditable:

- `lora_dropout` appears in configs, but the wrapper's dropout module is not applied in its forward pass. Effective LoRA dropout is zero.
- `warmup_steps` is recorded but unused. Learning rates remain constant.
- The identity sampler resets its RNG to the same seed each epoch and repeats index sampling; augmentations can still differ.
- Incomplete gradient-accumulation groups at epoch end are discarded.
- The validation pair sampler retains few negative pairs because its membership check only accepts some sampled indices. EER and low-FAR estimates can be unreliable; epoch selection also changes the validation sample seed.
- A fixed base seed is not a full determinism guarantee. CUDA deterministic algorithms and a complete dependency lock were not recorded.
- Numerically equivalent ONNX and PyTorch outputs do not establish recognition performance.

Future corrected experiments should use new configs, checkpoints, and separately labeled results. This release does not silently alter the historic training recipe.

## Lightweight Checks

```bash
python -m compileall -q src
python -m unittest discover -s tests -v
```

These cover manifest identity separation and split invariants without downloading CLIP or requiring a GPU. They do not validate convergence, ONNX inference, or competition scores.
