# Results and Artifact Provenance

## Recognition

Team ArogyaPandit (Gourav Gupta and Anant Gupta) received the **Third Place Award in Track 2: Limited Data Track** at IJCB 2026. The [organizer-issued certificate](../assets/third-place-certificate.pdf) is included as evidence. The award applies to Track 2 only.

The competition report is [arXiv:2607.24422](https://arxiv.org/abs/2607.24422), accepted at IEEE IJCB 2026. It includes the team, method description, and evaluation tables. The README transcribes our official verification and identification scores. Rankings use Borda count across benchmark metrics, not LFW accuracy alone.

## What Was Actually Submitted

- **Track 2:** adapted CLIP ViT-L/14, Q/V LoRA rank 4 and alpha 8, ArcFace, and CLIP preservation. Selected checkpoint: epoch 4, step 8,905 (zero-based epoch). Local LFW accuracy was 99.3167%; the official table reports 99.32%.
- **Track 1 original candidate:** Q/V LoRA rank 16 and alpha 32, ArcFace, no preservation. Selected checkpoint: epoch 2, step 34,026. It produced 50.00% local LFW accuracy. The organizers independently flagged the failure.
- **Track 1 replacement:** original, unadapted OpenAI CLIP ViT-L/14 visual encoder. Local LFW accuracy was 95.5167%; the organizers' full-data table reports 95.50%. This is a fallback result, not a successful adapted-model result.

The paper describes the originally proposed full-data LoRA method. Read that alongside this explicit record of the replacement artifact. Its trainable-parameter table also lists rank 8 for our limited-data row, whereas the approach table and recovered configuration specify rank 4. For reproducing this implementation use the configuration and source rather than that conflicting entry.

## Evaluation Caveats

The original official `test.py` loaded both ONNX files and generated `(12000,768)` LFW embeddings. Metric computation then raised `ValueError: Expect x to not have duplicates` in SciPy interpolation of repeated FAR values.

Local LFW accuracies were subsequently calculated using the official wrapper, loader, transforms, horizontal-flip feature sum, normalization, and 10-fold `calculate_roc` utility, bypassing the failing TAR/FAR interpolation stage. Do not describe this as an unmodified `test.py` run completing successfully.

CPU and CUDA PyTorch-to-ONNX comparisons passed on sampled synthetic validation images. These conversion checks did not detect the Track 1 recognition failure. A clean export alone is insufficient evidence of a useful face-recognition model.

The exact cause of Track 1 degradation was not established. Sparse synthetic validation, constant learning rates, and the absence of preservation are limitations or hypotheses, not proven causal explanations.

## Artifact Availability

This repository contains recovered source and configs. Historical checkpoint and ONNX ZIP links on Google Cloud Storage returned HTTP 404 when checked during repository preparation. This release does not offer downloadable trained weights. Datasets remain subject to organizer access and redistribution terms.

No new GPU training or hidden-benchmark evaluation was performed for this release. Official results and historical local checks are distinct from the lightweight source checks run by GitHub Actions.
