# VideoMAE V2 HMDB51 experiment plan (v7)

## Goal

Compare a stronger video-transformer pretraining path against the corrected MViT-V2-S system without changing split membership or consulting the test set during tuning.

## Backbone choices

- **ViT-S distilled** is the default reliability run. It is appropriate for common Colab T4-class memory limits.
- **ViT-B distilled** is the higher-capacity follow-up. It uses batch 1 and accumulation 8 to preserve an effective batch of 8.

Both checkpoints are Kinetics-710 distilled releases from OpenGVLab. The repository pins their source revision and validates exact SHA-256 digests before loading.

## Controlled comparison rules

1. Reuse the exact v6 `splits.json` when available.
2. Keep 16 frames, 224x224, temporal stride 4, effective batch 8, and the v6 augmentation profile.
3. Select checkpoints using validation accuracy, with validation loss as the tie-breaker.
4. Compare at least two or three training seeds before declaring one architecture better.
5. Do not run test evaluation after every seed or hyperparameter change.

## Recommended order

1. Run the one-epoch ViT-S smoke test.
2. Run ViT-S seed 42 to completion.
3. If the run is stable, repeat ViT-S with seeds 123 and 2026 on the identical split.
4. Run ViT-B with the same split and seed 42.
5. Continue ViT-B multi-seed validation only when its validation improvement justifies the extra runtime.
6. Lock the winning configuration and evaluate on an untouched official split.

## Diagnostic interpretation

- Training reaches nearly 100% again while validation does not improve: the remaining bottleneck is generalization, not capacity.
- Training and validation both remain low: inspect checkpoint preflight, learning rates, and temporal sampling.
- Validation improves but is highly variable across seeds: report the mean and standard deviation rather than the single best epoch.
- GPU OOM during validation: keep `eval_batch_size=1` and `eval_clip_chunk_size=1`; do not lower the final number of validation clips solely to hide memory pressure.

## Expected artifacts

- `pretrained_preflight.json`
- `best_model_wts.pt`
- `splits.json`
- `training_history.json`
- `run_config.json`
- `official_duplicate_audit.json`
- complete training log

The preflight and run metadata preserve checkpoint provenance, structural match information, model variant, temporal settings, augmentation settings, and split identity.
