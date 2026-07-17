# Audit fixes, VideoMAE V2 additions, and remaining limits

## Leakage and split integrity

- Training uses validation only for checkpoint selection and early stopping.
- Test evaluation is available only through explicit evaluation mode.
- Random and official splits are checked for duplicate paths. Representative frames identify candidates, and every extracted frame is hashed before exact-content overlap is confirmed.
- Reloaded split files receive the same integrity audit.
- Checkpoints store a SHA-256 digest of the split file used for model selection.
- Evaluation rejects a changed split file or mismatched architecture/input configuration.
- Split paths are stored relative to the frame root for portability.

The exact audit is byte-based. It removes representative-frame false positives by confirming all frames, but it still cannot prove the absence of re-encoded, resized, cropped, temporally shifted, or perceptually similar duplicates.

## Data pipeline

- Frame files are naturally sorted.
- `.jpg`, `.jpeg`, and `.png` are accepted consistently.
- Crop, flip, color-jitter, and random-erasing parameters are shared by every frame in a clip.
- Short videos use uniform sampling over their duration instead of repeated tail padding.
- Temporal statistics are displayed for the training split only.
- VideoMAE V2 uses the official fine-tuning normalization and bicubic preprocessing path.

## Model and optimization

- Torchvision and VideoMAE V2 video backbones receive `[B,C,T,H,W]` tensors.
- VideoMAE V2 ViT-S and ViT-B distilled checkpoints are supported through a self-contained local implementation.
- Source checkpoints are pinned by repository revision, expected file size, and SHA-256 before deserialization.
- Checkpoint loading requires at least 97% backbone-parameter coverage and reports incompatible keys.
- Gradient checkpointing reduces VideoMAE V2 activation memory.
- PyTorch scaled-dot-product attention enables memory-efficient GPU kernels where supported.
- Multi-clip evaluation can process clips sequentially instead of flattening all ten views into one accelerator batch.
- Gradient accumulation is implemented for both FP32 and AMP, including final partial groups.
- Transformer fine-tuning uses layer-wise learning-rate decay.
- Biases, one-dimensional normalization parameters, and attention-scale parameters receive zero AdamW decay.
- A shared warmup/cosine multiplier preserves all LR group ratios.
- Best weights are kept on CPU to reduce GPU memory pressure.
- Classification reports remain valid when a class is absent from a diagnostic subset.

## Reproducibility and maintainability

- cuDNN benchmarking is disabled for repeatable runs.
- Python, NumPy, PyTorch, DataLoader workers, and DataLoader generators are seeded.
- A separate split seed allows training-seed comparisons without changing validation membership.
- Regression tests cover the leakage/correctness fixes, VideoMAE V2 checkpoint matching, pinned checkpoint metadata, layer-wise LR decay, preprocessing, clip-consistent random erasing, and multi-clip chunking.

## Still not guaranteed

- VideoMAE V2 is a stronger transfer candidate, not a guaranteed 85% test result.
- The official distilled checkpoint still needs to download successfully in the training environment.
- Full optimizer/scheduler/scaler resume checkpoints are not implemented.
- A benchmark result should use all three official HMDB51 splits and report their mean.
