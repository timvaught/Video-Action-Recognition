# HMDB51 repository and notebook audit

## Executive verdict

The major model-selection leakage has been fixed in the uploaded repository: the training loop uses only train and validation loaders, checkpoints are selected from validation accuracy/loss, and test evaluation is a separate explicit mode.

The repository was **not fully fixed**, however. The most consequential remaining bug was that the configuration advertised gradient accumulation while the training loop zeroed gradients and stepped the optimizer every mini-batch. With `BATCH_SIZE=2`, the actual effective batch was 2 rather than 4. Additional integrity and reproducibility gaps could silently invalidate an experiment or evaluation.

The uploaded notebook contains no executed cells, outputs, history JSON, or training log, so the exact cause of the reported 79% plateau cannot be determined from curves. The recommendations below are based on code behavior and the supplied configuration.

## Findings in the uploaded version

| Severity | Finding | Evidence in uploaded code | Consequence |
|---|---|---|---|
| Critical | Gradient accumulation was not implemented | `train.py:40-61` clears gradients and performs an optimizer/scaler step on every mini-batch; `run.py` has no accumulation argument | `GRADIENT_ACCUMULATION_STEPS=2` in an external config has no effect; effective batch remains 2 |
| Fixed major issue | Test metrics are no longer used for checkpointing or early stopping | `train.py:111-126` runs only train and validation; test evaluation is under explicit `--mode eval` | The original test-selection leakage is removed, provided the final test command is not repeatedly used during tuning |
| High | Reloaded split files were trusted without overlap checks | `run.py:179-189` only deserializes paths and labels | A malformed or stale `splits.json` could place the same video in train, validation, and test |
| High | Official splits were not checked for duplicate content across boundaries | `video_datasets.py:243-315` creates the official train/validation/test sets and returns them directly | Duplicate videos or extraction copies could cross official split boundaries unnoticed |
| Medium | Duplicate detection is exact and partial | `video_datasets.py:145-154` hashes bytes from seven representative frames | It catches many exact copies but misses arbitrary re-encodes, resizes, crops, and other near-duplicates; it can also over-group videos whose sampled frames match while unsampled frames differ |
| High | Transformer normalization/bias parameters received global AdamW decay | `run.py:420-440` creates only backbone/head groups and applies global `weight_decay` | LayerNorm scale/bias and all biases are decayed; this is usually undesirable for transformer fine-tuning |
| High | The backbone learning rate is extremely small | Supplied config gives `5e-4 × 0.02 = 1e-5`, with unfreezing at epoch 5 | The pretrained MViT backbone may barely adapt to HMDB51, leaving the small new head to do most of the work |
| High | Temporal sampling may be mismatched to the pretrained weights | Supplied `16 × stride 2` spans 31 extracted-frame intervals | If adjacent extracted frames represent 30 fps, the clip covers about 1.03 seconds and samples at 15 fps rather than the pretrained evaluation rate near 7.5 fps |
| Medium | Checkpoints did not bind evaluation to the training input/split configuration | `run.py:456-466` stores configuration, but `run.py:498-505` checks architecture only | A checkpoint could be evaluated with a changed stride, frame count, image size, or split file without an error |
| Medium | Reproducibility claim contradicted implementation | `utils.py:140-146` says deterministic cuDNN behavior but enables `cudnn.benchmark=True` | Repeated runs can select different convolution algorithms and differ despite the same seed |
| Medium | Classification reports could fail when a class is absent | `run.py:516-529` and `test.py:8-12` pass 51 names without explicit labels | Small diagnostics or incomplete splits can raise a target-name/label-count error |
| Medium | Notebook and command configuration had drifted | Uploaded notebook still defaults to R(2+1)D-18, 112 px, batch 4, LR `1e-3`, and has no accumulation variable | The notebook did not represent the MViT configuration quoted in the question, so runs were difficult to reproduce or compare |
| Medium | Shell wrappers were stale | `train.sh` and `test.sh` used 24 frames and old checkpoint paths | Running the wrappers launched a different experiment or failed to find the checkpoint |
| Low | Best weights were copied on the active device | `train.py:92` and `145` deep-copy the state dictionary | For MViT this retains roughly another model-sized set of tensors on GPU and increases memory pressure |
| Low | No regression tests were included | No `tests/` directory in the uploaded repository | The leakage and accumulation bugs could return unnoticed |

## Correct parts of the uploaded version

- Torchvision video input is correctly permuted from `[B,T,C,H,W]` to `[B,C,T,H,W]`.
- MViT evaluation preprocessing matches the pretrained weights: 256 resize, 224 center crop, mean `0.45`, and standard deviation `0.225`.
- Spatial crop, horizontal flip, and color jitter are consistent across all frames in a clip.
- Frame files are naturally sorted and support `.jpg`, `.jpeg`, and `.png`.
- Multi-clip logits are averaged correctly.
- Short videos are sampled across their duration instead of being padded with long repeats of the last frame.
- Random splitting groups detected exact duplicates before stratification.
- Early stopping and checkpoint selection use validation metrics only.

## Patched copy

The audit-fixed copy implements and tests:

1. True FP32/AMP gradient accumulation, including a final partial accumulation group.
2. Path-overlap checks plus two-stage exact-content checks: representative frames identify candidates and all ordered extracted frames confirm exact copies.
3. Conflicting-label duplicate rejection.
4. A split-file SHA-256 stored in each selected checkpoint.
5. Strict evaluation checks for architecture, class count, frame count, image size, temporal stride, and split hash.
6. Four AdamW groups: backbone/head × decay/no-decay, with zero decay for biases and one-dimensional tensors.
7. Deterministic cuDNN settings and deterministic-algorithm warnings.
8. CPU-resident best-weight copies.
9. Explicit labels in classification reports.
10. Coherent MViT defaults in the Colab notebook and shell scripts.
11. Fifteen regression tests covering the critical fixes, official duplicate policies, all-frame confirmation, and official filename ambiguity handling.

The patch does not claim to detect every perceptual near-duplicate, guarantee 85%, or provide full optimizer/scheduler resume checkpoints.

## Recommended first experiment

Use this as the controlled starting point:

```python
MODEL_TYPE = "mvit_v2_s"
FRAMES_PER_VIDEO = 16
IMAGE_SIZE = 224

# Set from the actual extracted-frame rate.
EXTRACTED_FPS = 30.0
TEMPORAL_STRIDE = round(EXTRACTED_FPS / 7.5)  # 4 when EXTRACTED_FPS is 30

BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4  # effective batch 8

N_EPOCHS = 60
LEARNING_RATE = 3e-4
BACKBONE_LR_MULTIPLIER = 0.10    # backbone LR 3e-5
UNFREEZE_EPOCH = 2

DROPOUT = 0.30
LABEL_SMOOTHING = 0.05
WEIGHT_DECAY = 0.02              # no decay on bias/1-D parameters

VAL_CLIPS = 5
TEST_CLIPS = 10
EARLY_STOPPING_PATIENCE = 15
MINIMUM_EPOCHS = 20
```

`TEMPORAL_STRIDE=4` is only appropriate when adjacent extracted frames represent about 30 fps. If frames were extracted at 15 fps, `stride=2` already gives about 7.5 fps. If the extraction rate is unknown, determine it from the extraction command or source metadata before interpreting a stride experiment.

## Minimal ablation sequence

Keep the split file and seed fixed while tuning. Do not run the held-out test command between experiments.

| Run | Change from recommended baseline | Question answered |
|---|---|---|
| A | Baseline above | Establish the corrected optimizer/sampling result |
| B | Only change temporal stride from 4 to 2 | Does shorter, denser motion sampling fit this extraction better? |
| C | Only change head LR to `5e-4` and multiplier to `0.05` | Is the baseline under-adapting or over-updating the backbone? |
| D | Return to best LR and change weight decay from `0.02` to `0.05` | Does stronger transformer regularization improve validation? |
| E | Return to best optimizer and strengthen clip-consistent crop/color augmentation | Is the remaining gap caused by overfitting? |

Do not combine all changes at once. The effects are not additive, and a single split can be noisy.

## How to read the curves

- **Train accuracy remains below about 90%:** likely underfitting or optimization limitation. Unfreeze earlier, raise backbone LR modestly, or reduce dropout/label smoothing.
- **Train accuracy exceeds about 95% while validation remains around 79%:** likely overfitting or split noise. Increase augmentation/weight decay, consider repeated training views, and inspect class-level errors.
- **Train and validation oscillate strongly:** verify accumulation is active, lower head LR, and inspect gradient norms/AMP scale skips.
- **Validation improves but test does not:** stop tuning on test; verify exact split identity and near-duplicates, then use official splits and multiple seeds.
- **A few classes dominate the errors:** inspect the confusion matrix and video durations for those classes before changing global hyperparameters.

## Protocol for a defensible final result

1. Use an official HMDB51 split and carve validation only from its official training subset.
2. Select temporal stride, augmentation, optimizer settings, and a fixed epoch count using validation only.
3. Optionally retrain that selected configuration on the complete official training subset for the fixed selected epoch count.
4. Evaluate the official test subset once.
5. Repeat on official splits 1, 2, and 3 and report the mean, not the best split.

## Why 85% may require a stronger pretraining path

A corrected MViT-V2-S fine-tune may improve materially over 79%, but 85% is not guaranteed. If the corrected sampling/optimization ablations saturate below the target, the next high-leverage change is stronger video-domain self-supervised pretraining rather than increasingly aggressive test-time tuning. Larger VideoMAE-family models with domain-specific pretraining have reported HMDB51 results above 85%, but those results use substantially larger models and pretraining resources and are not directly comparable to a torchvision MViT-V2-S fine-tune.

## v5 official-split duplicate handling

The v2 strict audit correctly stopped when it found byte-identical extracted-frame
sequences assigned to opposite sides of an official HMDB51 split. v5 makes the
required policy explicit instead of forcing the user to disable the audit:

- `drop_train` (default) preserves the complete official test set and removes all
  exact training copies before validation is carved from the remaining training
  subset. The resulting protocol is leakage-safe but must be described as
  **official-derived**, not the untouched historical benchmark.
- `allow` preserves the original official assignments and reports the overlap.
- `error` retains the original strict-failure behavior.

Candidate duplicate groups are still found cheaply from representative frames, but
a group is now confirmed by hashing every extracted frame before any sample is
removed. The complete policy report is saved beside the split metadata.
