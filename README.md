# HMDB51 Action Recognition with VideoMAE V2 Distilled

This repository is the current version of an HMDB51 action-recognition project that began with a frame-based **LRCN** model: a 2D ResNet feature extractor followed by an LSTM. The primary experiment now fine-tunes **VideoMAE V2 distilled ViT-S or ViT-B** checkpoints on extracted HMDB51 frames.

The revision focuses on three goals:

1. use a stronger video-native pretrained backbone;
2. prevent train/validation/test contamination; and
3. make experiments reproducible, comparable, and practical on a single CUDA GPU or Google Colab.

The recommended starting point is **VideoMAE V2 ViT-S distilled**. ViT-B is included as a higher-capacity follow-up when GPU memory and runtime permit.

---

## Contents

- [HMDB51 dataset](#hmdb51-dataset)
- [Expected frame layout](#expected-frame-layout)
- [What changed from the original project](#what-changed-from-the-original-project)
- [Leakage and duplicate handling](#leakage-and-duplicate-handling)
- [Model choices and experiment history](#model-choices-and-experiment-history)
- [Recommended VideoMAE V2 configuration](#recommended-videomae-v2-configuration)
- [Quick start in Google Colab](#quick-start-in-google-colab)
- [Running from the command line](#running-from-the-command-line)
- [Training and evaluation workflow](#training-and-evaluation-workflow)
- [Practical run tips](#practical-run-tips)
- [Output artifacts](#output-artifacts)
- [Tests and project structure](#tests-and-project-structure)
- [Limitations and reporting](#limitations-and-reporting)
- [References](#references)

---

## HMDB51 dataset

HMDB51 is a human-action recognition benchmark containing **6,766 short video clips across 51 action classes**. Examples include `brush_hair`, `cartwheel`, `fencing`, `run`, `sit`, `throw`, and `wave`. The clips were collected from movies and online video sources and contain substantial variation in camera motion, viewpoint, video quality, background, and execution style.

The benchmark provides **three official train/test splits**. For each split and each class:

- 70 clips are assigned to the official training portion;
- 30 clips are assigned to the official test portion; and
- the remaining clips are marked as unused for that split.

This project reserves a validation subset **only from the official training portion**. Model selection, learning-rate scheduling, and early stopping use validation data. The official test portion is evaluated separately after the configuration has been selected.

For a paper-style HMDB51 result, run the final locked configuration on official splits 1, 2, and 3 and report the mean rather than choosing the best split.

### Dataset source

Obtain HMDB51 from an authorized source and follow its distribution terms. The original dataset and paper are linked in [References](#references).

---

## Expected frame layout

The training code consumes **directories of extracted image frames**, not raw AVI files directly.

```text
HMDB51/
├── brush_hair/
│   ├── video_001/
│   │   ├── frame_000001.jpg
│   │   ├── frame_000002.jpg
│   │   └── ...
│   └── video_002/
├── cartwheel/
├── catch/
└── ... 51 class directories total
```

Supported frame extensions are `.jpg`, `.jpeg`, and `.png`. Frame names should contain sortable numeric indices. The loader naturally sorts them before temporal sampling.

The official metadata directory contains files such as:

```text
testTrainMulti_7030_splits/
├── brush_hair_test_split1.txt
├── cartwheel_test_split1.txt
├── ...
├── brush_hair_test_split2.txt
└── ... 153 files for all three splits
```

The Colab notebook can download, checksum-verify, extract, and validate the small official split archive when it is missing.

### Raw-video preprocessing note

If only the original AVI files are available, extract one frame sequence per source video before training. Record the extraction rate because `EXTRACTED_FPS` determines the correct temporal stride. A single-video example is:

```bash
mkdir -p frames/example_video
ffmpeg -i example_video.avi -vf fps=30 frames/example_video/frame_%06d.jpg
```

Apply the same extraction rate to the complete dataset. Do not flatten frames from multiple source videos into one directory.

---

## What changed from the original project

| Area | Original project | Current repository |
|---|---|---|
| Primary model | LRCN: ResNet frame features plus an LSTM | VideoMAE V2 distilled ViT-S or ViT-B |
| Video representation | Individual 2D frame features aggregated recurrently | Joint spatiotemporal transformer tokens learned from video pretraining |
| Frame selection | First available frames, with limited temporal coverage control | Naturally sorted frames, fixed 16-frame clips, configurable temporal stride, uniform handling of short videos |
| Dataset protocol | Random stratified train/validation/test split | Official HMDB51 split files, with validation carved only from official training data |
| Duplicate protection | No content-level cross-split audit | Candidate fingerprints followed by full ordered-frame hashing |
| Augmentation | Frame transforms could vary independently across time | Crop, flip, color jitter, and random erasing are shared across every frame in a clip |
| Optimization | One Adam learning rate for the full model | AdamW groups, zero decay for bias/normalization parameters, backbone/head rates, layer-wise LR decay, warmup, and cosine decay |
| Effective batch | Direct mini-batch updates | Correct FP32/AMP gradient accumulation, including the last partial group |
| Evaluation | One sampled sequence per video | Deterministic multi-clip averaging, optional horizontal-flip TTA, and memory-safe clip chunking |
| Checkpoints | Model weights only | Best validation checkpoint plus architecture, input, and split metadata |
| Reproducibility | Basic split file and limited seed control | Separate split/training seeds, relative paths, split SHA-256 binding, run configuration, and regression tests |
| Colab workflow | Manual path and dependency setup | Dataset/repository extraction, official-split setup, pretrained checkpoint verification, smoke test, training, and evaluation controls |

The original LRCN training loop already chose its best weights using validation accuracy. The revised system strengthens that separation by making test evaluation an explicit mode, defaulting the notebook to `RUN_EVALUATION = False` during tuning, and refusing evaluation when the checkpoint, split, architecture, or input configuration does not match.

---

## Leakage and duplicate handling

The most important data issue found during the audit was **duplicate extracted video content crossing an official train/test boundary**. The folders could have different names while containing the same ordered frame sequence. Training on one copy and evaluating on the other would expose test content during training and inflate the result.

The current audit uses two stages:

1. representative frames identify possible duplicate groups efficiently;
2. every ordered extracted frame is hashed before a pair is confirmed as byte-identical.

The official duplicate policies are:

- `drop_train` — recommended for leakage-safe experiments. Preserve the official test item and remove its exact training copy before validation is created;
- `error` — abort when a confirmed train/test duplicate is found; or
- `allow` — keep the untouched historical assignments and report the known overlap.

The default is:

```text
--official_duplicate_policy drop_train
```

Using `drop_train` produces an **official-derived, decontaminated split**. It is not exactly the untouched historical 70-video-per-class training assignment, so describe it accurately in reports. The test assignment remains unchanged.

Audit details are written to:

```text
official_duplicate_audit.json
splits.json
```

The audit confirms byte-identical extracted frame sequences. It cannot guarantee detection of every re-encoded, resized, cropped, temporally shifted, or otherwise perceptually similar duplicate.

---

## Model choices and experiment history

### Current primary model: VideoMAE V2 ViT-S distilled

Use this first. It is the most practical VideoMAE V2 configuration for a typical 16 GB Colab GPU and supports a full 16-frame, 224 x 224 fine-tuning path.

```text
--architecture videomaev2_vit_s_distilled
```

### Higher-capacity option: VideoMAE V2 ViT-B distilled

ViT-B is considerably larger. It uses batch size 1 and more accumulation steps so the effective batch remains comparable with ViT-S.

```text
--architecture videomaev2_vit_b_distilled
```

### Historical MViT-V2-S experiment

The audited MViT-V2-S experiment reached:

```text
Best validation accuracy: 86.01%
Test loss:               1.2522
Test accuracy:           79.48%
```

Training accuracy approached 100%, indicating a generalization gap rather than insufficient model capacity. That result motivated stronger regularization, clip-consistent augmentation, and the VideoMAE V2 transfer experiment.

The split-1 test result has already been inspected. Do not keep tuning against that test score. Use validation for all remaining choices and prefer official split 2 or 3 as an untouched confirmation split.

### Other supported backbones

The code also supports:

```text
mvit_v2_s
r2plus1d_18
lrcn
```

Use them as controlled ablations, not as automatically interchangeable runs. Reuse the same `splits.json`, effective batch, temporal sampling, augmentation profile, and evaluation budget when comparing architectures.

---

## Recommended VideoMAE V2 configuration

### ViT-S default

| Setting | Value |
|---|---:|
| Frames | 16 |
| Resolution | 224 x 224 |
| Temporal stride | 4 when extracted frames represent about 30 fps |
| Mini-batch | 2 |
| Accumulation steps | 4 |
| Effective batch | 8 |
| Head learning rate | `3e-4` |
| Top backbone learning rate | `3e-5` |
| Layer-wise LR decay | `0.90` |
| Weight decay | `0.05` |
| Label smoothing | `0.10` |
| Dropout | `0.35` |
| Stochastic depth | `0.10` |
| Warmup | 5 epochs |
| Maximum epochs | 45 |
| Validation/test clips | 10 / 10 |
| Evaluation clip chunk | 1 |

### ViT-B switch

Change only the initial capacity settings for the first comparison:

```python
VIDEOMAE_VARIANT = "base"
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
DROP_PATH_RATE = 0.15
```

Keep the same effective batch, split, learning-rate scale, temporal sampling, augmentation, and validation protocol.

### Temporal stride

The default notebook computes:

```python
TEMPORAL_STRIDE = max(1, round(EXTRACTED_FPS / 7.5))
```

Examples:

| Rate represented by adjacent frame files | Suggested stride |
|---:|---:|
| 30 fps | 4 |
| 15 fps | 2 |
| 7.5 fps | 1 |

Set `EXTRACTED_FPS` from the actual extraction process. Do not assume the original video FPS if frames were subsampled during extraction.

---

## Quick start in Google Colab

The companion notebook is:

```text
HMDB51_Colab_VideoMAEv2_Distilled_v7.ipynb
```

### 1. Place the archives in Google Drive

The default notebook expects:

```text
/content/drive/MyDrive/Homework_4/
├── HMDB51.zip
└── hmdb51_repo_videomaev2_distilled_v7.zip
```

`HMDB51.zip` must contain the extracted frame hierarchy described above.

### 2. Select a GPU runtime

In Colab, choose a CUDA GPU runtime before executing the notebook. ViT-S is the recommended starting point.

### 3. Review the main configuration cell

For the first full experiment, keep:

```python
VIDEOMAE_VARIANT = "small"
RUN_SMOKE_TEST = True
RUN_FULL_TRAINING = True
RUN_MULTI_SEED_VALIDATION = False
RUN_TEMPORAL_STRIDE_ABLATION = False
RUN_EVALUATION = False
```

Update:

```python
EXTRACTED_FPS = 30.0  # replace with the real extracted-frame rate
```

The notebook will:

- mount Drive;
- validate configuration and write access;
- obtain and validate official split metadata when necessary;
- extract the repository and frame dataset into local `/content` storage;
- install dependencies;
- download and checksum-verify the pinned VideoMAE V2 checkpoint;
- run a structural checkpoint preflight;
- execute a one-epoch smoke test; and
- start full training.

### 4. Evaluate only after locking the configuration

After validation-based model selection, change the workflow flags to:

```python
RUN_SMOKE_TEST = False
RUN_FULL_TRAINING = False
RUN_MULTI_SEED_VALIDATION = False
RUN_TEMPORAL_STRIDE_ABLATION = False
RUN_EVALUATION = True
```

Then run the final evaluation cell. It loads `best_model_wts.pt` and the matching `splits.json`.

---

## Running from the command line

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

A CUDA-enabled PyTorch installation is strongly recommended.

### 2. Verify the pretrained checkpoint

Run the CPU preflight before a long job:

```bash
python preflight_videomaev2.py \
  --architecture videomaev2_vit_s_distilled \
  --cache_dir /path/to/persistent/videomae_cache \
  --verify_sha256 true
```

Training should not proceed unless the checkpoint reports at least 97% matched backbone-parameter coverage.

### 3. Train with the shell wrapper

For an official-split run:

```bash
export FRAME_DIR=/path/to/HMDB51
export OFFICIAL_SPLIT_DIR=/path/to/testTrainMulti_7030_splits
export OFFICIAL_SPLIT_NUMBER=1
export OFFICIAL_DUPLICATE_POLICY=drop_train
export OUTPUT_DIR=results_videomaev2_v7_small_seed42
export PRETRAINED_CACHE_DIR=/path/to/persistent/videomae_cache
export VIDEO_MAE_VARIANT=small
export WANDB_MODE=online

bash train.sh
```

Important: if `OFFICIAL_SPLIT_DIR` is omitted, `train.sh` falls back to a random split. Set it for official HMDB51 experiments.

To compare against an existing MViT run on identical data:

```bash
export SPLIT_FILE=/path/to/mvit_results/splits.json
bash train.sh
```

The wrapper copies that split into the new output directory and audits it before training.

### 4. Evaluate the selected checkpoint

```bash
export FRAME_DIR=/path/to/HMDB51
export OUTPUT_DIR=results_videomaev2_v7_small_seed42
export VIDEO_MAE_VARIANT=small

bash test.sh
```

By default, `test.sh` uses:

```text
${OUTPUT_DIR}/best_model_wts.pt
${OUTPUT_DIR}/splits.json
```

---

## Training and evaluation workflow

Use this order for defensible experiments:

1. run the smoke test;
2. train VideoMAE V2 ViT-S with seed 42;
3. inspect validation curves only;
4. repeat promising settings with seeds 123 and 2026 on the identical split;
5. test ViT-B only if ViT-S transfers well enough to justify the extra compute;
6. lock the architecture and hyperparameters;
7. evaluate once on an untouched official split; and
8. eventually report the mean over official splits 1, 2, and 3.

The notebook can enable multi-seed runs with:

```python
RUN_MULTI_SEED_VALIDATION = True
MULTI_SEED_VALUES = (42, 123, 2026)
```

It verifies that every seed used the same split hash.

---

## Practical run tips

### Avoid accidental test tuning

Keep `RUN_EVALUATION = False` while selecting architecture, augmentation, learning rate, temporal stride, or regularization. Test accuracy should not decide the next experiment.

### Use one output directory per experiment

Do not overwrite a previous run. A new directory keeps checkpoints, split files, configs, and logs unambiguous.

### Keep the dataset on local Colab storage during training

The notebook extracts `HMDB51.zip` from Drive to `/content/hmdb51_data`. Reading thousands of individual frame files directly from Google Drive can starve the GPU.

### Recover after an interruption

After any completed validation epoch, the best checkpoint so far is saved to Drive as:

```text
best_model_wts.pt
```

If `best_model_wts.pt` and `splits.json` exist, the test evaluator can use the selected checkpoint even if the training cell was interrupted. However, the repository does **not** save the complete optimizer, scheduler, scaler, and random-number-generator state required for exact mid-run continuation.

### Handle GPU out-of-memory errors

Use these changes in order:

1. keep `VIDEOMAE_VARIANT = "small"`;
2. set batch size to 1 and accumulation to 8 to retain effective batch 8;
3. keep gradient checkpointing enabled;
4. keep `EVAL_BATCH_SIZE = 1` and `EVAL_CLIP_CHUNK_SIZE = 1`; and
5. reduce validation clips only for a smoke test, not for the final model comparison.

### Compare architectures fairly

Hold constant:

```text
splits.json
split seed
frame count and temporal stride
effective batch
augmentation
evaluation clips and flip TTA
checkpoint-selection rule
```

Otherwise, an apparent architecture improvement may come from a different data or evaluation budget.

### Check temporal coverage

For 16 frames and stride 4, the sampled clip spans 61 source-frame positions. At 30 fps that is approximately 2.0 seconds. Confirm that this is appropriate for the frame extraction rate and the actions being classified.

---

## Output artifacts

A normal run writes the following files to its output directory:

```text
best_model_wts.pt             validation-selected model checkpoint
splits.json                   exact train/validation/test membership
training_history.json         per-epoch loss and accuracy
run_config.json               full run and augmentation configuration
official_duplicate_audit.json duplicate policy and any removed train copies
classification_report.txt     readable per-class test report
classification_report.json    machine-readable per-class test report
test_metrics.json             final test loss and accuracy
```

The checkpoint is bound to the split hash and important input settings. Evaluation rejects incompatible architecture, class-count, frame-count, image-size, temporal-stride, or split-file combinations.

---

## Tests and project structure

### Regression tests

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

The tests cover split overlap, exact-duplicate confirmation and decontamination, gradient accumulation, optimizer grouping, checkpoint/config binding, VideoMAE V2 tensor shapes and checkpoint matching, layer-wise learning-rate decay, clip-consistent augmentation, and chunked multi-clip inference.

### Main files

```text
run.py                             command-line entry point
train.py                           training loop and checkpoint selection
test.py                            evaluation and reports
video_datasets.py                  frame loading, official splits, duplicate audit
utils.py                           transforms, loaders, schedules, reproducibility
models.py                          LRCN, R(2+1)D, MViT, and VideoMAE wrappers
videomaev2.py                      local VideoMAE V2 encoder implementation
preflight_videomaev2.py            pretrained checkpoint verification
train.sh / test.sh                 command-line wrappers
HMDB51_Colab_VideoMAEv2_Distilled_v7.ipynb
```

Additional audit and experiment notes are retained in the repository for traceability.

---

## Limitations and reporting

- VideoMAE V2 is a stronger transfer candidate, not a guarantee of 85% test accuracy.
- Exact byte-level duplicate detection does not prove that no perceptual near-duplicates exist.
- `drop_train` is leakage-safe but creates an official-derived protocol rather than the untouched historical training assignment.
- The repository saves the best model for evaluation but not a complete exact-resume training state.
- A single best epoch on one small validation split can be noisy; compare multiple seeds.
- Do not compare results that use different numbers of clips, crops, TTA views, or test-set feedback.
- Since official split 1 test performance has already been examined in the MViT work, use split 2 or 3 for a cleaner final confirmation.

---

## References

- Kuehne, H., Jhuang, H., Garrote, E., Poggio, T., and Serre, T. **HMDB: A Large Video Database for Human Motion Recognition.** ICCV 2011. [Paper record](https://is.mpg.de/en/publications/kuhne-iccv-2011) and [DOI](https://doi.org/10.1109/ICCV.2011.6126543).
- [HMDB51 dataset resource page](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/).
- Wang, L. et al. **VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking.** CVPR 2023. [Paper](https://arxiv.org/abs/2303.16727) and [official repository](https://github.com/OpenGVLab/VideoMAEv2).
