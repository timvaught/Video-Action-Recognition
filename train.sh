#!/usr/bin/env bash
set -euo pipefail

: "${FRAME_DIR:?Set FRAME_DIR to the extracted HMDB51 frame root}"

VIDEO_MAE_VARIANT="${VIDEO_MAE_VARIANT:-small}"
case "${VIDEO_MAE_VARIANT}" in
  small)
    ARCHITECTURE="videomaev2_vit_s_distilled"
    DEFAULT_BATCH_SIZE=2
    DEFAULT_ACCUMULATION=4
    DEFAULT_DROP_PATH=0.10
    ;;
  base)
    ARCHITECTURE="videomaev2_vit_b_distilled"
    DEFAULT_BATCH_SIZE=1
    DEFAULT_ACCUMULATION=8
    DEFAULT_DROP_PATH=0.15
    ;;
  *)
    echo "VIDEO_MAE_VARIANT must be 'small' or 'base'" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-results_videomaev2_v7_${VIDEO_MAE_VARIANT}_seed42}"
PRETRAINED_CACHE_DIR="${PRETRAINED_CACHE_DIR:-${OUTPUT_DIR}/pretrained_cache}"
OFFICIAL_SPLIT_DIR="${OFFICIAL_SPLIT_DIR:-}"

split_args=(--split_protocol random)
if [[ -n "${OFFICIAL_SPLIT_DIR}" ]]; then
  split_args=(
    --split_protocol official
    --official_split_dir "${OFFICIAL_SPLIT_DIR}"
    --official_split_number "${OFFICIAL_SPLIT_NUMBER:-1}"
    --official_duplicate_policy "${OFFICIAL_DUPLICATE_POLICY:-drop_train}"
    --validation_size 0.15
  )
fi

if [[ -n "${SPLIT_FILE:-}" ]]; then
  mkdir -p "${OUTPUT_DIR}"
  cp "${SPLIT_FILE}" "${OUTPUT_DIR}/splits.json"
  split_args+=(
    --split_file "${OUTPUT_DIR}/splits.json"
    --reuse_existing_split true
  )
fi

python run.py \
  --frame_dir "${FRAME_DIR}" \
  --mode train \
  --output_dir "${OUTPUT_DIR}" \
  "${split_args[@]}" \
  --architecture "${ARCHITECTURE}" \
  --n_classes 51 \
  --fr_per_vid 16 \
  --image_size 224 \
  --temporal_stride "${TEMPORAL_STRIDE:-4}" \
  --batch_size "${BATCH_SIZE:-${DEFAULT_BATCH_SIZE}}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-${DEFAULT_ACCUMULATION}}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
  --eval_clip_chunk_size "${EVAL_CLIP_CHUNK_SIZE:-1}" \
  --workers "${WORKERS:-2}" \
  --val_clips "${VAL_CLIPS:-10}" \
  --test_clips "${TEST_CLIPS:-10}" \
  --flip_tta true \
  --pretrained true \
  --pretrained_cache_dir "${PRETRAINED_CACHE_DIR}" \
  --verify_pretrained_sha256 true \
  --gradient_checkpointing true \
  --drop_path_rate "${DROP_PATH_RATE:-${DEFAULT_DROP_PATH}}" \
  --layer_decay "${LAYER_DECAY:-0.90}" \
  --head_init_scale 0.001 \
  --dropout "${DROPOUT:-0.35}" \
  --train_crop_scale_min 0.60 \
  --train_crop_scale_max 1.00 \
  --train_crop_ratio_min 0.75 \
  --train_crop_ratio_max 1.333 \
  --color_jitter_brightness 0.20 \
  --color_jitter_contrast 0.20 \
  --color_jitter_saturation 0.20 \
  --color_jitter_hue 0.05 \
  --train_horizontal_flip_probability 0.50 \
  --random_erasing_probability 0.10 \
  --learning_rate "${LEARNING_RATE:-3e-4}" \
  --backbone_lr_multiplier "${BACKBONE_LR_MULTIPLIER:-0.10}" \
  --weight_decay "${WEIGHT_DECAY:-0.05}" \
  --label_smoothing "${LABEL_SMOOTHING:-0.10}" \
  --unfreeze_epoch "${UNFREEZE_EPOCH:-2}" \
  --warmup_epochs "${WARMUP_EPOCHS:-5}" \
  --minimum_lr_factor 0.01 \
  --n_epochs "${N_EPOCHS:-45}" \
  --minimum_epochs "${MINIMUM_EPOCHS:-15}" \
  --early_stopping_patience "${EARLY_STOPPING_PATIENCE:-12}" \
  --split_seed "${SPLIT_SEED:-42}" \
  --seed "${SEED:-42}" \
  --freeze_batch_norm false \
  --amp true \
  --wandb_project "${WANDB_PROJECT:-hmdb51-videomaev2-v7}" \
  --wandb_mode "${WANDB_MODE:-online}"
