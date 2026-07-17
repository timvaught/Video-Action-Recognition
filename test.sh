#!/usr/bin/env bash
set -euo pipefail

: "${FRAME_DIR:?Set FRAME_DIR to the extracted HMDB51 frame root}"
VIDEO_MAE_VARIANT="${VIDEO_MAE_VARIANT:-small}"
case "${VIDEO_MAE_VARIANT}" in
  small) ARCHITECTURE="videomaev2_vit_s_distilled" ;;
  base) ARCHITECTURE="videomaev2_vit_b_distilled" ;;
  *) echo "VIDEO_MAE_VARIANT must be 'small' or 'base'" >&2; exit 2 ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-results_videomaev2_v7_${VIDEO_MAE_VARIANT}_seed42}"

python run.py \
  --frame_dir "${FRAME_DIR}" \
  --mode eval \
  --output_dir "${OUTPUT_DIR}" \
  --split_file "${SPLIT_FILE:-${OUTPUT_DIR}/splits.json}" \
  --ckpt "${CKPT:-${OUTPUT_DIR}/best_model_wts.pt}" \
  --split_protocol "${SPLIT_PROTOCOL:-official}" \
  --official_split_number "${OFFICIAL_SPLIT_NUMBER:-1}" \
  --architecture "${ARCHITECTURE}" \
  --n_classes 51 \
  --fr_per_vid 16 \
  --image_size 224 \
  --temporal_stride "${TEMPORAL_STRIDE:-4}" \
  --val_clips "${VAL_CLIPS:-10}" \
  --test_clips "${TEST_CLIPS:-10}" \
  --flip_tta true \
  --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
  --eval_clip_chunk_size "${EVAL_CLIP_CHUNK_SIZE:-1}" \
  --workers "${WORKERS:-2}" \
  --dropout "${DROPOUT:-0.35}" \
  --label_smoothing "${LABEL_SMOOTHING:-0.10}" \
  --fingerprint_split_audit true \
  --amp true \
  --wandb_mode disabled
