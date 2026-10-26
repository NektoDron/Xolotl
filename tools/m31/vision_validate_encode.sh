#!/usr/bin/env bash
# Run after the first image-batch timing has shown that CPU cost is acceptable.
set -euo pipefail
M31_VISION_ROOT=/mnt/ai/Xolotl-m31
M31_VISION_JOB=$(realpath "$1")
exec >>"$M31_VISION_JOB/validate.log" 2>&1
for M31_VISION_CASE in image-single video-batch video-single; do
  /bin/bash "$M31_VISION_ROOT/tools/m31/vision_encode.sh" "$M31_VISION_JOB/$M31_VISION_CASE.json" "$M31_VISION_JOB/$M31_VISION_CASE"
done
for M31_VISION_LABEL in image video; do
  python3 "$M31_VISION_ROOT/tools/m31/vision_batch.py" compare \
    "$M31_VISION_JOB/$M31_VISION_LABEL-batch/manifest.json" \
    "$M31_VISION_JOB/$M31_VISION_LABEL-single/manifest.json" --atol 0 --rtol 0 \
    >"$M31_VISION_JOB/$M31_VISION_LABEL-compare.json"
done
