#!/usr/bin/env bash
set -euo pipefail
M31_VISION_ROOT=/mnt/ai/Xolotl-m31
M31_VISION_JOB=$1
exec >>"$M31_VISION_JOB/preprocess.log" 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8
for M31_VISION_CASE in image video; do
  "$M31_VISION_ROOT/build-vision-prepare/m31_vision_adapter" "$M31_VISION_JOB/$M31_VISION_CASE.json" "$M31_VISION_JOB/$M31_VISION_CASE-ipc"
  python3 "$M31_VISION_ROOT/tools/m31/vision_batch.py" assemble "$M31_VISION_JOB/$M31_VISION_CASE-ipc/manifest.json" "$M31_VISION_JOB/$M31_VISION_CASE-batch"
done
