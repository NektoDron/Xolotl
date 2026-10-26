#!/usr/bin/env bash
# One independently schedulable CPU task, releasing the shared resource lock at exit.
set -euo pipefail
M31_VISION_ROOT=/mnt/ai/Xolotl-m31
M31_VISION_SPEC=$(realpath "$1")
M31_VISION_OUT=$2
exec >>"${M31_VISION_OUT}.log" 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=16
/usr/bin/time -v "$M31_VISION_ROOT/build-vision-prepare/m31_vision_adapter" "$M31_VISION_SPEC" "$M31_VISION_OUT"
python3 "$M31_VISION_ROOT/tools/m31/vision_batch.py" assemble "$M31_VISION_OUT/manifest.json" "${M31_VISION_OUT}-batch"
