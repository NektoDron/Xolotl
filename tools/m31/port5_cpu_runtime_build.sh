#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/ai/Xolotl-m31
exec >>/mnt/ai/logs/m31-port/port5-cpu-runtime-build-v1.log 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cmake --build "$ROOT/build-port5-cpu" --target port5_cpu_affinity_check -j24
"$ROOT/build-port5-cpu/port5_cpu_affinity_check"
