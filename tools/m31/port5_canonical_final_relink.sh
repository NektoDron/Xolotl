#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/ai/Xolotl-m31
exec >>/mnt/ai/logs/m31-port/port5-canonical-final-relink-v2.log 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cmake --build "$ROOT/build-m31-cpu" --target m31_generate m31_serve m31_expert_check -j24
cmake --build "$ROOT/build-m31-cuda" --target m31_generate m31_serve m31_expert_check -j24
sha256sum "$ROOT/tools/m31/m31_engine.cpp"
