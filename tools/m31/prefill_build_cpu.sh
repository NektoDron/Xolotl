#!/usr/bin/env bash
set -euo pipefail
M31_PREFILL_ROOT=/mnt/ai/Xolotl-m31
exec >>/mnt/ai/logs/m31-port/prefill-build-cpu-only.log 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
cmake -S "$M31_PREFILL_ROOT/tools/m31/prefill_build" -B "$M31_PREFILL_ROOT/build-streaming-prefill/cpu-only" \
 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 -DM31_PREFILL_CUDA=OFF
cmake --build "$M31_PREFILL_ROOT/build-streaming-prefill/cpu-only" -j24
CUDA_VISIBLE_DEVICES= "$M31_PREFILL_ROOT/build-streaming-prefill/cpu-only/m31_prefill_check" --backend cpu \
 > /mnt/ai/logs/m31-port/prefill-cpu-only-check-v1.json
