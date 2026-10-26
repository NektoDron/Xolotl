#!/usr/bin/env bash
set -euo pipefail
M31_PREFILL_ROOT=/mnt/ai/Xolotl-m31
exec >>/mnt/ai/logs/m31-port/prefill-build.log 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
cmake -S "$M31_PREFILL_ROOT/tools/m31/prefill_build" -B "$M31_PREFILL_ROOT/build-streaming-prefill" \
 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 -DCUDAToolkit_ROOT=/usr/local/cuda
cmake --build "$M31_PREFILL_ROOT/build-streaming-prefill" -j24
CUDA_VISIBLE_DEVICES= "$M31_PREFILL_ROOT/build-streaming-prefill/m31_prefill_check" --backend cpu \
 --model /mnt/ai/models/MiniMax-M3.1-Flash-GGUF/UD-IQ3_S/MiniMax-M3.1-Flash-UD-IQ3_S-00001-of-00005.gguf \
 > /mnt/ai/logs/m31-port/prefill-cpu-check-v1.json
