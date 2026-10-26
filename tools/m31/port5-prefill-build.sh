#!/usr/bin/env bash
set -euo pipefail
exec >>/mnt/ai/logs/m31-port/port5-prefill-build-v1.log 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
cmake -S /mnt/ai/Xolotl-m31/tools/m31/port5-prefill-build -B /mnt/ai/Xolotl-m31/build-port5-prefill -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 -DCUDAToolkit_ROOT=/usr/local/cuda
cmake --build /mnt/ai/Xolotl-m31/build-port5-prefill -j24
CUDA_VISIBLE_DEVICES= /mnt/ai/Xolotl-m31/build-port5-prefill/m31_prefill_check --backend cpu > /mnt/ai/logs/m31-port/port5-prefill-cpu-v1.json
