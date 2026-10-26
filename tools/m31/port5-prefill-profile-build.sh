#!/usr/bin/env bash
set -euo pipefail
exec >>/mnt/ai/logs/m31-port/port5-prefill-build-v3.log 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
cmake --build /mnt/ai/Xolotl-m31/build-port5-prefill -j24
CUDA_VISIBLE_DEVICES= /mnt/ai/Xolotl-m31/build-port5-prefill/m31_prefill_check --backend cpu > /mnt/ai/logs/m31-port/port5-prefill-cpu-v3.json
M31_GPU_EXPERT_PROFILE=1 CUDA_VISIBLE_DEVICES= /mnt/ai/Xolotl-m31/build-port5-prefill/m31_prefill_check --backend cpu > /mnt/ai/logs/m31-port/port5-prefill-cpu-profile-v3.json
