#!/usr/bin/env bash
set -euo pipefail
exec >>/mnt/ai/logs/m31-port/port5-cpu-fused-speed-v1.log 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -x -w 86400 9
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
unset M31_CPU_AFFINITY M31_CPU_POLL
sha256sum /mnt/ai/logs/m31-port/port5-cpu-fused-v1/port5_cpu_probe
/mnt/ai/logs/m31-port/port5-cpu-fused-v1/port5_cpu_probe /mnt/ai/models/MiniMax-M3.1-Flash-GGUF/UD-IQ3_S/MiniMax-M3.1-Flash-UD-IQ3_S-00001-of-00005.gguf 64 5
