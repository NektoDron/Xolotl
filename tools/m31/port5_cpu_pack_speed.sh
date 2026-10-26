#!/usr/bin/env bash
set -euo pipefail
exec >>/mnt/ai/logs/m31-port/port5-cpu-pack-speed-v1.log 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -x -w 86400 9
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
unset M31_CPU_AFFINITY M31_CPU_POLL
sha256sum /mnt/ai/logs/m31-port/port5-cpu-pack-bandwidth-v1/port5_cpu_pack_speed
/mnt/ai/logs/m31-port/port5-cpu-pack-bandwidth-v1/port5_cpu_pack_speed
