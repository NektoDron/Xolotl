#!/usr/bin/env bash
# CPU-only correctness build: never initializes or reserves a GPU.
set -euo pipefail
M31_HOST_ROOT=/mnt/ai/Xolotl-m31
M31_HOST_LOG=/mnt/ai/logs/m31-port/cpu-gpu-host-build-$(date -u +%Y%m%dT%H%M%SZ).log
exec >>"$M31_HOST_LOG" 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
cmake -S "$M31_HOST_ROOT/tools/m31/cpu_gpu_host_build" -B "$M31_HOST_ROOT/build-cpu-gpu-host" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-14
cmake --build "$M31_HOST_ROOT/build-cpu-gpu-host" -j24
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
timeout 180 "$M31_HOST_ROOT/build-cpu-gpu-host/cpu_gpu_host_test"
