#!/usr/bin/env bash
set -euo pipefail
M31_CPU_ROOT=/mnt/ai/Xolotl-m31
M31_CPU_LOG=/mnt/ai/logs/m31-port/cpu-decode-build-$(date -u +%Y%m%dT%H%M%SZ).log
exec >>"$M31_CPU_LOG" 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
cmake -S "$M31_CPU_ROOT/tools/m31/cpu_build" -B "$M31_CPU_ROOT/build-cpu-decode" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 -DCUDAToolkit_ROOT=/usr/local/cuda
cmake --build "$M31_CPU_ROOT/build-cpu-decode" -j24
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
python3 "$M31_CPU_ROOT/tools/m31/tests/test_math.py" --engine "$M31_CPU_ROOT/build-cpu-decode/m31_generate" --inspect "$M31_CPU_ROOT/build-cpu-decode/m31_inspect"
python3 "$M31_CPU_ROOT/tools/m31/cpu_test_decode.py" --engine "$M31_CPU_ROOT/build-cpu-decode/m31_generate"
