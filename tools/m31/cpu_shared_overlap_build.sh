#!/usr/bin/env bash
set -euo pipefail
M31_SHARED_ROOT=/mnt/ai/Xolotl-m31
M31_SHARED_LOG=/mnt/ai/logs/m31-port/cpu-shared-overlap-build-$(date -u +%Y%m%dT%H%M%SZ).log
exec >>"$M31_SHARED_LOG" 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
cmake -S "$M31_SHARED_ROOT/tools/m31/cpu_shared_overlap_build" -B "$M31_SHARED_ROOT/build-cpu-shared-overlap" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 -DCUDAToolkit_ROOT=/usr/local/cuda -DM31_SHARED_OVERLAP_NVTX=ON -DM31_SHARED_OVERLAP_TESTING=ON
cmake --build "$M31_SHARED_ROOT/build-cpu-shared-overlap" -j24
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
python3 "$M31_SHARED_ROOT/tools/m31/cpu_shared_overlap_test.py" --check "$M31_SHARED_ROOT/build-cpu-shared-overlap/cpu_shared_overlap_check"
python3 "$M31_SHARED_ROOT/tools/m31/tests/test_math.py" --engine "$M31_SHARED_ROOT/build-cpu-shared-overlap/m31_generate" --inspect "$M31_SHARED_ROOT/build-cpu-shared-overlap/m31_inspect"
