#!/usr/bin/env bash
set -euo pipefail
M31_THP_ROOT=/mnt/ai/Xolotl-m31
M31_THP_TAG=${1:-v1}
[[ "$M31_THP_TAG" =~ ^[A-Za-z0-9_-]+$ ]]
exec >>/mnt/ai/logs/m31-port/prefill-thp-build.log 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
cmake -S "$M31_THP_ROOT/tools/m31/prefill_thp_build" -B "$M31_THP_ROOT/build-streaming-prefill/thp" \
 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-14
cmake --build "$M31_THP_ROOT/build-streaming-prefill/thp" -j24
M31_THP_FIXTURE=/mnt/ai/logs/m31-port/prefill-thp-tiny-${M31_THP_TAG}.gguf
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
 python3 "$M31_THP_ROOT/tools/m31/tests/make_fixture.py" "$M31_THP_FIXTURE"
CUDA_VISIBLE_DEVICES= "$M31_THP_ROOT/build-streaming-prefill/thp/m31_thp_check" \
 --model "$M31_THP_FIXTURE" --out "/mnt/ai/logs/m31-port/prefill-thp-small-${M31_THP_TAG}.json"
