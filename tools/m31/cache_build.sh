#!/usr/bin/env bash
set -euo pipefail
M31_CACHE_ROOT=/mnt/ai/Xolotl-m31
exec >>/mnt/ai/logs/m31-port/cache-ranking-build.log 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
cmake -S "$M31_CACHE_ROOT/tools/m31/cache_build" -B "$M31_CACHE_ROOT/build-cache-ranking" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 -DCUDAToolkit_ROOT=/usr/local/cuda
cmake --build "$M31_CACHE_ROOT/build-cache-ranking" -j24
