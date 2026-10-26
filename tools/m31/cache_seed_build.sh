#!/usr/bin/env bash
set -euo pipefail
M31_SEED_ROOT=/mnt/ai/Xolotl-m31
exec >>/mnt/ai/logs/m31-port/cache-seed-build-tests.log 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
cmake -S "$M31_SEED_ROOT/tools/m31/cache_seed_build" -B "$M31_SEED_ROOT/build-cache-ranking/seed-reader" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-14
cmake --build "$M31_SEED_ROOT/build-cache-ranking/seed-reader" -j24
python3 "$M31_SEED_ROOT/tools/m31/cache_seed_test.py" \
    --reader "$M31_SEED_ROOT/build-cache-ranking/seed-reader/m31_cache_seed_reader"
