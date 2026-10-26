#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/ai/Xolotl-m31
OUT=/mnt/ai/logs/m31-port/port5-cpu-pack-bandwidth-v1
exec >>/mnt/ai/logs/m31-port/port5-cpu-pack-speed-build-v1.log 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
mkdir "$OUT"
cp "$ROOT/tools/m31/port5_cpu_pack_speed.cpp" "$ROOT/tools/m31/cpu_pack_spans.hpp" \
   "$ROOT/tools/m31/cpu_gpu_host_executor.hpp" "$ROOT/tools/m31/cpu_gpu_host_executor.cpp" "$OUT/"
/usr/bin/g++-14 -std=c++20 -O3 -DNDEBUG -frounding-math -fno-fast-math -I"$OUT" \
    "$OUT/port5_cpu_pack_speed.cpp" "$OUT/cpu_gpu_host_executor.cpp" -pthread -lm -o "$OUT/port5_cpu_pack_speed"
sha256sum "$OUT"/*
