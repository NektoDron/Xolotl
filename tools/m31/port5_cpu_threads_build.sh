#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/ai/Xolotl-m31
OUT=/mnt/ai/logs/m31-port/port5-cpu-threads-v1
exec >>/mnt/ai/logs/m31-port/port5-cpu-threads-build-v1.log 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
mkdir "$OUT"
cp "$ROOT/tools/m31/port5_cpu_threads_probe.cpp" "$ROOT/tools/m31/cpu_fused_decode.hpp" \
   "$ROOT/tools/m31/cpu_direct_decode.hpp" "$ROOT/tools/m31/cpu_gpu_host_executor.hpp" \
   "$ROOT/tools/m31/cpu_threadpool_config.hpp" "$OUT/"
/usr/bin/g++-14 -std=c++20 -O3 -DNDEBUG -frounding-math -fno-fast-math \
    -I"$OUT" -I"$ROOT/tools/m31" -I"$ROOT/include" -I"$ROOT/build-m31-cpu/_deps/strata_llamacpp-src/ggml/include" \
    "$OUT/port5_cpu_threads_probe.cpp" "$ROOT/build-m31-cpu/ggml/src/libggml.a" \
    "$ROOT/build-m31-cpu/ggml/src/libggml-cpu.a" "$ROOT/build-m31-cpu/ggml/src/libggml-base.a" \
    -pthread -lm -ldl -o "$OUT/port5_cpu_threads_probe"
sha256sum "$OUT"/*
