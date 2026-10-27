#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/ai/Xolotl-m31
OUT=${1:-$ROOT/build-port7-cpu-actual}
mkdir -p "$OUT"
exec >>"$OUT/build.log" 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
cp "$ROOT/tools/m31/port7_cpu_actual_replay.cpp" "$ROOT/tools/m31/cpu_batched_verify.hpp" "$ROOT/tools/m31/cpu_batched_dot.hpp" "$ROOT/tools/m31/cpu_fused_decode.hpp" "$ROOT/tools/m31/cpu_direct_decode.hpp" "$ROOT/tools/m31/cpu_gpu_host_executor.hpp" "$ROOT/tools/m31/cpu_threadpool_config.hpp" "$OUT/"
/usr/bin/g++-14 -std=c++20 -O3 -DNDEBUG -frounding-math -fno-fast-math \
 -I"$OUT" -I"$ROOT/tools/m31" -I"$ROOT/include" -I"$ROOT/build-m31-cpu/_deps/strata_llamacpp-src/ggml/include" \
 "$OUT/port7_cpu_actual_replay.cpp" "$ROOT/build-m31-cpu/ggml/src/libggml.a" \
 "$ROOT/build-m31-cpu/ggml/src/libggml-cpu.a" "$ROOT/build-m31-cpu/ggml/src/libggml-base.a" \
 -pthread -lm -ldl -o "$OUT/port7_cpu_actual_replay"
sha256sum "$OUT"/port7_cpu_actual_replay "$OUT"/*.hpp "$OUT"/port7_cpu_actual_replay.cpp >"$OUT/build.sha256"
