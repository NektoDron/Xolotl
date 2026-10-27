#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/ai/Xolotl-m31
OUT=${1:-$ROOT/build-port7-cpu-pair}
mkdir -p "$OUT"
exec >>"$OUT/build.log" 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
cp "$ROOT/tools/m31/port7_cpu_pair_check.cpp" "$ROOT/tools/m31/cpu_batched_dot.hpp" "$OUT/"
/usr/bin/g++-14 -std=c++20 -O3 -DNDEBUG -frounding-math -fno-fast-math \
 -I"$OUT" -I"$ROOT/tools/m31" -I"$ROOT/include" -I"$ROOT/build-m31-cpu/_deps/strata_llamacpp-src/ggml/include" \
 "$OUT/port7_cpu_pair_check.cpp" "$ROOT/build-m31-cpu/ggml/src/libggml.a" "$ROOT/build-m31-cpu/ggml/src/libggml-cpu.a" "$ROOT/build-m31-cpu/ggml/src/libggml-base.a" \
 -pthread -lm -ldl -o "$OUT/port7_cpu_pair_check"
sha256sum "$OUT"/port7_cpu_pair_check "$OUT"/*.hpp "$OUT"/*.cpp >"$OUT/build.sha256"
