#!/usr/bin/env bash
# Prepared only. Coordinator grants a slot after V9 build.
set -euo pipefail
ROOT=/mnt/ai/Xolotl-m31
OUT=$ROOT/build-port7-cpu-skew/v2
mkdir -p /mnt/ai/logs/m31-port/port7-cpu-skew-build-v2
exec >/mnt/ai/logs/m31-port/port7-cpu-skew-build-v2/build.log 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
mkdir -p "$OUT"
cp "$ROOT"/build-port7-cpu-skew-proposal/v1/*.hpp "$ROOT"/build-port7-cpu-skew-proposal/v1/*.cpp "$ROOT"/build-port7-cpu-skew-proposal/v1/proposal.json "$OUT/"
/usr/bin/g++-14 -std=c++20 -O3 -DNDEBUG -frounding-math -fno-fast-math -I"$OUT" -I"$ROOT/include" -I"$ROOT/build-m31-cpu/_deps/strata_llamacpp-src/ggml/include" "$OUT/port7_cpu_actual_replay.cpp" "$ROOT/build-m31-cpu/ggml/src/libggml.a" "$ROOT/build-m31-cpu/ggml/src/libggml-cpu.a" "$ROOT/build-m31-cpu/ggml/src/libggml-base.a" -pthread -lm -ldl -o "$OUT/port7_cpu_skew_replay"
sha256sum "$OUT"/port7_cpu_skew_replay "$OUT"/*.hpp "$OUT"/*.cpp >"$OUT/build.sha256"
