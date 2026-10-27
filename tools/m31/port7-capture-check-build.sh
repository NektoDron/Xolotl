#!/usr/bin/env bash
set -euo pipefail
root=/mnt/ai/Xolotl-m31
out=$root/build-port7-prefix/capture-v1
exec >/mnt/ai/logs/m31-port/port7-capture-check-v1.log 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
mkdir -p "$out"
inc=$root/build-m31-cpu/_deps/strata_llamacpp-src/ggml
/usr/bin/g++-14 -std=c++20 -O2 -I"$root/tools/m31" -I"$inc/include" \
  "$root/tools/m31/tests/capture_host_check.cpp" -o "$out/capture_host_check" \
  "$root/build-m31-cpu/ggml/src/libggml-cpu.a" "$root/build-m31-cpu/ggml/src/libggml-base.a" -ldl -lm -pthread
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "$out/capture_host_check" | tee /mnt/ai/logs/m31-port/port7-capture-check-v1.json
