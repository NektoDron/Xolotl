#!/usr/bin/env bash
set -euo pipefail
M31_Q2_ROOT=/mnt/ai/Xolotl-m31
M31_Q2_OUT=${1:?new output directory required}
mkdir "$M31_Q2_OUT"
exec >"$M31_Q2_OUT/build.log" 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
mkdir "$M31_Q2_OUT/source"
cp "$M31_Q2_ROOT"/tools/m31/{q2_cpu_check.cpp,cpu_batched_verify.hpp,cpu_batched_dot.hpp,cpu_fused_decode.hpp,cpu_direct_decode.hpp,cpu_gpu_host_executor.hpp,cpu_gpu_host_executor.cpp,cpu_threadpool_config.hpp,m31_geometry.hpp} "$M31_Q2_OUT/source/"
M31_Q2_FLAGS=(-std=c++20 -O3 -DNDEBUG -frounding-math -fno-fast-math -I"$M31_Q2_OUT/source" -I"$M31_Q2_ROOT/include" -I"$M31_Q2_ROOT/build-m31-cpu/_deps/strata_llamacpp-src/ggml/include")
/usr/bin/g++-14 "${M31_Q2_FLAGS[@]}" "$M31_Q2_OUT/source/q2_cpu_check.cpp" "$M31_Q2_OUT/source/cpu_gpu_host_executor.cpp" \
 "$M31_Q2_ROOT/build-m31-cpu/ggml/src/libggml.a" "$M31_Q2_ROOT/build-m31-cpu/ggml/src/libggml-cpu.a" "$M31_Q2_ROOT/build-m31-cpu/ggml/src/libggml-base.a" \
 -pthread -lm -ldl -o "$M31_Q2_OUT/q2_cpu_check"
sha256sum "$M31_Q2_OUT/q2_cpu_check" "$M31_Q2_OUT"/source/* "$M31_Q2_ROOT"/build-m31-cpu/ggml/src/libggml{,-cpu,-base}.a >"$M31_Q2_OUT/build.sha256"
