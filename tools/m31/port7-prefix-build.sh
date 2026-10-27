#!/usr/bin/env bash
# Run only after the coordinator closes the exclusive speed window.
set -euo pipefail
root=/mnt/ai/Xolotl-m31
out=$root/build-port7-prefix
tag=${M31_PREFIX_BUILD_TAG:-v1}
log=/mnt/ai/logs/m31-port/port7-prefix-build-${tag}.log
exec >>"$log" 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
mkdir -p "$out"
inc=$root/build-m31-cpu/_deps/strata_llamacpp-src
common=(-std=c++20 -O2 -DNDEBUG -DGGML_USE_CPU -I"$root/tools/m31" -I"$root/include" -I"$inc/ggml/include" -I"$inc/ggml/src" -I"$inc/vendor/nlohmann")
cp "$root/tools/m31/m31_serve.cpp" "$out/m31_serve.cpp"
if ! rg -q server_prefix_cache.hpp "$out/m31_serve.cpp"; then
    patch -p3 -d "$out" <"$root/tools/m31/port7-prefix-worker.patch"
fi
/usr/bin/g++-14 "${common[@]}" "$root/tools/m31/tests/prefix_cache_check.cpp" -o "$out/prefix_cache_check"
"$out/prefix_cache_check" | tee /mnt/ai/logs/m31-port/port7-prefix-helper-check-${tag}.json
/usr/bin/g++-14 "${common[@]}" -c "$root/tools/m31/m31_engine.cpp" -o "$out/engine.o"
/usr/bin/g++-14 "${common[@]}" -c "$out/m31_serve.cpp" -o "$out/worker.o"
/usr/bin/g++-14 "${common[@]}" -c "$root/tools/m31/m31_gpu_experts.cpp" -o "$out/gpu_experts.o"
/usr/bin/g++-14 "$out/engine.o" "$out/worker.o" "$out/gpu_experts.o" -o "$out/m31_serve" \
    "$root/build-m31-cpu/libm31_engine.a" "$root/build-m31-cpu/ggml/src/libggml.a" \
    "$root/build-m31-cpu/ggml/src/libggml-cpu.a" "$root/build-m31-cpu/ggml/src/libggml-base.a" -ldl -lm -pthread
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= python3 "$root/tools/m31/tests/test_prefix_cache.py" \
    --engine "$out/m31_serve" | tee /mnt/ai/logs/m31-port/port7-prefix-protocol-check-${tag}.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= python3 "$root/tools/m31/test_server.py"
