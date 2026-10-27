#!/usr/bin/env bash
set -euo pipefail
VISION_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
VISION_LOG=${1:-/mnt/ai/logs/m31-vision/build-native-v1.log}
exec >>"$VISION_LOG" 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
VISION_VENDOR="$VISION_ROOT/build-vision-vendor"
if [ ! -d "$VISION_VENDOR" ]; then
    cp -a --reflink=auto /mnt/ai/Xolotl-m31/build-m31-cpu/_deps/strata_llamacpp-src "$VISION_VENDOR"
fi
VISION_FLAGS=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/gcc-14 -DCMAKE_CXX_COMPILER=/usr/bin/g++-14
 -DSTRATA_BUILD_TESTS=OFF -DSTRATA_DENSE=OFF -DSTRATA_M31=ON -DSTRATA_DSV41_FQ_OP=OFF -DSTRATA_DSV41_HC_MIX=OFF
 -DSTRATA_ENABLE_CUDA=ON -DSTRATA_GGML_CUDA=ON -DSTRATA_M31_CPU_QUANTS_CUDA=ON -DSTRATA_M31_FAST_FFN_CUDA=ON
 -DSTRATA_M31_REGISTERED_ARENA_CUDA=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CUDA_ARCHITECTURES=120a
 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCUDAToolkit_ROOT=/usr/local/cuda -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14
 '-DCMAKE_CUDA_FLAGS=-U_GNU_SOURCE -D_DEFAULT_SOURCE -include /home/nekto/.local/include/cuda13_glibc_compat.h'
 "-DFETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP=$VISION_VENDOR")
cmake -S "$VISION_ROOT" -B "$VISION_ROOT/build-m31-cuda" "${VISION_FLAGS[@]}"
cmake --build "$VISION_ROOT/build-m31-cuda" --target m31_inspect m31_generate m31_serve m31_expert_check -j24
