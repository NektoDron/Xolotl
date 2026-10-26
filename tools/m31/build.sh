#!/usr/bin/env bash
# Run through sudo systemd-run -p User=nekto, AFTER codex-m31-ref completes.
set -euo pipefail
M31_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
M31_MODE=${1:-cpu}
case "$M31_MODE" in cpu|cuda) ;; *) echo 'usage: build.sh cpu|cuda' >&2; exit 2;; esac
if systemctl is-active -q codex-m31-ref; then
    echo 'codex-m31-ref active: refusing configure/build/tests' >&2
    exit 75
fi
mkdir -p /mnt/ai/logs/m31-port
M31_LOG=/mnt/ai/logs/m31-port/build-${M31_MODE}-$(date -u +%Y%m%dT%H%M%SZ).log
exec >>"$M31_LOG" 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -w 86400 9
check_ref() {
    if systemctl is-active -q codex-m31-ref; then
        echo 'codex-m31-ref active: stopping before next operation' >&2
        exit 75
    fi
}
check_ref
git -C "$M31_ROOT" log -1 --format='source %H %s'
git -C "$M31_ROOT" status --short
M31_BUILD="$M31_ROOT/build-m31-$M31_MODE"
export CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14
M31_FLAGS=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/gcc-14 -DCMAKE_CXX_COMPILER=/usr/bin/g++-14
           -DSTRATA_BUILD_TESTS=OFF -DSTRATA_DENSE=OFF -DSTRATA_M31=ON
           -DSTRATA_DSV41_FQ_OP=OFF -DSTRATA_DSV41_HC_MIX=OFF)
if [ "$M31_MODE" = cuda ]; then
    # Reuse OUR completed CPU fetch, without another clone and without touching up*/PR.
    M31_VENDOR="$M31_ROOT/build-m31-cpu/_deps/strata_llamacpp-src"
    if [ -f "$M31_VENDOR/ggml/CMakeLists.txt" ]; then
        M31_FLAGS+=("-DFETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP=$M31_VENDOR")
    fi
    M31_FLAGS+=(-DSTRATA_ENABLE_CUDA=ON -DSTRATA_GGML_CUDA=ON -DSTRATA_M31_CPU_QUANTS_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a
                -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCUDAToolkit_ROOT=/usr/local/cuda
                -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14
                '-DCMAKE_CUDA_FLAGS=-U_GNU_SOURCE -D_DEFAULT_SOURCE -include /home/nekto/.local/include/cuda13_glibc_compat.h')
else
    M31_FLAGS+=(-DSTRATA_ENABLE_CUDA=OFF -DSTRATA_GGML_CUDA=OFF -DSTRATA_M31_CPU_QUANTS_CUDA=OFF)
fi
# Root CMake fetches its own pinned ggml and applies its patch in OUR build tree.
# Never point its write-capable FetchContent path at up5 or the reference agent's worktree.
cmake -S "$M31_ROOT" -B "$M31_BUILD" "${M31_FLAGS[@]}"
# CMake's compiler-cache migration reconfigures with default options; reapply ours.
if ! grep -q '^STRATA_M31:BOOL=ON$' "$M31_BUILD/CMakeCache.txt"; then
    cmake -S "$M31_ROOT" -B "$M31_BUILD" "${M31_FLAGS[@]}"
fi
check_ref
cmake --build "$M31_BUILD" --target m31_inspect m31_generate m31_serve m31_expert_check -j "${M31_JOBS:-24}"
check_ref
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
python3 "$M31_ROOT/tools/m31/tests/test_compare.py"
python3 "$M31_ROOT/tools/m31/tests/test_math.py" --engine "$M31_BUILD/m31_generate" --inspect "$M31_BUILD/m31_inspect"
python3 "$M31_ROOT/tools/m31/test_server.py"
if [ "$M31_MODE" = cuda ]; then
    check_ref
    cmake -S "$M31_ROOT/tools/m31/oracle" -B "$M31_ROOT/build-m31-oracle" \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 \
        -DCUDAToolkit_ROOT=/usr/local/cuda \
        -DM31_LLAMA_SOURCE=/mnt/ai/llama.cpp-m31pr-1026 \
        -DM31_LLAMA_BUILD=/mnt/ai/llama.cpp-m31pr-1026/build
    check_ref
    cmake --build "$M31_ROOT/build-m31-oracle" -j 4
fi
printf 'Finished %s build and CPU fixture checks: %s\n' "$M31_MODE" "$(date -u --iso-8601=seconds)"
