#!/bin/bash
set -euo pipefail
M31_ROOT=/mnt/ai/Xolotl-m31
M31_BUILD=$M31_ROOT/build-gpu-correctness
exec >>/mnt/ai/logs/m31-port/gpu-build-engine-quants-${1:-v1}.log 2>&1
exec 8>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 8
cd "$M31_ROOT"
M31_GGML=$M31_ROOT/build-m31-cpu/_deps/strata_llamacpp-src/ggml
mkdir -p "$M31_BUILD/source"
cp tools/m31/*.hpp tools/m31/*.cpp tools/m31/gpu_cpu_quants.cu "$M31_BUILD/source/"
M31_INC=(-I"$M31_ROOT/include" -I"$M31_BUILD/source" -I"$M31_GGML/include" -I"$M31_GGML/src")
/usr/local/cuda/bin/nvcc -std=c++20 -O3 -arch=sm_120a -ccbin=/usr/bin/g++-14 -U_GNU_SOURCE -D_DEFAULT_SOURCE -include /home/nekto/.local/include/cuda13_glibc_compat.h "${M31_INC[@]}" -c "$M31_BUILD/source/gpu_cpu_quants.cu" -o "$M31_BUILD/gpu_cpu_quants.o"
M31_PIDS=()
for M31_TU in m31_engine m31_gpu_experts m31_generate; do
 /usr/bin/g++-14 -std=c++20 -O3 -DNDEBUG -DM31_HAVE_CUDA=1 -DM31_ENGINE_CUDA=1 -DM31_CPU_QUANTS_CUDA=1 "${M31_INC[@]}" -c "$M31_BUILD/source/$M31_TU.cpp" -o "$M31_BUILD/$M31_TU.o" &
 M31_PIDS+=("$!")
done
for M31_PID in "${M31_PIDS[@]}"; do wait "$M31_PID"; done
python3 - <<'PY'
import shlex,subprocess
from pathlib import Path
root=Path('/mnt/ai/Xolotl-m31'); source=root/'build-m31-cuda'; dest=root/'build-gpu-correctness'
args=shlex.split((source/'CMakeFiles/m31_generate.dir/link.txt').read_text())
idx=args.index('-o');args[idx-1]=str(dest/'m31_generate.o');args[idx+1]=str(dest/'m31_generate')
args.remove('libm31_engine.a')
args[idx-1:idx-1]=[str(dest/(n+'.o')) for n in ('m31_engine','m31_gpu_experts','gpu_cpu_quants')]
subprocess.run(args,cwd=source,check=True)
PY
printf 'GPU CPU-quants engine build completed: %s\n' "$(date -u --iso-8601=seconds)"
