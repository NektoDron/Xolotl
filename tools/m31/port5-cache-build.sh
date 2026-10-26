#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/ai/Xolotl-m31
BUILD=$ROOT/build-port5-cache
exec >>/mnt/ai/logs/m31-port/port5-cache-build-v1.log 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
mkdir -p "$BUILD/source"
cp "$ROOT"/tools/m31/*.hpp "$ROOT"/tools/m31/*.cuh "$ROOT/tools/m31/gpu_cpu_quants.cu" \
   "$ROOT/tools/m31/m31_gpu_experts.cpp" "$ROOT/tools/m31/port5-cache-check.cpp" \
   "$ROOT/tools/m31/cpu_gpu_host_executor.cpp" "$ROOT/tools/m31/cpu_gpu_host_ops.cpp" "$BUILD/source/"
INC=(-I"$ROOT/include" -I"$BUILD/source" -I"$ROOT/build-m31-cpu/_deps/strata_llamacpp-src/ggml/include" -I"$ROOT/build-m31-cpu/_deps/strata_llamacpp-src/ggml/src")
for TU in m31_gpu_experts port5-cache-check cpu_gpu_host_executor cpu_gpu_host_ops; do
 FLAGS=(-std=c++20 -O3 -DNDEBUG -DM31_ENGINE_CUDA=1 -DM31_CPU_QUANTS_CUDA=1 -DM31_GPU_PREFILL_TESTING=1)
 case "$TU" in cpu_gpu_host_executor|cpu_gpu_host_ops) FLAGS+=(-frounding-math -fno-fast-math);; esac
 /usr/bin/g++-14 "${FLAGS[@]}" "${INC[@]}" -c "$BUILD/source/$TU.cpp" -o "$BUILD/$TU.o"
done
/usr/local/cuda/bin/nvcc -std=c++20 -O3 -arch=sm_120a -ccbin=/usr/bin/g++-14 -U_GNU_SOURCE -D_DEFAULT_SOURCE -include /home/nekto/.local/include/cuda13_glibc_compat.h "${INC[@]}" -c "$BUILD/source/gpu_cpu_quants.cu" -o "$BUILD/gpu_cpu_quants.o"
python3 - <<'PY'
import shlex,subprocess
from pathlib import Path
r=Path('/mnt/ai/Xolotl-m31');src=r/'build-m31-cuda';out=r/'build-port5-cache'
args=shlex.split((src/'CMakeFiles/m31_generate.dir/link.txt').read_text())
i=args.index('-o');args[i-1]=str(out/'port5-cache-check.o');args[i+1]=str(out/'port5-cache-check')
args.remove('libm31_engine.a')
objects=[out/(name+'.o') for name in ('m31_gpu_experts','gpu_cpu_quants','cpu_gpu_host_executor','cpu_gpu_host_ops')]
args[i-1:i-1]=[str(p) for p in objects]
subprocess.run(args,cwd=src,check=True)
PY
