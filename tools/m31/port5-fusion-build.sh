#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/ai/Xolotl-m31
BUILD=$ROOT/build-port5-cache
exec >>/mnt/ai/logs/m31-port/port5-fusion-build-v1.log 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
INC=(-I"$ROOT/include" -I"$BUILD/fusion-source" -I"$ROOT/build-m31-cpu/_deps/strata_llamacpp-src/ggml/include")
for TU in m31_engine m31_generate; do
 /usr/bin/g++-14 -std=c++20 -O3 -DNDEBUG -DM31_ENGINE_CUDA=1 -DM31_HAVE_CUDA=1 -DM31_CPU_QUANTS_CUDA=1 "${INC[@]}" -c "$BUILD/fusion-source/$TU.cpp" -o "$BUILD/fusion-$TU.o"
done
python3 - <<'PY'
import shlex,subprocess
from pathlib import Path
r=Path('/mnt/ai/Xolotl-m31');src=r/'build-m31-cuda';out=r/'build-port5-cache';frozen=out/'candidate-v1'
args=shlex.split((src/'CMakeFiles/m31_generate.dir/link.txt').read_text())
i=args.index('-o');args[i-1]=str(out/'fusion-m31_generate.o');args[i+1]=str(out/'fusion-m31_generate')
args.remove('libm31_engine.a')
objects=[out/'fusion-m31_engine.o']+[frozen/(name+'.o') for name in ('m31_gpu_experts','gpu_cpu_quants','cpu_gpu_host_executor','cpu_gpu_host_ops')]
args[i-1:i-1]=[str(p) for p in objects]
subprocess.run(args,cwd=src,check=True)
PY
