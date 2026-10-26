#!/bin/bash
# Reuse a self-consistent pre-dense-decode engine/header snapshot; only transport changes.
set -euo pipefail
M31_ROOT=/mnt/ai/Xolotl-m31
M31_BUILD=$M31_ROOT/build-gpu-correctness
exec >>/mnt/ai/logs/m31-port/gpu-build-pipeline-${1:-v1}.log 2>&1
exec 8>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 8
cd "$M31_ROOT"
M31_GGML=$M31_ROOT/build-m31-cpu/_deps/strata_llamacpp-src/ggml
M31_SOURCE=$M31_BUILD/source-pipeline
mkdir -p "$M31_SOURCE"
cp "$M31_BUILD/source/"*.hpp "$M31_BUILD/source/"*.cpp "$M31_BUILD/source/gpu_cpu_quants.cu" "$M31_SOURCE/"
cp tools/m31/m31_gpu_experts.cpp tools/m31/prefill_pipeline.hpp "$M31_SOURCE/"
M31_INC=(-I"$M31_ROOT/include" -I"$M31_SOURCE" -I"$M31_GGML/include" -I"$M31_GGML/src")
/usr/bin/g++-14 -std=c++20 -O3 -DNDEBUG -DM31_HAVE_CUDA=1 -DM31_ENGINE_CUDA=1 -DM31_CPU_QUANTS_CUDA=1 "${M31_INC[@]}" -c "$M31_SOURCE/m31_gpu_experts.cpp" -o "$M31_BUILD/m31_gpu_experts-pipeline.o"
python3 - <<'PY'
import shlex,subprocess
from pathlib import Path
root=Path('/mnt/ai/Xolotl-m31'); source=root/'build-m31-cuda'; dest=root/'build-gpu-correctness'
args=shlex.split((source/'CMakeFiles/m31_generate.dir/link.txt').read_text())
idx=args.index('-o');args[idx-1]=str(dest/'m31_generate.o');args[idx+1]=str(dest/'m31_generate-pipeline')
args.remove('libm31_engine.a')
args[idx-1:idx-1]=[str(dest/(n+'.o')) for n in ('m31_engine','m31_gpu_experts-pipeline','gpu_cpu_quants')]
subprocess.run(args,cwd=source,check=True)
PY
printf 'GPU pipeline engine build completed: %s\n' "$(date -u --iso-8601=seconds)"
