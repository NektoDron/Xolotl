#!/bin/bash
set -euo pipefail
M31_ROOT=/mnt/ai/Xolotl-m31
M31_BUILD=$M31_ROOT/build-gpu-correctness
exec >>/mnt/ai/logs/m31-port/gpu-build-cpu-quants-${1:-v1}.log 2>&1
exec 8>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 8
cd "$M31_ROOT"
M31_GGML=$M31_ROOT/build-m31-cpu/_deps/strata_llamacpp-src/ggml
M31_INC=(-I"$M31_ROOT/include" -I"$M31_ROOT/tools/m31" -I"$M31_GGML/include" -I"$M31_GGML/src")
/usr/local/cuda/bin/nvcc -std=c++20 -O3 -arch=sm_120a -ccbin=/usr/bin/g++-14 -U_GNU_SOURCE -D_DEFAULT_SOURCE -include /home/nekto/.local/include/cuda13_glibc_compat.h "${M31_INC[@]}" -c tools/m31/gpu_cpu_quants.cu -o "$M31_BUILD/gpu_cpu_quants.o"
/usr/bin/g++-14 -std=c++20 -O3 -DNDEBUG -DM31_HAVE_CUDA=1 -DM31_CPU_QUANTS_CUDA=1 "${M31_INC[@]}" -c tools/m31/m31_expert_check.cpp -o "$M31_BUILD/m31_expert_check.o"
python3 - <<'PY'
import shlex,subprocess
from pathlib import Path
root=Path('/mnt/ai/Xolotl-m31'); source=root/'build-m31-cuda'
args=shlex.split((source/'CMakeFiles/m31_expert_check.dir/link.txt').read_text())
idx=args.index('-o');args[idx-1]=str(root/'build-gpu-correctness/m31_expert_check.o');args[idx+1]=str(root/'build-gpu-correctness/m31_expert_check')
args.insert(idx-1,str(root/'build-gpu-correctness/gpu_cpu_quants.o'))
subprocess.run(args,cwd=source,check=True)
PY
printf 'GPU CPU-quants probe build completed: %s\n' "$(date -u --iso-8601=seconds)"
