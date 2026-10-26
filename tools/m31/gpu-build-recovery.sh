#!/bin/bash
set -euo pipefail
M31_ROOT=/mnt/ai/Xolotl-m31
M31_BUILD=$M31_ROOT/build-gpu-correctness
exec >>/mnt/ai/logs/m31-port/gpu-build-recovery-v1.log 2>&1
exec 8>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 8
cd "$M31_ROOT"
M31_GGML=$M31_ROOT/build-m31-cpu/_deps/strata_llamacpp-src/ggml
M31_INC=(-I"$M31_ROOT/include" -I"$M31_ROOT/tools/m31" -I"$M31_GGML/include")
for M31_TU in m31_gpu_experts gpu_uploader_recovery_check; do
 /usr/bin/g++-14 -std=c++20 -O3 -DNDEBUG -DM31_ENGINE_CUDA=1 -DM31_GPU_PREFILL_TESTING=1 "${M31_INC[@]}" -c "tools/m31/$M31_TU.cpp" -o "$M31_BUILD/$M31_TU-recovery.o"
done
python3 - <<'PY'
import shlex,subprocess
from pathlib import Path
root=Path('/mnt/ai/Xolotl-m31'); source=root/'build-m31-cuda'; dest=root/'build-gpu-correctness'
args=shlex.split((source/'CMakeFiles/m31_generate.dir/link.txt').read_text())
idx=args.index('-o');args[idx-1]=str(dest/'gpu_uploader_recovery_check-recovery.o');args[idx+1]=str(dest/'gpu_uploader_recovery_check')
args.remove('libm31_engine.a');args.insert(idx-1,str(dest/'m31_gpu_experts-recovery.o'))
subprocess.run(args,cwd=source,check=True)
PY
printf 'GPU recovery probe build completed: %s\n' "$(date -u --iso-8601=seconds)"
