#!/usr/bin/env bash
set -euo pipefail
M31_SEED_ROOT=/mnt/ai/Xolotl-m31
M31_SEED_BUILD=$M31_SEED_ROOT/build-cache-ranking/seed-runtime
exec >>/mnt/ai/logs/m31-port/cache-seed-runtime-build.log 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
mkdir -p "$M31_SEED_BUILD/source"
cp "$M31_SEED_ROOT"/tools/m31/*.hpp "$M31_SEED_ROOT/tools/m31/m31_gpu_experts.cpp" \
   "$M31_SEED_ROOT/tools/m31/cache_seed_gpu_check.cpp" "$M31_SEED_BUILD/source/"
M31_SEED_INC=(-I"$M31_SEED_ROOT/include" -I"$M31_SEED_BUILD/source" \
 -I"$M31_SEED_ROOT/build-m31-cpu/_deps/strata_llamacpp-src/ggml/include")
M31_SEED_PIDS=()
for M31_SEED_TU in m31_gpu_experts cache_seed_gpu_check; do
 /usr/bin/g++-14 -std=c++20 -O3 -DNDEBUG -DM31_ENGINE_CUDA=1 -DM31_GPU_PREFILL_TESTING=1 \
   "${M31_SEED_INC[@]}" -c "$M31_SEED_BUILD/source/$M31_SEED_TU.cpp" -o "$M31_SEED_BUILD/$M31_SEED_TU.o" &
 M31_SEED_PIDS+=("$!")
done
for M31_SEED_PID in "${M31_SEED_PIDS[@]}"; do wait "$M31_SEED_PID"; done
/usr/bin/g++-14 -std=c++17 -O3 -DNDEBUG "${M31_SEED_INC[@]}" \
 "$M31_SEED_ROOT/tools/m31/cache_seed_receipt_check.cpp" -o "$M31_SEED_BUILD/cache_seed_receipt_check"
python3 - <<'PY'
import shlex,subprocess
from pathlib import Path
r=Path('/mnt/ai/Xolotl-m31');source=r/'build-m31-cuda';out=r/'build-cache-ranking/seed-runtime'
args=shlex.split((source/'CMakeFiles/m31_generate.dir/link.txt').read_text())
index=args.index('-o');args[index-1]=str(out/'cache_seed_gpu_check.o');args[index+1]=str(out/'cache_seed_gpu_check')
args.remove('libm31_engine.a');args.insert(index-1,str(out/'m31_gpu_experts.o'))
subprocess.run(args,cwd=source,check=True)
PY
python3 "$M31_SEED_ROOT/tools/m31/cache_seed_test.py" \
 --reader "$M31_SEED_ROOT/build-cache-ranking/seed-reader/m31_cache_seed_reader" \
 --receipt-reader "$M31_SEED_BUILD/cache_seed_receipt_check"
