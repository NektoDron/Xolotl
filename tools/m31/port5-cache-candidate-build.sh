#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/ai/Xolotl-m31
VERSION=${1:-v1}
if [[ ! "$VERSION" =~ ^v[0-9]+$ ]]; then exit 2; fi
BUILD=$ROOT/build-port5-cache/candidate-$VERSION
export M31_CACHE_CANDIDATE_BUILD="$BUILD"
exec >>"/mnt/ai/logs/m31-port/port5-cache-candidate-build-$VERSION.log" 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
INC=(-I"$ROOT/include" -I"$BUILD/source" -I"$ROOT/build-m31-cpu/_deps/strata_llamacpp-src/ggml/include" -I"$ROOT/build-m31-cpu/_deps/strata_llamacpp-src/ggml/src" -I"$ROOT/build-m31-cpu/_deps/strata_llamacpp-src/vendor/nlohmann")
PIDS=()
for TU in m31_engine m31_gpu_experts cpu_gpu_host_executor cpu_gpu_host_ops m31_generate m31_serve; do
 FLAGS=(-std=c++20 -O3 -DNDEBUG -DM31_ENGINE_CUDA=1 -DM31_HAVE_CUDA=1 -DM31_CPU_QUANTS_CUDA=1)
 case "$TU" in cpu_gpu_host_executor|cpu_gpu_host_ops) FLAGS+=(-frounding-math -fno-fast-math);; esac
 /usr/bin/g++-14 "${FLAGS[@]}" "${INC[@]}" -c "$BUILD/source/$TU.cpp" -o "$BUILD/$TU.o" &
 PIDS+=("$!")
done
/usr/local/cuda/bin/nvcc -std=c++20 -O3 -arch=sm_120a -ccbin=/usr/bin/g++-14 -U_GNU_SOURCE -D_DEFAULT_SOURCE -include /home/nekto/.local/include/cuda13_glibc_compat.h "${INC[@]}" -c "$BUILD/source/gpu_cpu_quants.cu" -o "$BUILD/gpu_cpu_quants.o" &
PIDS+=("$!")
FAIL=0
for PID in "${PIDS[@]}"; do wait "$PID" || FAIL=1; done
if [ "$FAIL" != 0 ]; then exit 1; fi
python3 - <<'PY'
import hashlib,json,os,shlex,subprocess
from pathlib import Path
r=Path('/mnt/ai/Xolotl-m31');src=r/'build-m31-cuda';out=Path(os.environ['M31_CACHE_CANDIDATE_BUILD'])
for target in ('m31_generate','m31_serve'):
 args=shlex.split((src/f'CMakeFiles/{target}.dir/link.txt').read_text())
 i=args.index('-o');args[i-1]=str(out/f'{target}.o');args[i+1]=str(out/target)
 args.remove('libm31_engine.a')
 args[i-1:i-1]=[str(out/(name+'.o')) for name in ('m31_engine','m31_gpu_experts','gpu_cpu_quants','cpu_gpu_host_executor','cpu_gpu_host_ops')]
 subprocess.run(args,cwd=src,check=True)
receipt={'source':json.loads((out/'source-sha256.json').read_text()),'elf':{name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ('m31_generate','m31_serve')}}
(out/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
PY
