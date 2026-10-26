#!/usr/bin/env bash
set -euo pipefail
M31_ROUTE_ROOT=/mnt/ai/Xolotl-m31
M31_ROUTE_BUILD=$M31_ROUTE_ROOT/build-streaming-prefill/routes
exec >>/mnt/ai/logs/m31-port/prefill-route-build.log 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
mkdir -p "$M31_ROUTE_BUILD/source"
python3 - <<'PY'
from pathlib import Path
import hashlib,json,shutil
root=Path('/mnt/ai/Xolotl-m31'); tools=root/'tools/m31'; out=root/'build-streaming-prefill/routes/source'
source=tools/'gpu_cpu_quants.cu'
sha=hashlib.sha256(source.read_bytes()).hexdigest()
prototype=(tools/'prefill_route_quants.cu').read_text()
assert ('Frozen source SHA256 '+sha) in prototype, 'production kernel changed: review/rebase independent prototype first'
files=[tools/'gpu_cpu_quants.cu',tools/'gpu_cpu_quants.hpp']
files += [p for p in tools.glob('prefill_route_*') if p.suffix in ('.cpp','.hpp','.cu')]
manifest={}
for p in files:
 shutil.copy2(p,out/p.name)
 manifest[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
PY
cmake -S "$M31_ROUTE_ROOT/tools/m31/prefill_route_build" -B "$M31_ROUTE_BUILD" \
 -DM31_ROUTE_SOURCE="$M31_ROUTE_BUILD/source" -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14 -DCMAKE_CUDA_ARCHITECTURES=120a \
 -DCUDAToolkit_ROOT=/usr/local/cuda \
 '-DCMAKE_CUDA_FLAGS=-U_GNU_SOURCE -D_DEFAULT_SOURCE -include /home/nekto/.local/include/cuda13_glibc_compat.h'
cmake --build "$M31_ROUTE_BUILD" -j24
CUDA_VISIBLE_DEVICES= "$M31_ROUTE_BUILD/m31_route_worklist_test" \
 > /mnt/ai/logs/m31-port/prefill-route-cpu-check-v1.json
