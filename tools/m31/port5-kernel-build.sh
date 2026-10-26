#!/usr/bin/env bash
set -euo pipefail
M31_KERNEL_ROOT=/mnt/ai/Xolotl-m31
M31_KERNEL_BUILD=$M31_KERNEL_ROOT/build-port5-kernel
M31_KERNEL_VERSION=${1:-v1}
exec >>/mnt/ai/logs/m31-port/port5-kernel-build-$M31_KERNEL_VERSION.log 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
mkdir -p "$M31_KERNEL_BUILD/source"
cp "$M31_KERNEL_ROOT"/tools/m31/{gpu_cpu_quants.cu,gpu_cpu_quants.hpp,gpu_cpu_quants_batched.cuh,prefill_route_quants.cu,prefill_route_quants.hpp,prefill_route_worklist.hpp,prefill_route_probe.cpp,prefill_route_worklist_test.cpp,cpu_gpu_host_executor.cpp,cpu_gpu_host_executor.hpp,cpu_gpu_host_ops.cpp,cpu_gpu_host_ops.hpp} "$M31_KERNEL_BUILD/source/"
cp "$M31_KERNEL_ROOT/tools/m31/port5_kernel_speed.cpp" "$M31_KERNEL_BUILD/source/"
cp "$M31_KERNEL_ROOT/tools/m31/gpu_cpu_quants_tensor.cuh" "$M31_KERNEL_BUILD/source/"
cp "$M31_KERNEL_ROOT/tools/m31/gpu_pinned_scratch.hpp" "$M31_KERNEL_BUILD/source/"
cp "$M31_KERNEL_ROOT"/tools/m31/{gpu_prepared_pair.cu,gpu_prepared_pair.hpp,port5_prepared_pair_check.cpp} "$M31_KERNEL_BUILD/source/"
cp "$M31_KERNEL_ROOT/tools/m31/port5_tensor_check.cpp" "$M31_KERNEL_BUILD/source/"
cmake -S "$M31_KERNEL_ROOT/tools/m31/port5-kernel-build" -B "$M31_KERNEL_BUILD" \
 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 \
 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14 \
 -DCMAKE_CUDA_ARCHITECTURES=120a -DCUDAToolkit_ROOT=/usr/local/cuda \
 '-DCMAKE_CUDA_FLAGS=-U_GNU_SOURCE -D_DEFAULT_SOURCE -include /home/nekto/.local/include/cuda13_glibc_compat.h'
cmake --build "$M31_KERNEL_BUILD" -j24
sha256sum "$M31_KERNEL_BUILD/source/"* "$M31_KERNEL_BUILD/m31_route_probe"
python3 - "$M31_KERNEL_VERSION" <<'PY'
from pathlib import Path
import hashlib,json,shutil,sys
root=Path('/mnt/ai/Xolotl-m31/build-port5-kernel')
out=root/('frozen-'+sys.argv[1])
out.mkdir(exist_ok=False)
shutil.copytree(root/'source',out/'source')
names=['m31_route_probe','port5_kernel_speed','port5_prepared_pair_check','port5_tensor_check']
for name in names: shutil.copy2(root/name,out/name)
manifest={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for p in out.rglob('*'):
    if p.is_file(): p.chmod(0o555 if p.name in names else 0o444)
print('immutable artifacts',out)
PY
