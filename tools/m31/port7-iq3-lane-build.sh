#!/usr/bin/env bash
set -euo pipefail
exec >>/mnt/ai/logs/m31-port/port7-iq3-lane-build-v1.log 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
python3 - <<'PY'
from pathlib import Path
import hashlib,json,shlex,shutil,subprocess
r=Path('/mnt/ai/Xolotl-m31');o=r/'build-port7-dense/iq3-lane-v1';o.mkdir();s=o/'source';s.mkdir();b=r/'build-m31-cuda';v=r/'build-m31-cpu/_deps/strata_llamacpp-src'
for name in ('port7_iq3_lane_imma.cu','port7_iq3_lane_check.cpp','port7_iq3_lane_probe.hpp','port7_iq3_lane_reference.hpp','gpu_resident_dot.hpp','m31_geometry.hpp'):shutil.copy2(r/'tools/m31'/name,s/name)
entries=json.loads((b/'compile_commands.json').read_text());old=next(x for x in entries if x['file'].endswith('/gpu_cpu_quants.cu'))
argv=shlex.split(old['command']);argv[argv.index('-c')+1]=str(s/'port7_iq3_lane_imma.cu');argv[argv.index('-o')+1]=str(o/'imma.o');argv[1:1]=['-O3','--ftz=false','--fmad=true','--prec-div=true','--prec-sqrt=true','-Xptxas=-v','-I'+str(s)]
subprocess.run(argv,cwd=old['directory'],check=True);cuda_argv=argv
argv=['/usr/bin/g++-14','-std=c++20','-O3','-DNDEBUG','-fno-fast-math','-ffp-contract=off','-frounding-math','-I'+str(s),'-I'+str(r/'include'),'-I'+str(v/'ggml/src'),'-I'+str(v/'ggml/include'),'-I'+str(v/'vendor/nlohmann'),'-I/usr/local/cuda/include','-I/usr/local/cuda/include/cccl','-c',str(s/'port7_iq3_lane_check.cpp'),'-o',str(o/'check.o')]
subprocess.run(argv,check=True);host_argv=argv
argv=shlex.split((b/'CMakeFiles/m31_generate.dir/link.txt').read_text());argv=[x for x in argv if not x.startswith('-Wl,--dependency-file=')]
k=argv.index('-o');argv[k-1]=str(o/'check.o');argv[k+1]=str(o/'iq3-lane-check');k=argv.index('libm31_engine.a');argv[k:k]=[str(o/'imma.o')]
subprocess.run(argv,cwd=b,check=True)
fixture=Path('/mnt/ai/logs/m31-port/port7-iq3-lane-fixture-v1');fixture.mkdir();source=Path('/mnt/ai/logs/m31-port/gpu-integrated-l3-v1/l3_input.bin');shutil.copy2(source,fixture/'l3_input.bin')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt={'scope':'Isolated new IQ3 n1 masked-int8IMMA stage1 exactI32 perK256/CPUlane8 plus originalFMA finish. Sequential oneNVCC thenoneCXX; no runtime/default/vendor changes. Actual CPU/GPU bit+timing proof pending.', 'source':{p.name:sha(p) for p in s.iterdir()},'cuda_compile':cuda_argv,'host_compile':host_argv,'link_argv':argv,'elf_sha256':sha(o/'iq3-lane-check'),'old_engine_archive_sha256':sha(b/'libm31_engine.a'),'old_cuda_library_sha256':sha(b/'ggml/src/ggml-cuda/libggml-cuda.a'),'old_cpu_library_sha256':sha(b/'ggml/src/libggml-cpu.a'),'ggml_common_sha256':sha(v/'ggml/src/ggml-common.h'),'stock_x86_quants_sha256':sha(v/'ggml/src/ggml-cpu/arch/x86/quants.c'),'fixture':str(fixture/'l3_input.bin'),'fixture_sha256':sha(fixture/'l3_input.bin'),'nvidia_primary':'https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16832'}
(o/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
PY
