#!/usr/bin/env bash
set -euo pipefail
exec >>/mnt/ai/logs/m31-port/port7-rms-selective-build-v1.log 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
python3 - <<'PY'
from pathlib import Path
import hashlib,json,shlex,shutil,subprocess
r=Path('/mnt/ai/Xolotl-m31');o=r/'build-port7-dense/rms-selective-v1';o.mkdir();s=o/'source';s.mkdir();b=r/'build-m31-cuda';v=r/'build-m31-cpu/_deps/strata_llamacpp-src'
for name in ('port7-rms-selective.cu','port7-rms-selective-check.cpp','port7-dense-ops-check.cpp','m31_geometry.hpp','m31_dspark_batched_ops.hpp'):shutil.copy2(r/'tools/m31'/name,s/name)
entries=json.loads((b/'compile_commands.json').read_text());old=next(x for x in entries if x['file'].endswith('/ggml-cuda.cu'))
argv=shlex.split(old['command']);argv[argv.index('-c')+1]=str(s/'port7-rms-selective.cu');argv[argv.index('-o')+1]=str(o/'rms.o');argv[1:1]=['-I'+str(v/'ggml/src')]
subprocess.run(argv,cwd=old['directory'],check=True);cuda_argv=argv
argv=['/usr/bin/g++-14','-std=c++20','-O3','-DNDEBUG','-I'+str(s),'-I'+str(r/'include'),'-I'+str(v/'ggml/include'),'-I'+str(v/'vendor/nlohmann'),'-c',str(s/'port7-rms-selective-check.cpp'),'-o',str(o/'check.o')]
subprocess.run(argv,check=True);host_argv=argv
argv=shlex.split((b/'CMakeFiles/m31_generate.dir/link.txt').read_text());argv=[x for x in argv if not x.startswith('-Wl,--dependency-file=')]
k=argv.index('-o');argv[k-1]=str(o/'check.o');argv[k+1]=str(o/'rms-selective-check');k=argv.index('libm31_engine.a');argv[k:k+1]=[str(o/'rms.o'),str(r/'build-port7-dense/ops-v1/timing.o')]
subprocess.run(argv,cwd=b,check=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt={'scope':'Direct stock rms_norm+mul versus stock rms_norm_fused only; same archived kernels, no backend/default modifications. CUDA graph replay component timing.', 'source':{p.name:sha(p) for p in s.iterdir()},'cuda_compile':cuda_argv,'host_compile':host_argv,'link_argv':argv,'elf_sha256':sha(o/'rms-selective-check'),'cuda_archive_sha256':sha(b/'ggml/src/ggml-cuda/libggml-cuda.a'),'norm_source_sha256':sha(v/'ggml/src/ggml-cuda/norm.cu')}
(o/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
PY
