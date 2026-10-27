#!/usr/bin/env bash
set -euo pipefail
M31_DENSE_OPS_VERSION=${1:-v1}
[[ "$M31_DENSE_OPS_VERSION" =~ ^v[12]$ ]] || exit 2
export M31_DENSE_OPS_VERSION
exec >>"/mnt/ai/logs/m31-port/port7-dense-ops-build-$M31_DENSE_OPS_VERSION.log" 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
python3 - <<'PY'
from pathlib import Path
import hashlib,json,shlex,shutil,subprocess,os
r=Path('/mnt/ai/Xolotl-m31');o=r/'build-port7-dense'/('ops-'+os.environ['M31_DENSE_OPS_VERSION']);o.mkdir();s=o/'source';s.mkdir()
b=r/'build-m31-cuda';v=r/'build-m31-cpu/_deps/strata_llamacpp-src';records=[]
for name in ('port7-dense-ops-check.cpp','port7-dense-ops-timing.cu','port7_dense_ops_prepare.py','m31_geometry.hpp','m31_dspark_batched_ops.hpp','port7-batched-ops-contract.cpp'):
    shutil.copy2(r/'tools/m31'/name,s/name)
argv=['/usr/bin/g++-14','-std=c++20','-O3','-DNDEBUG','-I'+str(s),'-I'+str(r/'include'),'-I'+str(v/'ggml/include'),'-I'+str(v/'vendor/nlohmann'),'-c',str(s/'port7-dense-ops-check.cpp'),'-o',str(o/'check.o')]
if os.environ['M31_DENSE_OPS_VERSION']=='v2':argv.insert(1,'-DM31_OPS_PRODUCTION_HELPERS=1')
subprocess.run(argv,check=True);records.append(argv)
commands=json.loads((b/'compile_commands.json').read_text());old=next(x for x in commands if x['file'].endswith('/ggml-cuda.cu'))
argv=shlex.split(old['command']);argv[argv.index('-c')+1]=str(s/'port7-dense-ops-timing.cu');argv[argv.index('-o')+1]=str(o/'timing.o')
if os.environ['M31_DENSE_OPS_VERSION']=='v1':
    subprocess.run(argv,cwd=old['directory'],check=True);records.append(argv)
else:shutil.copy2(r/'build-port7-dense/ops-v1/timing.o',o/'timing.o')
argv=shlex.split((b/'CMakeFiles/m31_generate.dir/link.txt').read_text());argv=[x for x in argv if not x.startswith('-Wl,--dependency-file=')]
k=argv.index('-o');argv[k-1]=str(o/'check.o');argv[k+1]=str(o/'dense-ops-check');argv[argv.index('libm31_engine.a')]=str(o/'timing.o')
subprocess.run(argv,cwd=b,check=True)
fixture=Path('/mnt/ai/logs/m31-port/port7-dense-ops-fixture-v1')
if not fixture.exists():subprocess.run(['/usr/bin/python3',str(s/'port7_dense_ops_prepare.py'),'--out',str(fixture)],check=True)
contract=['/usr/bin/g++-14','-std=c++20','-O2','-I'+str(s),'-I'+str(r/'include'),'-I'+str(v/'ggml/include'),str(s/'port7-batched-ops-contract.cpp'),str(b/'ggml/src/libggml-base.a'),'-lpthread','-lm','-o',str(o/'batched-ops-contract')]
subprocess.run(contract,check=True)
checked=subprocess.run([str(o/'batched-ops-contract')],capture_output=True,text=True,check=True)
(o/'cpu-contract.json').write_text(checked.stdout)
receipt={'scope':'Stock unary/RMS/RoPE/GLU/router batching component; original canonical v8 CUDA library; helper CUDA TU only records events and launches stock graphs, no new kernels.', 'source':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in s.iterdir()},'compile_argv':records,'link_argv':argv,'elf_sha256':hashlib.sha256((o/'dense-ops-check').read_bytes()).hexdigest(),'cuda_library_sha256':hashlib.sha256((b/'ggml/src/ggml-cuda/libggml-cuda.a').read_bytes()).hexdigest(),'common_cuh_sha256':hashlib.sha256((v/'ggml/src/ggml-cuda/common.cuh').read_bytes()).hexdigest(),'cpu_contract':json.loads(checked.stdout),'production_helpers':os.environ['M31_DENSE_OPS_VERSION']=='v2','fixture':str(fixture),'fixture_manifest_sha256':hashlib.sha256((fixture/'manifest.json').read_bytes()).hexdigest()}
(o/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
PY
