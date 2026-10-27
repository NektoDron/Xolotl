#!/usr/bin/env bash
set -euo pipefail
M31_DENSE_MSA_VERSION=${1:-v1}
[[ "$M31_DENSE_MSA_VERSION" =~ ^v[12]$ ]] || exit 2
export M31_DENSE_MSA_VERSION
exec >>"/mnt/ai/logs/m31-port/port7-dense-msa-build-$M31_DENSE_MSA_VERSION.log" 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
python3 - <<'PY'
from pathlib import Path
import hashlib,json,shlex,shutil,subprocess,os
r=Path('/mnt/ai/Xolotl-m31');version=os.environ['M31_DENSE_MSA_VERSION'];o=r/('build-port7-dense/msa-'+version);o.mkdir();s=o/'source';s.mkdir()
b=r/'build-m31-cuda';v=r/'build-m31-cpu/_deps/strata_llamacpp-src';api=r/'build-port7-dense/mmv-api-v3'
for name in ('port7-dense-msa-check.cpp','port7-dense-ops-check.cpp','port7-dense-ops-timing.cu','port7_dense_msa_prepare.py','m31_geometry.hpp','m31_dspark_batched_ops.hpp','m31_dspark_batched_msa.hpp'):
    shutil.copy2(r/'tools/m31'/name,s/name)
compile_argv=['/usr/bin/g++-14','-std=c++20','-O3','-DNDEBUG','-I'+str(s),'-I'+str(r/'include'),'-I'+str(v/'ggml/include'),'-I'+str(v/'vendor/nlohmann'),'-c',str(s/'port7-dense-msa-check.cpp'),'-o',str(o/'check.o')]
if version=='v2':compile_argv[1:1]=['-DM31_MSA_PRODUCTION_HELPER=1','-I'+str(api/'vendor/ggml/include')]
subprocess.run(compile_argv,check=True)
shutil.copy2(r/'build-port7-dense/ops-v1/timing.o',o/'timing.o')
argv=shlex.split((b/'CMakeFiles/m31_generate.dir/link.txt').read_text());argv=[x for x in argv if not x.startswith('-Wl,--dependency-file=')]
k=argv.index('-o');argv[k-1]=str(o/'check.o');argv[k+1]=str(o/'dense-msa-check');argv[argv.index('libm31_engine.a')]=str(o/'timing.o')
cuda=b/'ggml/src/ggml-cuda/libggml-cuda.a'
if version=='v2':cuda=api/'libggml-cuda.a';argv[argv.index('ggml/src/ggml-cuda/libggml-cuda.a')]=str(cuda)
subprocess.run(argv,cwd=b,check=True)
fixture=Path('/mnt/ai/logs/m31-port/port7-dense-msa-fixture-v1')
if not fixture.exists():subprocess.run(['/usr/bin/python3',str(s/'port7_dense_msa_prepare.py'),'--out',str(fixture)],check=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt={'scope':'Immutable real-derived MSA ne2-channel stock ops only. CXX-only existing event timer; no score marker/FA/Engine changes. V2 compares actual production context helper against n1 scalar stages on exact-MmvAPIv3 backend.', 'production_helper':version=='v2','source':{p.name:sha(p) for p in s.iterdir()},'compile_argv':compile_argv,'link_argv':argv,'elf_sha256':sha(o/'dense-msa-check'),'cuda_library_sha256':sha(cuda),'common_cuh_sha256':sha(v/'ggml/src/ggml-cuda/common.cuh'),'fixture':str(fixture),'fixture_manifest_sha256':sha(fixture/'manifest.json')}
(o/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
PY
