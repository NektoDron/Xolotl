#!/usr/bin/env bash
set -euo pipefail
M31_MMV_API_VERSION=${1:-v2}
[[ "$M31_MMV_API_VERSION" =~ ^v[23]$ ]] || exit 2
export M31_MMV_API_VERSION
exec >>"/mnt/ai/logs/m31-port/port7-mmv-api-build-$M31_MMV_API_VERSION.log" 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
python3 - <<'PY'
from pathlib import Path
import hashlib,json,os,shlex,shutil,subprocess
r=Path('/mnt/ai/Xolotl-m31');o=r/'build-port7-dense'/('mmv-api-'+os.environ['M31_MMV_API_VERSION']);b=r/'build-m31-cuda';s=o/'source';s.mkdir(exist_ok=False)
vendor=r/'build-m31-cpu/_deps/strata_llamacpp-src'
for name in ('port7-dense-mmv-check.cpp','port7-dense-mmv-kernels.cu','port7_dense_mmv_probe.hpp','port7-mmv-api-contract.cpp','m31_geometry.hpp'):
    shutil.copy2(r/'tools/m31'/name,s/name)
commands=json.loads((b/'compile_commands.json').read_text());records=[]
inc=['-I'+str(o/'vendor/ggml/include'),'-I'+str(s)]
old=next(v for v in commands if v['file'].endswith('/ggml-cuda.cu'))
for name,source in [('ggml-cuda.cu',o/'vendor/ggml/src/ggml-cuda/ggml-cuda.cu'),('kernels',s/'port7-dense-mmv-kernels.cu')]:
    argv=shlex.split(old['command']);argv[1:1]=inc
    argv[argv.index('-o')+1]=str(o/(name+'.o'));argv[argv.index('-c')+1]=str(source)
    subprocess.run(argv,cwd=old['directory'],check=True);records.append(argv)
archive=o/'libggml-cuda.a';archive.write_bytes((b/'ggml/src/ggml-cuda/libggml-cuda.a').read_bytes())
subprocess.run(['ar','r',str(archive),str(o/'ggml-cuda.cu.o')],check=True)
links={}
for target,source in [('dense-mmv-check','port7-dense-mmv-check.cpp'),('mmv-api-contract','port7-mmv-api-contract.cpp')]:
    argv=['/usr/bin/g++-14','-std=c++20','-O3','-DNDEBUG',*inc,'-I'+str(r/'include'),'-I'+str(vendor/'ggml/include'),'-I'+str(vendor/'vendor/nlohmann'),'-c',str(s/source),'-o',str(o/(target+'.o'))]
    subprocess.run(argv,check=True);records.append(argv)
    argv=shlex.split((b/'CMakeFiles/m31_generate.dir/link.txt').read_text())
    argv=[v for v in argv if not v.startswith('-Wl,--dependency-file=')]
    k=argv.index('-o');argv[k-1]=str(o/(target+'.o'));argv[k+1]=str(o/target)
    if target=='dense-mmv-check':argv[argv.index('libm31_engine.a')]=str(o/'kernels.o')
    else:argv.remove('libm31_engine.a')
    argv[argv.index('ggml/src/ggml-cuda/libggml-cuda.a')]=str(archive)
    subprocess.run(argv,cwd=b,check=True);links[target]=argv
cpu=subprocess.run([str(o/'mmv-api-contract')],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,check=True)
(o/'cpu-contract.json').write_text(cpu.stdout)
files={str(p.relative_to(o)):hashlib.sha256(p.read_bytes()).hexdigest() for sub in ('source','vendor') for p in (o/sub).rglob('*') if p.is_file()}
receipt={'format':'m31-port7-mmv-api-build-'+os.environ['M31_MMV_API_VERSION'],'scope':'Per-node exact MMV marker, scalar/batch actual projection component, CPU API boundary contract. Canonical CUDA kernel libraries with one replaced ggml-cuda.cu.o; no private context layout changes.', 'source':files,'elf':{n:hashlib.sha256((o/n).read_bytes()).hexdigest() for n in links},'compile_argv':records,'link_argv':links,'cpu_contract':json.loads(cpu.stdout),'cuda_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'common_cuh_sha256':hashlib.sha256((vendor/'ggml/src/ggml-cuda/common.cuh').read_bytes()).hexdigest(),'patch_sha256':{name:hashlib.sha256((r/'tools/m31'/name).read_bytes()).hexdigest() for name in ('ggml-decode-fusion.patch','ggml-exact-mmv.patch')}}
(o/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'elf':receipt['elf'],'cpu_contract':receipt['cpu_contract']},indent=2))
PY
