#!/usr/bin/env bash
set -euo pipefail
export M31_RMS_BACKEND_VERSION=${1:-v1}
[[ "$M31_RMS_BACKEND_VERSION" =~ ^v[12]$ ]] || exit 2
exec >>"/mnt/ai/logs/m31-port/port7-rms-backend-build-${M31_RMS_BACKEND_VERSION}.log" 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
python3 - <<'PY'
from pathlib import Path
import hashlib,json,shlex,shutil,subprocess,os
r=Path('/mnt/ai/Xolotl-m31');o=r/'build-port7-dense'/('rms-backend-'+os.environ['M31_RMS_BACKEND_VERSION']);o.mkdir();b=r/'build-m31-cuda';vendor=r/'build-m31-cpu/_deps/strata_llamacpp-src';base=r/'build-port7-dense/mmv-api-v3/vendor';v=o/'vendor';v.mkdir()
for rel in ('ggml/include/ggml-cuda.h','ggml/src/ggml-cuda/ggml-cuda.cu'):
    p=v/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/rel,p)
patch=o/'ggml-rms-only-fusion.patch';shutil.copy2(r/'tools/m31/ggml-rms-only-fusion.patch',patch)
subprocess.run(['git','apply',str(patch)],cwd=v,env=dict(os.environ,GIT_CEILING_DIRECTORIES=str(o)),check=True)
# Verify CMake consumes both a new coherent state and its repeat configure.
check=o/'cmake-check.cmake';check.write_text('set(STRATA_M31 ON)\ninclude("'+str(r/'tools/m31/cmake/ggml_decode_fusion.cmake')+'")\nm31_patch_decode_fusion("'+str(v)+'" FALSE)\n')
subprocess.run(['cmake','-P',str(check)],check=True)
entries=json.loads((b/'compile_commands.json').read_text());old=next(x for x in entries if x['file'].endswith('/ggml-cuda.cu'))
argv=shlex.split(old['command']);argv[1:1]=['-I'+str(v/'ggml/include')];argv[argv.index('-c')+1]=str(v/'ggml/src/ggml-cuda/ggml-cuda.cu');argv[argv.index('-o')+1]=str(o/'ggml-cuda.cu.o')
subprocess.run(argv,cwd=old['directory'],check=True);cuda=argv
archive=o/'libggml-cuda.a';shutil.copy2(b/'ggml/src/ggml-cuda/libggml-cuda.a',archive);subprocess.run(['ar','r',str(archive),str(o/'ggml-cuda.cu.o')],check=True)
old=next(x for x in entries if x['file'].endswith('/m31_engine.cpp'));argv=shlex.split(old['command']);argv[1:1]=['-I'+str(v/'ggml/include')];argv[argv.index('-o')+1]=str(o/'m31_engine.cpp.o')
subprocess.run(argv,cwd=old['directory'],check=True);engine=argv
cpu=['/usr/bin/g++-14','-std=c++20','-O0','-fsyntax-only','-DNDEBUG','-DGGML_USE_CPU','-I'+str(r/'include'),'-I'+str(r/'tools/m31'),'-I'+str(vendor/'ggml/include'),'-I'+str(vendor/'ggml/src'),str(r/'tools/m31/m31_engine.cpp')];subprocess.run(cpu,check=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt={'scope':'RMS-only backend mode2; empty override atomic fastpath; one replaced ggml-cuda.cu TU; canonical kernel objects reused. CPU and CUDA Engine syntax/compile; model accuracy not yet proven.', 'cuda_compile':cuda,'engine_compile':engine,'cpu_syntax':cpu,'archive_sha256':sha(archive),'header_sha256':sha(v/'ggml/include/ggml-cuda.h'),'backend_source_sha256':sha(v/'ggml/src/ggml-cuda/ggml-cuda.cu'),'patch_sha256':sha(patch),'engine_sha256':sha(r/'tools/m31/m31_engine.cpp')}
(o/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
PY
