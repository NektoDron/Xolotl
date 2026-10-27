#!/usr/bin/env python3
"""Small isolated packed routed metadata component; no Engine/CMake edits."""
import argparse,hashlib,json,shlex,shutil,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=Path('/mnt/ai/Xolotl-m31');base=r/'build-m31-cuda';v=r/'build-m31-cpu/_deps/strata_llamacpp-src';o=a.out.resolve();o.mkdir(parents=True,exist_ok=False);s=o/'source';s.mkdir()
for n in ['m31_dspark_packed_routes.hpp','port7_packed_routes_check.cpp']:shutil.copy2(r/'tools/m31'/n,s/n)
compile=['/usr/bin/g++-14','-std=c++20','-O3','-DNDEBUG','-fno-fast-math','-frounding-math','-DM31_ENGINE_CUDA=1','-I'+str(s),'-I'+str(v/'ggml/include'),'-c',str(s/'port7_packed_routes_check.cpp'),'-o',str(o/'check.o')];subprocess.run(compile,check=True)
link=shlex.split((base/'CMakeFiles/m31_generate.dir/link.txt').read_text());link=[x for x in link if not x.startswith('-Wl,--dependency-file=')];i=link.index('-o');link[i-1]=str(o/'check.o');link[i+1]=str(o/'packed-routes-check');subprocess.run(link,cwd=base,check=True)
run=subprocess.run([str(o/'packed-routes-check'),'cpu',str(o/'cpu-result.json')],text=True,capture_output=True);(o/'cpu.log').write_text(run.stdout+run.stderr);run.check_returncode()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();receipt=dict(format='m31-port7-packed-routes-build-v1',source={p.name:sha(p) for p in s.iterdir()},compile=compile,link=link,cpu_result=json.loads((o/'cpu-result.json').read_text()),elf_sha256=sha(o/'packed-routes-check'),cuda_library_sha256=sha(base/'ggml/src/ggml-cuda/libggml-cuda.a'),scope='Standalone concat/cast/unpack only; live Engine and common CMake unchanged; CUDA runtime proof pending')
(o/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt['cpu_result']))
