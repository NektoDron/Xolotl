#!/usr/bin/env python3
"""Frozen integrated IQ3/GU helper component; one NVCC, then two CXX."""
import argparse,concurrent.futures,hashlib,json,shlex,shutil,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=Path('/mnt/ai/Xolotl-m31');b=r/'build-m31-cuda';o=a.out.resolve();o.mkdir(parents=True,exist_ok=False);s=o/'source';s.mkdir()
for f in (r/'tools/m31').iterdir():
 if f.is_file() and f.suffix in ['.hpp','.cpp','.inc','.cuh','.cu']:shutil.copy2(f,s/f.name)
shutil.copy2('/mnt/ai/logs/m31-port/port7-cached-grouped-fixture-v1.json',o/'fixture.json');entries=json.loads((b/'compile_commands.json').read_text());commands={}
e=next(e for e in entries if e['file'].endswith('/gpu_cpu_quants.cu'));cmd=shlex.split(e['command']);cmd[cmd.index('-o')+1]=str(o/'gpu_cpu_quants.o');cmd[cmd.index('-c')+1]=str(s/'gpu_cpu_quants.cu');cmd[1:1]=['-O3','--ftz=false','--fmad=true','--prec-div=true','--prec-sqrt=true','-I'+str(s)];commands['gpu_cpu_quants']=cmd;subprocess.run(cmd,cwd=b,check=True)
e=next(e for e in entries if e['file'].endswith('/m31_engine.cpp'))
def compile(n):
 cmd=shlex.split(e['command']);cmd=[('-I'+str(s)) if x=='-I'+str(r/'tools/m31') else x for x in cmd];cmd[cmd.index('-o')+1]=str(o/(n+'.o'));cmd[-1]=str(s/(n+'.cpp'));cmd[1:1]=['-fno-fast-math','-frounding-math','-I'+str(r/'build-m31-cpu/_deps/strata_llamacpp-src/vendor/nlohmann')];commands[n]=cmd;subprocess.run(cmd,cwd=b,check=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(compile,['gpu_cached_exact','port7_cached_imma_gate_up_check']))
link=shlex.split((b/'CMakeFiles/m31_generate.dir/link.txt').read_text());link=[x for x in link if not x.startswith('-Wl,--dependency-file=')];i=link.index('-o');link[i-1]=str(o/'port7_cached_imma_gate_up_check.o');link[i+1]=str(o/'cached-imma-gu-check');i=link.index('libm31_engine.a');link[i:i]=[str(o/'gpu_cached_exact.o'),str(o/'gpu_cpu_quants.o')];subprocess.run(link,cwd=b,check=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();receipt=dict(format='m31-port7-cached-imma-gu-build-v1',source={p.name:sha(p) for p in s.iterdir()},compile=commands,link=link,fixture_sha256=sha(o/'fixture.json'),elf_sha256=sha(o/'cached-imma-gu-check'),scope='Integrated exact IQ3 count1 plus adjacent GU helper A/B; CPU scales/OAI/original rank unchanged; numerical and cold helper timing pending; no Engine/TG acceptance');(o/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['elf_sha256'])
