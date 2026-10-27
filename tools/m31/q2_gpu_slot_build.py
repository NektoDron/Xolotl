#!/usr/bin/env python3
"""One CXX compilation against the frozen canonical port7 CUDA libraries."""
import argparse,hashlib,json,shlex,shutil,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
r=Path('/mnt/ai/Xolotl-m31');base=r/'build-port7-canonical/v2';b=base/'build-cuda';o=a.out.resolve();o.mkdir(parents=True,exist_ok=False)
src=o/'q2_gpu_slot_check.cpp';shutil.copy2(r/'tools/m31/q2_gpu_slot_check.cpp',src)
entries=json.loads((base/'cuda-compile_commands.json').read_text())
e=next(e for e in entries if e['file'].endswith('/m31_expert_check.cpp'))
cmd=shlex.split(e['command']);cmd[cmd.index('-o')+1]=str(o/'q2_gpu_slot_check.o');cmd[cmd.index('-c')+1]=str(src)
cmd[1:1]=['-I'+str(base/'vendor-managed/vendor')]
subprocess.run(cmd,cwd=b,check=True)
link=shlex.split((base/'cuda-m31_expert_check-link.txt').read_text())
link=[x for x in link if not x.startswith('-Wl,--dependency-file=') and x!='libm31_engine.a']
i=link.index('-o');link[i-1]=str(o/'q2_gpu_slot_check.o');link[i+1]=str(o/'q2-gpu-slot-check')
subprocess.run(link,cwd=b,check=True)
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
report={'format':'m31-q2-gpu-slot-build-v1','source_sha256':sha(src),'elf_sha256':sha(o/'q2-gpu-slot-check'),'compile':cmd,'link':link,'canonical_receipt_sha256':sha(base/'build-receipt.json'),'scope':'Standalone real IQ2 native GGML CUDA cache-view component; no runtime modifications; numerical proof pending'}
(o/'build-receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(report['elf_sha256'])
