#!/usr/bin/env python3
"""One CXX pure CPU merge proposal; check only here, clean benchmark separately."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=Path('/mnt/ai/Xolotl-m31');o=a.out.resolve();o.mkdir(parents=True,exist_ok=False);s=o/'source';s.mkdir()
for n in ['cached_decode_merge.hpp','cached_decode_merge_rank_outer.hpp','port7_cached_merge_rank_check.cpp']:shutil.copy2(r/'tools/m31'/n,s/n)
cmd=['/usr/bin/g++-14','-std=c++20','-O3','-frounding-math','-fno-fast-math','-ffp-contract=off','-I'+str(s),str(s/'port7_cached_merge_rank_check.cpp'),'-o',str(o/'merge-rank-check')];subprocess.run(cmd,check=True);run=subprocess.run([str(o/'merge-rank-check'),'check',str(o/'check-result.json')],capture_output=True,text=True);(o/'check.log').write_text(run.stdout+run.stderr);run.check_returncode()
for symbol in ['port7_merge_old','port7_merge_rank_outer']:(o/(symbol+'.asm')).write_text(subprocess.check_output(['objdump','-d','-C','--disassemble='+symbol,str(o/'merge-rank-check')],text=True))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();receipt=dict(format='m31-port7-cached-merge-rank-build-v1',source={p.name:sha(p) for p in s.iterdir()},compile=cmd,check=json.loads((o/'check-result.json').read_text()),elf_sha256=sha(o/'merge-rank-check'),scope='Unused CPU helper proposal; raw bitproof only here; ns/call benchmark requires clean X window; no production changes');(o/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt['check']))
