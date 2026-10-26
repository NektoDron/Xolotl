#!/usr/bin/env python3
"""Prepare reviewable arena parity/speed jobs. Never allocate/load/launch anything."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
LOG=Path('/mnt/ai/logs/m31-port')
MODEL='/mnt/ai/models/MiniMax-M3.1-Flash-GGUF/UD-IQ3_S/MiniMax-M3.1-Flash-UD-IQ3_S-00001-of-00005.gguf'

def write(path,value):
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--engine',default=str(ROOT/'build-m31-cuda/m31_serve'))
    p.add_argument('--generate',default=str(ROOT/'build-m31-cuda/m31_generate'))
    p.add_argument('--tag',default='v1')
    a=p.parse_args()
    if not a.tag or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for ch in a.tag):p.error('invalid tag')
    out=a.directory.resolve();out.mkdir(parents=True,exist_ok=False)
    manifest={'preparation_only':True,'speed':[],'limits':[
        'Rebuild all engine/GPU TUs against the same m31_arena.hpp before running.',
        'Do not run while frozen gate20 or another model owns GPU0.',
        'Only the coordinator opens speed windows; exclusive build lock covers each speed session.',
        '170GiB budget packs the full 152.51GiB arena; none is allocated by this helper.',
        'THP arm is guarded, FORCE=0. Inspect server log mapped/advised/AnonHugePages; guard fallback is a baseline result, not a THP speed result.',
        'No forced full-arena THP arm is prepared: direct compaction risk requires a separate deliberate choice.',
        'Reference sampling uses top-level reasoning_effort=low; same om_bench warmup,4K,4K and F16/8192/chunk2048/64threads.',
        'Arena packing startup and measured generation speed are separate; server readiness excludes pack time from client timing.',
    ]}
    for label,budget,thp in [('mmap',0,0),('packed',170,0),('thp-guarded',170,1)]:
        arm=f'm31-arena-{label}-{a.tag}';directory=out/label
        command=[sys.executable,str(ROOT/'tools/m31/server_make_session.py'),
                 '--directory',str(directory),'--arm',arm,'--ctx','8192','--chunk','2048','--threads','64',
                 '--mode','speed4k','--engine',a.engine,'--arena-gib',str(budget),
                 '--cpu-decode-cache','--dense-decode-cache']
        result=subprocess.run(command,check=True,capture_output=True,text=True)
        (directory/'preparation.stdout').write_text(result.stdout)
        job=directory/(arm+'-card-job.json')
        spec=json.loads(job.read_text())
        spec['measurement']=True
        spec['commands'][0]=['/usr/bin/env',f'M31_ARENA_THP={thp}','M31_ARENA_THP_FORCE=0']+spec['commands'][0]
        job.write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n')
        session=directory/(arm+'-http-session.json')
        data=json.loads(session.read_text())
        for client in data['clients']:
            if client.get('name')=='speed':client['env']['OM_OFF']='{"reasoning_effort":"low"}'
        session.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
        manifest['speed'].append({'label':label,'arena_gib':budget,'THP':thp,'FORCE':0,
            'job':str(job),'launch':[sys.executable,str(ROOT/'tools/m31/card_launch.py'),str(job),'--unit',arm],
            'results':str(directory/'speed.tsv')})
    commands=[]
    for label,thp in [('packed',0),('thp-guarded',1)]:
        destination=out/(label+'-code128')
        native=['/usr/bin/env',f'M31_ARENA_THP={thp}','M31_ARENA_THP_FORCE=0',a.generate,
            '-m',MODEL,'--backend','cuda','--ctx','4096','--chunk','128','--threads','64',
            '--reuse-allocations','1','--persistent-cpu-pool','1','--cpu-residual','1',
            '--cpu-decode-cache','1','--dense-decode-cache','1','--arena-gib','170',
            '--cases',str(LOG/'ids-code128-v1'),'--out',str(destination)]
        commands.append(native)
        commands.append([sys.executable,str(ROOT/'tools/m31/compare_logits.py'),str(LOG/'code128-oracle-v1'),
            str(destination),'--out',str(out/(label+'-code128-compare.json'))])
    parity=out/('job-arena-parity-'+a.tag+'.json')
    write(parity,{'measurement':False,'commands':commands})
    manifest['parity']={'job':str(parity),'launch':[sys.executable,str(ROOT/'tools/m31/card_launch.py'),str(parity),'--unit','m31-arena-parity-'+a.tag]}
    write(out/'manifest.json',manifest)
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
