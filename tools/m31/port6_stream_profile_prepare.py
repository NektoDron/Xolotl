#!/usr/bin/env python3
"""Freeze a new isolated fast-FFN Nsight card job from an accepted numeric job.

Preparation only; never launches the GPU or mutates the supplied job/artifacts.
The operator supplies the index of its successful large-prompt native command.
"""
import argparse
import copy
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-job',type=Path,required=True)
    p.add_argument('--command',type=int,required=True)
    p.add_argument('--stem',type=Path,required=True)
    a=p.parse_args()
    src=json.loads(a.source_job.read_text())
    result=a.source_job.with_suffix('.result.json')
    receipt=json.loads(result.read_text())
    if receipt.get('returncode')!=0:
        p.error('source numeric job failed; select a fully successful gate receipt')
    if not (0<=a.command<len(src['commands'])):
        p.error('command index outside source job')
    statuses=[r for r in receipt['commands'] if r['argv']==src['commands'][a.command]]
    if not statuses or statuses[-1]['returncode']!=0:
        p.error('selected native command has no successful execution receipt')
    command=copy.deepcopy(src['commands'][a.command])
    if not any(x in command for x in ('--cases','--ids-file')) or '--out' not in command:
        p.error('select a native generate command with cases/out')
    stem=a.stem.resolve()
    if any(stem.with_suffix(ext).exists() for ext in ('.json','.nsys-rep','.sqlite')):
        p.error('stem already used; provide a fresh name')
    command[command.index('--out')+1]=str(stem)+'-native'
    # Ranking launcher also writes one receipt: give it an independent fresh path.
    if '--output' in command:
        command[command.index('--output')+1]=str(stem)+'-preflight.json'
    binary_index=next((i for i,s in enumerate(command) if Path(s).name.startswith('m31_generate')),None)
    if binary_index is None:
        p.error('m31_generate ELF not found')
    # Place nsys immediately before the native ELF, after the ranking preflight.
    prefix=['/usr/local/cuda-13.1/bin/nsys','profile','--trace=cuda,nvtx','--sample=none',
            '--cpuctxsw=none','--cuda-graph-trace=node','--cuda-memory-usage=false','--force-overwrite=false',
            '--output',str(stem)]
    command[binary_index:binary_index]=prefix
    job=dict(measurement=False,preset='m31-xolotl-check',commands=[command,
        ['/usr/local/cuda-13.1/bin/nsys','export','--type','sqlite','--output',str(stem.with_suffix('.sqlite')),str(stem.with_suffix('.nsys-rep'))],
        ['/usr/bin/python3','/mnt/ai/Xolotl-m31/tools/m31/port6_stream_trace.py','--database',str(stem.with_suffix('.sqlite')),'--out',str(stem)+'-overlap.json']])
    with stem.with_suffix('.json').open('x') as f:
        json.dump(job,f,indent=2);f.write('\n')
    print(stem.with_suffix('.json'))


if __name__=='__main__':
    main()
