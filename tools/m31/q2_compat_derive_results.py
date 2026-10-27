#!/usr/bin/env python3
"""Derive exact per-case command receipts from one real completed card job."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--combined-job',type=Path,required=True);p.add_argument('--target-job',type=Path,action='append',required=True);a=p.parse_args()
source=a.combined_job.resolve();source_result=source.with_suffix('.result.json');job=json.loads(source.read_text());result=json.loads(source_result.read_text())
if len(result['commands'])>len(job['commands']):raise ValueError('too many completed commands')
for planned,observed in zip(job['commands'],result['commands']):
 if planned!=observed['argv'] or type(observed['returncode']) is not int:raise ValueError('combined planned/completed argv mismatch')
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
for target in a.target_job:
 target=target.resolve();spec=json.loads(target.read_text())
 if len(spec['commands'])!=1:raise ValueError('only singleton source plans supported')
 selected=[(i,row) for i,row in enumerate(result['commands']) if row['argv']==spec['commands'][0]]
 if len(selected)!=1:raise ValueError('selected actual command missing or ambiguous')
 i,entry=selected[0]
 receipt={'commands':[entry],'returncode':entry['returncode'],'derived_from':{'job':str(source),'job_sha256':sha(source),'result':str(source_result),'result_sha256':sha(source_result),'source_combined_returncode':result['returncode'],'command_index':i,'policy':'Derived exact selected-command receipt from one actual combined card session. This per-case plan was not launched as a separate unit; other source commands may have failed.'}}
 with target.with_suffix('.result.json').open('x')as stream:json.dump(receipt,stream,indent=2)
 print(target.with_suffix('.result.json'))
