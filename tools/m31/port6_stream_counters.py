#!/usr/bin/env python3
"""Reconcile cumulative transport counters into per-chunk real model deltas.

Host route-read counts are measured. Metadata/embedding/mask H2D counts are
source-accounted for the accepted native REG1/CHAIN1/CAUSAL_FILL0 configuration;
CUDA activity counts and device overlap require an independent CUPTI trace.
"""
import argparse
import json
from pathlib import Path
import re


def values(line):
    return {k: float(v) if '.' in v else int(v) for k,v in
            re.findall(r'(\w+)=(-?\d+(?:\.\d+)?)',line)}


def parse(path):
    previous = {'expert': {}, 'prefetch': {}, 'foreground': {}}
    frames=[]; current=None; startup=[]
    for number,line in enumerate(path.read_text().splitlines(),1):
        if 'registered arena:' in line or 'RAM arena:' in line:
            startup.append({'line':number,'text':line})
        if line.startswith('M31 forward pos='):
            current={'line':number,**values(line)}
        if current is None:
            continue
        for needle,field in [('M31 host-reads ','host_reads'),('M31 memory ','memory'),('M31 stages ','stages')]:
            if line.startswith(needle):current[field]=values(line)
        for needle,field in [('M31 GPU expert profile ','expert'),('M31 GPU prefetch profile ','prefetch'),('M31 GPU upload profile ','foreground')]:
            if line.startswith(needle):
                now=values(line)
                current[field]={k:v-previous[field].get(k,0) for k,v in now.items()}
                previous[field]=now
                if field=='foreground':
                    if current['n']>1:frames.append(current)
                    current=None
    requests=[]
    for frame in frames:
        if frame['pos']==0:requests.append({'chunks':[]})
        n=frame['n'];nk=((frame['pos']+n+255)//256)*256
        measured=frame['host_reads']
        expected=57*4*n*8
        if measured['reads']!=114 or measured['bytes']!=expected:
            raise ValueError(f'{path}:{frame["line"]}: host reads exceed/differ from route metadata envelope')
        if frame['prefetch']['batches']!=56 or frame['prefetch']['experts']!=7168:
            raise ValueError('incomplete full-next-layer prefetch population')
        if frame['expert']['bytes']!=frame['prefetch']['bytes']+frame['foreground']['bytes']:
            raise ValueError('expert bytes do not reconcile')
        frame['route_D2H_measured']={'calls':114,'bytes':expected,'ids_I32_bytes':expected//2,'weights_F32_bytes':expected//2}
        frame['FFN_intermediate_D2H_bytes']=0
        frame['H2D_source_accounting']={'route_ids_weights_calls':114,'route_ids_weights_bytes':expected,
            'external_scale_calls':3,'external_scale_bytes':3*290*4,
            'embedding_seed_calls':1,'embedding_seed_bytes':6144*n*4,
            'attention_input_calls':4,'attention_input_bytes':4*(n+nk*n+(nk//128)*n+nk)}
        requests[-1]['chunks'].append(frame)
    for request in requests:
        chunks=request['chunks']
        request.update(prefill_tokens=sum(c['n'] for c in chunks),chunk_count=len(chunks),
            prefetch_batches=sum(c['prefetch']['batches'] for c in chunks),
            prefetched_experts=sum(c['prefetch']['experts'] for c in chunks),
            prefetch_bytes=sum(c['prefetch']['bytes'] for c in chunks),
            foreground_bytes=sum(c['foreground']['bytes'] for c in chunks),
            total_expert_bytes=sum(c['expert']['bytes'] for c in chunks),
            measured_route_D2H_bytes=sum(c['route_D2H_measured']['bytes'] for c in chunks),
            measured_route_D2H_calls=114*len(chunks),FFN_intermediate_D2H_bytes=0)
    return {'log':str(path),'startup':startup,'requests':requests}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log',type=Path,action='append',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    result={'scope':'Real log counter deltas; prefetch submissions/completions are not measured device overlap; host-read counter excludes logits/head reads',
            'native_geometry':{'routed_layers':57,'experts':128,'used':4,'embd':6144,'slots':290,'scaled_role_tensors':3},
            'inputs':[parse(path) for path in a.log]}
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{'log':v['log'],'requests':[{k:x for k,x in r.items() if k!='chunks'} for r in v['requests']]} for v in result['inputs']],indent=2))


if __name__=='__main__':main()
