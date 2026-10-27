#!/usr/bin/env python3
"""Freeze real n=1 route sequences for a CPU-only FFN component replay."""
import argparse,hashlib,json,pathlib,struct
p=argparse.ArgumentParser();p.add_argument('--trace',required=True);p.add_argument('--out',required=True);a=p.parse_args()
source=pathlib.Path(a.trace);out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=False)
layers=[3,10,52,58,59];rows={}
for line in source.open():
    r=json.loads(line)
    if r.get('kind')=='routes' and r.get('phase')=='decode' and r.get('n')==1 and r['layer'] in layers:
        key=(r['case'],r['layer']);rows.setdefault(key,{})
        i=r['generated_index'];assert i not in rows[key];assert len(r['ids'])==len(r['weights'])==4 and len(set(r['ids']))==4
        rows[key][i]=r
cases=sorted({c for c,l in rows});assert len(cases)==8
records=[]
for case_index,case in enumerate(cases):
    counts=[]
    for layer in layers:
        data=rows[(case,layer)];count=len(data);assert sorted(data)==list(range(count));counts.append(count)
        for step,r in sorted(data.items()):records.append((case_index,step,layer,*r['ids'],*r['weights']))
    assert len(set(counts))==1
payload=out/'routes.bin'
with payload.open('wb') as f:
    f.write(b'M31RPL01');f.write(struct.pack('<I',len(records)))
    for record in records:f.write(struct.pack('<III4i4f',*record))
manifest={'format':'m31-cpu-actual-route-replay-input-v1','source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'payload':str(payload),'payload_sha256':hashlib.sha256(payload.read_bytes()).hexdigest(),'record_count':len(records),'cases':cases,'layers':layers,'used':4,'case_token_rows':{case:len(rows[(case,layers[0])]) for case in cases},'scope':'Actual decode IDs and F32 route weights; activation vectors in the component replay are synthetic and are identical in serial and grouped arms; no model TPS claim.'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
