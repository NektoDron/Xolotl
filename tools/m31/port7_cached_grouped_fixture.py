#!/usr/bin/env python3
"""Freeze actual held-out cached assignments for same-expert grouped dot A/B."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
from port7_decode_rank import load
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();base=Path('/mnt/ai/logs/m31-port');rp=base/'port7-actual-decode-rank-v1/rank.json';rank=json.loads(rp.read_text());trace=Path(rank['source_trace']);h,docs,routes=load(trace);hot={(r['layer'],r['expert']) for r in rank['records'][:290]};cases=[]
for layer in [3,10,52]:
 for n in range(1,9):
  eligible=[]
  for name in rank['held_cases']:
   rows=1+max(t for t,l in routes[name]);
   for first in range(0,rows-n+1,n):
    hit=[(tok,rank_,expert) for tok in range(n) for rank_,expert in enumerate(routes[name][first+tok,layer]['ids']) if (layer,expert) in hot]
    if hit:eligible.append((len(hit),len({e for t,r,e in hit}),name,first,hit))
  if not eligible:continue
  eligible.sort();target=eligible[len(eligible)//2];H,G,name,first,hit=target
  rows=[routes[name][first+t,layer] for t in range(n)]
  cases.append(dict(name=f'L{layer}-N{n}-held-median',layer=layer,n=n,case=name,first=first,full_ids=[e for r in rows for e in r['ids']],full_weights=[w for r in rows for w in r['weights']],hits=[dict(token=t,rank=r,expert=e) for t,r,e in hit],H=H,G=G))
 # One real window with strongest repeat reuse, not synthetic all-hot.
 n=8;eligible=[]
 for name in rank['held_cases']:
  rows=1+max(t for t,l in routes[name]);
  for first in range(0,rows-n+1,n):
   hit=[(tok,rank_,expert) for tok in range(n) for rank_,expert in enumerate(routes[name][first+tok,layer]['ids']) if (layer,expert) in hot];g=len({e for t,r,e in hit})
   if hit and g:eligible.append((len(hit)/g,len(hit),-g,name,first,hit))
 if eligible:
  _,H,negG,name,first,hit=max(eligible);rows=[routes[name][first+t,layer] for t in range(n)];cases.append(dict(name=f'L{layer}-N8-held-maxreuse',layer=layer,n=n,case=name,first=first,full_ids=[e for r in rows for e in r['ids']],full_weights=[w for r in rows for w in r['weights']],hits=[dict(token=t,rank=r,expert=e) for t,r,e in hit],H=H,G=-negG))
report=dict(format='m31-port7-cached-grouped-actual-fixture-v1',rank_sha256=hashlib.sha256(rp.read_bytes()).hexdigest(),trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),model=rank['identity']['geometry']['model'],hot=290,cases=cases,scope='Actual held greedy routes; activation values from earlier real L3 x fixture, not true per-case/per-layer hidden states; benchmark helper assignments, not model output')
with a.out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps([(c['name'],c['H'],c['G']) for c in cases]))
