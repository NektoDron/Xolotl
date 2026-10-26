#!/usr/bin/env python3
"""Combine the minimal Q6_K+OAI numeric probe and actual Nsight kernel names."""
import argparse
import json
from pathlib import Path
import re
import sqlite3
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--log',type=Path,required=True)
p.add_argument('--database',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
log=a.log.read_text()
m=re.search(r'(?m)^\{\s*"cases"\s*:',log)
if not m:raise SystemExit('missing probe JSON object')
result,_=json.JSONDecoder().raw_decode(log[m.start():])
c=sqlite3.connect(a.database.resolve().as_uri()+'?mode=ro',uri=True)
strings=dict(c.execute('select id,value from StringIds'))
names={}
for start,end,name_id in c.execute('select start,end,demangledName from CUPTI_ACTIVITY_KIND_KERNEL'):
    name=strings.get(name_id,'')
    if 'mul_mat_q<' in name:
        value=names.setdefault(name,dict(calls=0,device_ns=0))
        value['calls']+=1;value['device_ns']+=end-start
c.close()
result['trace_mmqs']=[dict(kernel=name,**value)for name,value in sorted(names.items())]
result['fused_mmq_kernel_observed']=any(re.search(r',\s*true\s*>',name)for name in names)
result['scope']='Original Q6_K gate/up+OAI fusion numeric probe plus actual CUDA kernel templates; NOT a speed benchmark'
with a.out.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2))
