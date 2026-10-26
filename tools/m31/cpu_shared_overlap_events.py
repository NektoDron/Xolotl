#!/usr/bin/env python3
"""Read Nsight SQLite activity: prove GPU work intersects the CPU MoE NVTX range.

This reports event intervals only. Profiling overhead makes this unsuitable as
an inference throughput benchmark.
"""
import argparse
import json
from pathlib import Path
import sqlite3

p=argparse.ArgumentParser()
p.add_argument('--database',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
con=sqlite3.connect('file:'+str(a.database.resolve())+'?mode=ro',uri=True)
tables={r[0] for r in con.execute("select name from sqlite_master where type='table'")}
if 'NVTX_EVENTS' not in tables:raise SystemExit('missing NVTX events')
cols={r[1] for r in con.execute('pragma table_info(NVTX_EVENTS)')}
strings={r[0]:r[1] for r in con.execute('select id,value from StringIds')} if 'StringIds' in tables else {}
text_column='text' if 'text' in cols else 'NULL'
id_column='textId' if 'textId' in cols else 'NULL'
ranges=[]
for start,end,text,text_id in con.execute(f'select start,end,{text_column},{id_column} from NVTX_EVENTS'):
    label=text or strings.get(text_id,'')
    if end is not None and end>start and label=='M31 CPU MoE while shared FFN is queued':ranges.append((start,end))
activities=[]
for table in ('CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_CUDA_GRAPH'):
    if table not in tables:continue
    gpu_cols={r[1] for r in con.execute(f'pragma table_info({table})')}
    if not {'start','end'}<=gpu_cols:continue
    name='demangledName' if 'demangledName' in gpu_cols else ('shortName' if 'shortName' in gpu_cols else 'NULL')
    for start,end,name_id in con.execute(f'select start,end,{name} from {table}'):
        activities.append((start,end,strings.get(name_id,str(name_id)),table))
examples=[];intersected=0;total=0
for cpu_start,cpu_end in ranges:
    intervals=[]
    for gpu_start,gpu_end,name,table in activities:
        first=max(cpu_start,gpu_start);last=min(cpu_end,gpu_end)
        if first<last:
            intervals.append((first,last))
            if len(examples)<12:examples.append(dict(cpu_start_ns=cpu_start,cpu_end_ns=cpu_end,
                gpu_start_ns=gpu_start,gpu_end_ns=gpu_end,intersection_ns=last-first,kernel=name,table=table))
    if intervals:
        intersected+=1
        intervals.sort();first,last=intervals[0]
        for begin,end in intervals[1:]:
            if begin<=last:last=max(last,end)
            else:total+=last-first;first,last=begin,end
        total+=last-first
result={'scope':'GPU activity intersection with CPU MoE NVTX interval; NOT a speed benchmark',
        'database':str(a.database.resolve()),'cpu_ranges':len(ranges),'gpu_activities':len(activities),
        'ranges_with_gpu_activity':intersected,'intersection_ns_union_per_cpu_range':total,
        'examples':examples,'observed_device_cpu_overlap':bool(intersected)}
with a.out.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
print(json.dumps({k:v for k,v in result.items() if k!='examples'},indent=2))
if not ranges or not activities:raise SystemExit('proof incomplete: missing CPU ranges or GPU activities')
