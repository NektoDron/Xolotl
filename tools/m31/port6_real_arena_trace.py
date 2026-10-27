#!/usr/bin/env python3
"""Reconcile real-arena full-byte counters with Nsight CUDA memcpy activities.
Every measurement completes all streams before the next starts. Thus physical
H2D activities can be partitioned in time order by exact cumulative byte counts,
without guessing stream IDs or treating a wall-time counter as PCIe bandwidth.
"""
import argparse
import json
from pathlib import Path
import sqlite3


def merged(items):
    out=[]
    for lo,hi in sorted(items):
        if hi<=lo: continue
        if out and lo<=out[-1][1]:out[-1][1]=max(out[-1][1],hi)
        else:out.append([lo,hi])
    return out


def duration(items):return sum(hi-lo for lo,hi in merged(items))


def intersection(a,b):
    a,b=merged(a),merged(b);i=j=0;out=[]
    while i<len(a) and j<len(b):
        lo,hi=max(a[i][0],b[j][0]),min(a[i][1],b[j][1])
        if hi>lo:out.append([lo,hi])
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return duration(out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--database',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();report=json.loads(a.report.read_text())
    con=sqlite3.connect(a.database.resolve().as_uri()+'?mode=ro',uri=True)
    names=dict(con.execute('select id,value from StringIds'))
    enum=[row[0] for row in con.execute('select id,name from ENUM_CUDA_MEMCPY_OPER') if row[1]=='CUDA_MEMCPY_KIND_HTOD']
    if len(enum)!=1:raise SystemExit('unexpected H2D enum')
    copies=list(con.execute('select start,end,bytes,streamId,deviceId,contextId,globalPid from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=? and bytes>0 order by start',(enum[0],)))
    expected=sum(c['bytes'] for c in report['cases']);actual=sum(row[2] for row in copies)
    if actual!=expected:raise SystemExit(f'exact H2D byte audit failed: CUDA{actual} expected{expected}')
    registration=[];unregistration=[]
    for lo,hi,name in con.execute('select start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME'):
        label=names.get(name,'')
        if 'cudaHostRegister' in label:registration.append((lo,hi))
        elif 'cudaHostUnregister' in label:unregistration.append((lo,hi))
    cursor=0;results=[]
    for case in report['cases']:
        group=[];total=0
        while total<case['bytes']:
            row=copies[cursor];cursor+=1;group.append(row);total+=row[2]
        if total!=case['bytes']:raise SystemExit('a CUDA activity crosses a measurement byte boundary')
        intervals=[(x[0],x[1]) for x in group];busy=sum(hi-lo for lo,hi in intervals)
        union=duration(intervals);span=max(x[1] for x in group)-min(x[0] for x in group)
        result=dict(mode=case['mode'],streams=case['streams'],bytes=total,activity_count=len(group),
            stream_ids=sorted({x[3] for x in group}),copy_activity_sum_ns=busy,copy_activity_union_ns=union,
            first_to_last_copy_ns=span,activity_sum_GB_s=total/busy,activity_union_GB_s=total/union,
            first_to_last_GB_s=total/span,whole_wall_seconds=case['wall_seconds'],whole_wall_GB_s=case['wall_GB_s'],
            registration_overlap_ns=intersection(intervals,registration))
        results.append(result)
    result=dict(scope='Actual full ExpertArena original bytes; byte-exact reconciliation of per-case source/DMA counters and CUPTI H2D copy activities. GPU-copy intervals are not inferred from total request wall time.',
        arena_bytes=report['arena_bytes'],total_H2D_bytes=actual,exact_byte_audit=True,cases=results,
        registration_calls=len(registration),registration_API_sum_ns=sum(hi-lo for lo,hi in registration),
        registration_API_union_ns=duration(registration),unregistration_calls=len(unregistration),
        unregistration_API_sum_ns=sum(hi-lo for lo,hi in unregistration),
        pin_bytes=report.get('pin_bytes'),registered_chunks=report.get('registrations'),unregistered_chunks=report.get('unregistered_chunks'))
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
