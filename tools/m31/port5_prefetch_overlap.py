#!/usr/bin/env python3
"""Verify routed CUDA kernels overlap next-layer H2D in an Nsight SQLite trace."""
import argparse
import json
from pathlib import Path
import sqlite3


def union(intervals):
    result=[]
    for start,end in sorted(intervals):
        if result and start<=result[-1][1]: result[-1][1]=max(result[-1][1],end)
        else: result.append([start,end])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('sqlite',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args()
    with sqlite3.connect(f'file:{a.sqlite}?mode=ro',uri=True) as db:
        kernels=db.execute('''SELECT k.start,k.end,k.streamId,s.value
          FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id
          ORDER BY k.start''').fetchall()
        transfers=db.execute('''SELECT start,end,streamId,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY
          WHERE copyKind=1 AND deviceId=0 ORDER BY start''').fetchall()
    routed=[k for k in kernels if 'multiply_' in k[3] and '_kernel' in k[3] and 'm31' in k[3]]
    if not routed: raise RuntimeError('trace has no M31 routed CUDA kernels')
    kernel_streams={k[2] for k in kernels}
    stream_bytes={}
    for start,end,stream,size in transfers:
        if stream not in kernel_streams: stream_bytes[stream]=stream_bytes.get(stream,0)+size
    if not stream_bytes: raise RuntimeError('trace has no separate upload-only streams')
    # For this explicit rank64 large gate, background prefetch transfers ~80GB;
    # its private transfer stream is the largest upload-only stream. Main misses
    # transfer ~71GB, while model initialization is much smaller.
    background=max(stream_bytes,key=stream_bytes.get)
    copied=[t for t in transfers if t[2]==background]
    spans=union((k[0],k[1]) for k in routed)
    overlaps=[];examples=[];i=0;matched=0
    for start,end,stream,size in copied:
        while i<len(spans) and spans[i][1]<=start:i+=1
        j=i;hit=False
        while j<len(spans) and spans[j][0]<end:
            lo,hi=max(start,spans[j][0]),min(end,spans[j][1])
            if lo<hi:
                hit=True;overlaps.append((lo,hi))
                if len(examples)<5:examples.append({'copy_start':start,'copy_end':end,'bytes':size,'overlap_start':lo,'overlap_end':hi})
            j+=1
        matched+=hit
    merged=union(overlaps)
    report={'scope':'GPU event interval overlap; not an om_bench speed measurement',
            'sqlite':str(a.sqlite),'background_stream_selection':'largest H2D byte volume among streams with no kernels',
            'upload_only_stream_bytes':stream_bytes,'background_stream':background,
            'background_transfer_count':len(copied),'background_transfer_bytes':sum(t[3] for t in copied),
            'routed_kernel_count':len(routed),'routed_kernel_streams':sorted({k[2] for k in routed}),
            'overlapped_transfers':matched,'overlap_union_ns':sum(end-start for start,end in merged),
            'examples':examples,'pass':bool(matched and merged)}
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0 if report['pass'] else 1


if __name__=='__main__':raise SystemExit(main())
