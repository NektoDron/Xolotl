#!/usr/bin/env python3
"""Measure real FFN-stream/H2D overlap from an isolated Nsight SQLite trace.

The fast executor owns a private CUDA stream. Its named dequant/OAI kernels
identify that stream; GEMMs on the same stream belong to the FFN pipeline.
No speedup is inferred from overlap or from individual DMA duration sums.
"""
import argparse
import bisect
import collections
import json
from pathlib import Path
import sqlite3


def union(intervals):
    result = []
    for lo, hi in sorted(intervals):
        if hi <= lo:
            continue
        if result and lo <= result[-1][1]:
            result[-1][1] = max(hi, result[-1][1])
        else:
            result.append([lo, hi])
    return result


def length(intervals):
    return sum(hi-lo for lo, hi in union(intervals))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--database', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--min-copy-bytes', type=int, default=1024*1024)
    p.add_argument('--require-overlap', action='store_true')
    a = p.parse_args()
    if a.min_copy_bytes < 1:
        p.error('min-copy-bytes must be positive')
    db = a.database.resolve(strict=True)
    con = sqlite3.connect(db.as_uri()+'?mode=ro', uri=True)
    names = dict(con.execute('select id,value from StringIds'))
    h2d = [r[0] for r in con.execute('select id,name from ENUM_CUDA_MEMCPY_OPER') if r[1] == 'CUDA_MEMCPY_KIND_HTOD']
    if len(h2d) != 1:
        raise SystemExit('unexpected H2D enum')
    kernels = []
    expert_streams = set()
    for start, end, dev, ctx, stream, pid, name_id in con.execute(
            'select start,end,deviceId,contextId,streamId,globalPid,demangledName from CUPTI_ACTIVITY_KIND_KERNEL'):
        name = names.get(name_id, '')
        key = (dev, ctx, pid, stream)
        if end > start:
            kernels.append((start, end, key, name))
            if any(n in name for n in ('fast_dequant_weights', 'fast_oai', 'fast_scatter', 'fast_reduce')):
                expert_streams.add(key)
    if not expert_streams:
        raise SystemExit('no fast FFN anchor kernels; refusing to label dense GEMMs as experts')
    selected = sorted((x for x in kernels if x[2] in expert_streams), key=lambda x:x[0])
    starts = [x[0] for x in selected]
    longest = max(x[1]-x[0] for x in selected)
    stats = collections.defaultdict(lambda: dict(bytes=0, copies=0, intervals=[], overlap=[]))
    examples = []
    all_overlap = []
    for lo, hi, dev, ctx, stream, pid, size in con.execute(
            'select start,end,deviceId,contextId,streamId,globalPid,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=? and bytes>=? order by start', (h2d[0], a.min_copy_bytes)):
        if hi <= lo:
            continue
        key = (dev, ctx, pid, stream)
        s = stats[key]
        s['bytes'] += size
        s['copies'] += 1
        s['intervals'].append((lo,hi))
        for kb, ke, kk, name in selected[bisect.bisect_left(starts,lo-longest):bisect.bisect_left(starts,hi)]:
            if kk[:3] != key[:3] or kk[3] == key[3]:
                continue
            first, last = max(lo,kb), min(hi,ke)
            if last <= first:
                continue
            all_overlap.append((first,last))
            s['overlap'].append((first,last))
            if len(examples) < 20:
                examples.append(dict(H2D_stream=stream, expert_stream=kk[3], copy_bytes=size,
                    copy_start_ns=lo, copy_end_ns=hi, kernel_start_ns=kb,
                    kernel_end_ns=ke, kernel=name, overlap_ns=last-first))
    rows=[]
    for key, s in sorted(stats.items()):
        total=sum(hi-lo for lo,hi in s['intervals'])
        span=max(hi for _,hi in s['intervals'])-min(lo for lo,_ in s['intervals'])
        rows.append(dict(device=key[0],context=key[1],process=key[2],stream=key[3],
            bytes=s['bytes'],copies=s['copies'],dma_sum_ns=total,dma_union_ns=length(s['intervals']),
            first_to_last_ns=span,active_dma_GB_s=s['bytes']/total if total else None,
            first_to_last_GB_s=s['bytes']/span if span else None,
            expert_overlap_union_ns=length(s['overlap'])))
    compute=length([(lo,hi) for lo,hi,_,_ in selected]);overlap=length(all_overlap)
    result=dict(scope='Isolated real-model trace: anchored private fast-FFN stream, large H2D on different same-process/device/context streams; overlap is not speedup',
        database=str(db),expert_streams=[dict(device=d,context=c,process=p,stream=s) for d,c,p,s in sorted(expert_streams)],
        expert_kernel_count=len(selected),expert_kernel_union_ns=compute,overlap_union_ns=overlap,
        compute_fraction_overlapped=overlap/compute if compute else 0,H2D_streams=rows,examples=examples)
    with a.out.open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='examples'},indent=2))
    if a.require_overlap and not overlap:
        raise SystemExit('real overlap was not observed')


if __name__=='__main__':
    main()
