#!/usr/bin/env python3
"""Prove large H2D DMA overlaps exact expert CUDA kernels in an Nsight trace.

Intervals alone do not measure throughput or prove a speedup. With this engine,
main routed-weight uploads and synchronous quant H2D finish before expert
kernels. Large copies on another stream during such kernels therefore identify
next-layer background prefetch; require an isolated native-only model session.
"""
import argparse
import bisect
import json
from pathlib import Path
import sqlite3


def union_length(intervals):
    if not intervals:
        return 0
    intervals.sort()
    total = 0
    first, last = intervals[0]
    for begin, end in intervals[1:]:
        if begin > last:
            total += last-first
            first, last = begin, end
        else:
            last = max(last, end)
    return total+last-first


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--database', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--min-copy-bytes', type=int, default=1024*1024)
    a = p.parse_args()
    if a.min_copy_bytes < 1:
        p.error('min-copy-bytes must be positive')
    database = a.database.resolve(strict=True)
    con = sqlite3.connect(database.as_uri()+'?mode=ro', uri=True)
    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    required = {'StringIds', 'CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_KERNEL', 'ENUM_CUDA_MEMCPY_OPER'}
    if not required <= tables:
        raise SystemExit('missing trace tables: '+str(sorted(required-tables)))
    names = dict(con.execute('select id,value from StringIds'))
    h2d_ids = [r[0] for r in con.execute('select id,name from ENUM_CUDA_MEMCPY_OPER') if r[1] == 'CUDA_MEMCPY_KIND_HTOD']
    if len(h2d_ids) != 1:
        raise SystemExit('missing/unexpected H2D enum')
    kernels = []
    for start, end, device, context, stream, process, name_id in con.execute(
            'select start,end,deviceId,contextId,streamId,globalPid,demangledName from CUPTI_ACTIVITY_KIND_KERNEL'):
        name = names.get(name_id, '')
        if end > start and any(part in name for part in ('multiply_batched_kernel<', 'multiply_tensor_kernel<', 'multiply_kernel<')):
            kernels.append((start, end, device, context, stream, process, name))
    kernels.sort(key=lambda value: value[0])
    starts = [k[0] for k in kernels]
    longest = max((end-start for start,end,*_ in kernels), default=0)
    examples, intersections = [], []
    copies = 0
    overlapping = 0
    for first, last, device, context, stream, process, size in con.execute(
            'select start,end,deviceId,contextId,streamId,globalPid,bytes from CUPTI_ACTIVITY_KIND_MEMCPY '
            'where copyKind=? and bytes>=? order by start', (h2d_ids[0], a.min_copy_bytes)):
        if last <= first:
            continue
        copies += 1
        found = False
        # Bound the scan by the longest expert kernel rather than comparing every
        # copy with every kernel. Same device/context and different streams only.
        for begin, end, kd, kc, ks, kp, name in kernels[bisect.bisect_left(starts, first-longest):bisect.bisect_left(starts, last)]:
            if kd != device or kc != context or kp != process or ks == stream:
                continue
            lo, hi = max(first, begin), min(last, end)
            if lo >= hi:
                continue
            found = True
            intersections.append((lo, hi))
            if len(examples) < 12:
                examples.append(dict(copy_start_ns=first, copy_end_ns=last,
                    copy_bytes=size, copy_stream=stream, kernel_start_ns=begin,
                    kernel_end_ns=end, kernel_stream=ks, kernel=name,
                    device=device, context=context, process=process, intersection_ns=hi-lo))
        overlapping += int(found)
    con.close()
    result = dict(scope='Same-process/device/context large H2D DMA and exact expert kernels on different streams; event proof, NOT a speed benchmark',
        database=str(database), min_copy_bytes=a.min_copy_bytes,
        exact_expert_kernels=len(kernels), large_h2d_copies=copies,
        h2d_copies_with_expert_overlap=overlapping,
        observed_device_copy_compute_overlap=bool(overlapping),
        overlap_wall_ns_union=union_length(intersections), examples=examples)
    with a.out.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'examples'}, indent=2))
    if not kernels or not overlapping:
        raise SystemExit('overlap was not observed')


if __name__ == '__main__':
    main()
