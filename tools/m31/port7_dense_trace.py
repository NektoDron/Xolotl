#!/usr/bin/env python3
"""Separate actual GPU work and DMA from host CUDA API waits in decode.

Uses completed logits D2H as token boundaries. This is a profiler diagnostic,
not an om_bench speed result. Both raw API and GPU activity intervals survive.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import statistics


def merge(intervals):
    result = []
    for lo, hi in sorted(intervals):
        if hi <= lo:
            continue
        if result and lo <= result[-1][1]:
            result[-1][1] = max(result[-1][1], hi)
        else:
            result.append([lo, hi])
    return result


def union_ns(intervals):
    return sum(hi - lo for lo, hi in merge(intervals))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--database', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--logits-bytes', type=int, default=800256)
    p.add_argument('--last-steps', type=int, default=128)
    p.add_argument('--skip-first', type=int, default=4)
    args = p.parse_args()
    con = sqlite3.connect(args.database.resolve().as_uri() + '?mode=ro', uri=True)
    names = dict(con.execute('select id,value from StringIds'))
    copies = [dict(zip(('start', 'end', 'bytes', 'kind', 'stream', 'correlation'), row))
              for row in con.execute('select start,end,bytes,copyKind,streamId,correlationId from CUPTI_ACTIVITY_KIND_MEMCPY order by start')]
    dtoh = next(row[0] for row in con.execute('select id,name from ENUM_CUDA_MEMCPY_OPER') if row[1] == 'CUDA_MEMCPY_KIND_DTOH')
    logits = [c for c in copies if c['bytes'] == args.logits_bytes and c['kind'] == dtoh]
    if len(logits) < args.last_steps + 1:
        raise RuntimeError('not enough logits boundaries for requested decode window')
    ends = [c['end'] for c in logits[-(args.last_steps + 1):]]
    runtime = [(lo, hi, names.get(name, str(name)), correlation)
               for lo, hi, name, correlation in con.execute('select start,end,nameId,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME order by start')]
    kernels = [(lo, hi, names.get(name, str(name))) for lo, hi, name in con.execute(
        'select start,end,shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start')]
    steps = []
    for index, (lo, hi) in enumerate(zip(ends, ends[1:])):
        selected_kernel = [(max(lo, a), min(hi, b), name) for a, b, name in kernels if a < hi and b > lo]
        selected_copy = [c for c in copies if lo <= c['start'] < hi]
        selected_runtime = [(max(lo, a), min(hi, b), name, correlation) for a, b, name, correlation in runtime if a < hi and b > lo]
        api = defaultdict(lambda: [0, 0])
        runtime_by_correlation = {}
        for a, b, name, correlation in selected_runtime:
            api[name][0] += 1
            api[name][1] += b - a
            runtime_by_correlation[correlation] = (a, b, name)
        roles = defaultdict(lambda: {'copies': 0, 'bytes': 0, 'dma_ns': 0, 'api_ns': 0})
        for c in selected_copy:
            role = f"kind{c['kind']}_bytes{c['bytes']}"
            data = roles[role]
            data['copies'] += 1
            data['bytes'] += c['bytes']
            data['dma_ns'] += c['end'] - c['start']
            call = runtime_by_correlation.get(c['correlation'])
            if call:
                data['api_ns'] += call[1] - call[0]
        kernel_names = defaultdict(lambda: [0, 0])
        for a, b, name in selected_kernel:
            kernel_names[name][0] += 1
            kernel_names[name][1] += b - a
        kernel_union = union_ns((a, b) for a, b, _ in selected_kernel)
        dma_union = union_ns((max(lo, c['start']), min(hi, c['end'])) for c in selected_copy)
        busy_union = union_ns([(a, b) for a, b, _ in selected_kernel] +
                              [(max(lo, c['start']), min(hi, c['end'])) for c in selected_copy])
        steps.append(dict(index=index, start_ns=lo, end_ns=hi, wall_ns=hi-lo,
                          kernel_count=len(selected_kernel), kernel_union_ns=kernel_union,
                          dma_union_ns=dma_union, device_busy_union_ns=busy_union,
                          device_idle_ns=hi-lo-busy_union, api=dict(api),
                          copy_roles=dict(roles), kernel_names=dict(kernel_names)))
    measured = steps[args.skip_first:]
    if not measured:
        raise RuntimeError('empty measured window')
    keys = ('wall_ns', 'kernel_count', 'kernel_union_ns', 'dma_union_ns', 'device_busy_union_ns', 'device_idle_ns')
    summary = {key: {'mean': statistics.mean(row[key] for row in measured),
                     'median': statistics.median(row[key] for row in measured)} for key in keys}
    api_totals = defaultdict(lambda: [0, 0])
    role_totals = defaultdict(lambda: {'copies': 0, 'bytes': 0, 'dma_ns': 0, 'api_ns': 0})
    kernel_totals = defaultdict(lambda: [0, 0])
    for row in measured:
        for name, value in row['api'].items():
            api_totals[name][0] += value[0]
            api_totals[name][1] += value[1]
        for role, values in row['copy_roles'].items():
            for key, value in values.items():
                role_totals[role][key] += value
        for name, value in row['kernel_names'].items():
            kernel_totals[name][0] += value[0]
            kernel_totals[name][1] += value[1]
    result = dict(format='m31-port7-decode-cu-trace-v1', database=str(args.database),
                  scope='Actual profiler token windows. GPU kernel intervals are work, async compute_ms is CPU enqueue only. Device idle also includes routed CPU work, so it is not wholly transfer overhead.',
                  steps=len(steps), measured_steps=len(measured), skip_first=args.skip_first,
                  summary=summary, api_totals=dict(api_totals), copy_role_totals=dict(role_totals),
                  kernel_totals=dict(kernel_totals), raw_steps=steps)
    with args.out.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'steps': len(steps), 'measured': len(measured), 'summary': summary}, indent=2))


if __name__ == '__main__':
    main()
