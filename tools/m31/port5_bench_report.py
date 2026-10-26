#!/usr/bin/env python3
"""Summarize a prepared speed session, retaining per-request stage/telemetry evidence."""
import argparse
import collections
import json
import math
from pathlib import Path
import re
import statistics

import bench_m31


def stages(path):
    requests, raw, current = [], [], []
    last_forward_n = None
    latest_counters, previous_counters, raw_counters = {}, {}, {}
    count_fields = {'calls', 'kernel_timed_calls', 'forwards', 'cached', 'hits', 'misses', 'bytes',
                    'batches', 'experts', 'serial_jobs', 'parallel_jobs', 'worker_ranges', 'fallback_calls', 'growths'}
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            if line.startswith('M31 request timings '):
                timings = json.loads(line.removeprefix('M31 request timings '))
                grouped = collections.defaultdict(list)
                for category, phase, fields in current:
                    grouped[phase, category].append(fields)
                summaries = {}
                for (phase, category), records in grouped.items():
                    totals = collections.defaultdict(float)
                    for record in records:
                        for key, value in record.items():
                            if key.endswith('_ms') or key in ('seconds', 'bytes', 'reads', 'reuses', 'growths'):
                                totals[key] += value
                    summaries.setdefault(phase, {})[category] = {
                        'calls': len(records), 'sum': dict(totals),
                        'mean_per_call': {key: value / len(records) for key, value in totals.items()}}
                snapshot = {key: dict(value) for key, value in latest_counters.items()}
                deltas = {category: {key: value - previous_counters.get(category, {}).get(key, 0)
                                    for key, value in values.items() if key.endswith('_ms') or key in count_fields}
                          for category, values in snapshot.items()}
                requests.append({'line': number, 'timings': timings, 'stages': summaries,
                                 'gpu_counters_cumulative': snapshot, 'gpu_counter_deltas': deltas,
                                 'prefill_chunks': [fields for category, phase, fields in current
                                                    if category == 'forward' and phase == 'prefill'],
                                 'decode_forward_count': sum(category == 'forward' and phase == 'decode'
                                                             for category, phase, fields in current)})
                current, last_forward_n = [], None
                previous_counters = snapshot
                continue
            profile = re.match(r'M31 (CPU-quants profile|CPU-quants host profile|CPU-quants pinned profile|GPU expert profile|GPU prefetch profile|GPU upload profile) (.*)', line)
            if profile:
                category, values = profile.groups()
                latest_counters[category] = {key: float(value) for key, value in re.findall(r'(\w+)=(-?\d+(?:\.\d+)?)', values)}
                raw_counters[category] = {'line': number, 'text': line.rstrip()}
                continue
            match = re.match(r'M31 (forward|stages|dense-stages|shared-stages|host-reads|head) (.*)', line)
            if match:
                category, values = match.groups()
                fields = {key: float(value) for key, value in re.findall(r'(\w+)=(-?\d+(?:\.\d+)?)', values)}
                if category == 'forward':
                    last_forward_n = fields.get('n')
                n = last_forward_n if category == 'head' else fields.get('n')
                phase = 'decode' if n == 1 else 'prefill'
                current.append((category, phase, fields))
            elif line.startswith(('M31 expert cache:', 'M31 GPU', 'M31 CPU', 'm31 gpu', 'm31 cpu')):
                raw.append({'line': number, 'text': line.rstrip()})
    return {'requests': requests, 'final_counters': sorted(raw + list(raw_counters.values()), key=lambda item: item['line']),
            'unassigned_stage_lines': len(current),
            'limits': 'Stage intervals can overlap; summing dense/shared/CPU totals is not a wall-time decomposition.'}


def telemetry(path):
    gpu, rss = [], []
    metrics = collections.defaultdict(list)
    with path.open() as stream:
        for line in stream:
            value = json.loads(line)
            if str(value.get('gpu0_used_mib', '')).isdigit():
                gpu.append(int(value['gpu0_used_mib']))
            for name, raw in value.get('gpu0', {}).items():
                try:
                    number = float(raw)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(number):
                    metrics[name].append(number)
            total = 0
            for entry in value.get('rss', {}).values():
                match = re.match(r'VmRSS:\s*(\d+)\s+kB', entry)
                if match:
                    total += int(match.group(1))
            rss.append(total)
    return {'gpu0_peak_mib': max(gpu, default=None), 'descendant_sum_rss_peak_kib': max(rss, default=None),
            'gpu0_ranges': {name: {'min': min(values), 'median': statistics.median(values), 'max': max(values)}
                            for name, values in metrics.items()},
            'samples': len(rss), 'rss_scope': 'Sum of per-process VmRSS, including file mappings; not unique physical memory.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.directory / 'manifest.json').read_text())
    session_paths = list(args.directory.glob('*-http-session.json'))
    if len(session_paths) != 1:
        parser.error('expected exactly one session')
    prefix = session_paths[0].with_suffix('')
    result_path = Path(str(prefix) + '.result.json')
    result = {'format': 'm31-port5-speed-report-v1', 'manifest': manifest,
              'speed': bench_m31.summarize(argparse.Namespace(speed=[args.directory / 'speed.tsv'], quality=[])),
              'profile': stages(Path(str(prefix) + '.server.log')),
              'telemetry': telemetry(Path(str(prefix) + '.telemetry.jsonl')),
              'session_result': json.loads(result_path.read_text()) if result_path.exists() else None}
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(args.out), 'arms': result['speed']['arms'], 'telemetry': result['telemetry'],
                      'requests': len(result['profile']['requests'])}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
