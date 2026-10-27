#!/usr/bin/env python3
"""Read saved short-prefill results/logs, including incomplete runs. No inference."""
import argparse
import json
from pathlib import Path
import shlex
import statistics
import time

from shortprefill_summarize import summarize


def fields(line):
    result = {}
    for part in shlex.split(line):
        if '=' not in part:
            continue
        key, value = part.split('=', 1)
        try:
            result[key] = float(value) if '.' in value or 'e' in value else int(value)
        except ValueError:
            result[key] = value
    return result


def diagnostic_log(path):
    samples = []
    current = None
    for line in path.read_text(errors='replace').splitlines():
        if line.startswith('M31 shortprefill sample '):
            current = {**fields(line), 'cpu_layers': [], 'hybrid_layers': []}
            samples.append(current)
        elif current is not None and line.startswith('M31 shortprefill_cpu '):
            current['cpu_layers'].append(fields(line))
        elif current is not None and line.startswith('M31 shortprefill layer='):
            current['hybrid_layers'].append(fields(line))
    for sample in samples:
        sample['cpu_stage_sums_ms'] = {key: sum(row.get(key, 0) for row in sample['cpu_layers'])
                                      for key in ('norm_read_ms', 'graph_build_ms', 'host_copy_ms',
                                                  'cpu_compute_ms', 'cpu_read_ms', 'merge_build_ms', 'gpu_combine_ms')}
    return samples


def inspect(root):
    paths, errors = [], []
    for path in sorted(root.glob('*/summary.json')):
        try:
            value = json.loads(path.read_text())
            if value.get('format') == 'm31-shortprefill-result-v1':
                paths.append(path)
        except (OSError, ValueError) as error:
            errors.append({'path': str(path), 'read_error': str(error)})
    aggregate = summarize(paths)
    first_vs_warm = []
    for path in paths:
        value = json.loads(path.read_text())
        groups = {}
        for row in value['runs']:
            groups.setdefault((row['case'], row['arm']), []).append(row)
        for (case, arm), rows in groups.items():
            measured = [r for r in rows if not r.get('warmup', False)]
            if not measured:
                continue
            cold = rows[0]
            first_vs_warm.append({'session': str(path.parent), 'case': case, 'arm': arm,
                                  'first_sample_is_warmup': cold.get('warmup', False),
                                  'first_forward_ms': cold['forward_ms'],
                                  'measured_forward_median_ms': statistics.median(r['forward_ms'] for r in measured),
                                  'first_counters': cold.get('counters', {}),
                                  'measured_counters': [r.get('counters', {}) for r in measured],
                                  'note': 'First sample is shape/route-cache dependent; this is not a new Engine initialization.'})
    logs = {}
    for path in sorted(root.glob('*-[0-9][0-9].log')):
        samples = diagnostic_log(path)
        if samples:
            logs[str(path)] = samples
    return {'format': 'm31-shortprefill-watch-v1', 'aggregate': aggregate,
            'first_vs_measured': first_vs_warm, 'diagnostic_logs': logs, 'read_errors': errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--watch-seconds', type=float, default=0,
                        help='Optional bounded polling; run long watches via systemd.')
    parser.add_argument('--interval', type=float, default=10)
    args = parser.parse_args()
    if args.watch_seconds < 0 or args.interval <= 0:
        parser.error('invalid polling duration')
    start = time.monotonic()
    while True:
        value = inspect(args.root)
        args.out.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps({'sessions': len(value['aggregate']['sessions']),
                          'complete': value['aggregate']['all_sessions_complete'],
                          'rows': len(value['aggregate']['rows']), 'out': str(args.out)}), flush=True)
        remaining = args.watch_seconds-(time.monotonic()-start)
        if remaining <= 0:
            break
        time.sleep(min(args.interval, remaining))


if __name__ == '__main__':
    main()
