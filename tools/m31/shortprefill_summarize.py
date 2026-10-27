#!/usr/bin/env python3
"""Aggregate saved native short-prefill runs; this command never runs inference."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def summarize(paths):
    groups = defaultdict(list)
    receipts = []
    for path in paths:
        value = json.loads(path.read_text())
        if value.get('format') != 'm31-shortprefill-result-v1':
            raise ValueError(f'unknown result format: {path}')
        receipts.append({'path': str(path.resolve()), 'complete': value.get('pass', False),
                         'error': value.get('error')})
        for row in value['runs']:
            if not row.get('warmup', False):
                groups[(str(path.parent.resolve()), row['case'], row['arm'])].append(row)
    rows = []
    for (session, case, arm), samples in groups.items():
        rows.append({'session': session, 'case': case, 'arm': arm, 'samples': len(samples),
                     'pass': all(r.get('pass', False) for r in samples),
                     'tokens': samples[0]['tokens'], 'prefix_tokens': samples[0]['prefix_tokens'],
                     'forward_ms': statistics.median(r['forward_ms'] for r in samples),
                     'native_ttft_ms': statistics.median(r['native_ttft_ms'] for r in samples),
                     'head_ms': statistics.median(r['head_ms'] for r in samples),
                     'tps': statistics.median(r['tokens_per_second'] for r in samples),
                     'gpu_upload_bytes': statistics.median(r.get('counters', {}).get('gpu_upload_bytes', 0) for r in samples),
                     'cpu_demand_bytes': statistics.median(r.get('counters', {}).get('cpu_weight_demand_bytes', 0) for r in samples),
                     'cpu_compute_ms': statistics.median(r.get('counters', {}).get('cpu_compute_ms', 0) for r in samples),
                     'cpu_setup_ms': statistics.median(r.get('counters', {}).get('cpu_setup_ms', 0) for r in samples),
                     'actual_chunks': [[c['tokens'] for c in r['chunks']] for r in samples],
                     'gates': [r['gate'] for r in samples if 'gate' in r]})
    baseline = {(r['session'], r['case']): r for r in rows if r['arm'] == 'off'}
    for row in rows:
        if (row['session'], row['case']) in baseline:
            row['speedup_vs_off'] = baseline[(row['session'], row['case'])]['forward_ms']/row['forward_ms']
    rows.sort(key=lambda r: (r['prefix_tokens'], r['tokens'], r['session'], r['case'], r['arm']))
    winners = [r for r in rows if r['arm'] == 'cpu' and r['pass'] and not r['prefix_tokens']
               and r.get('speedup_vs_off', 0) > 1]
    return {'format': 'm31-shortprefill-aggregate-v1', 'sessions': receipts,
            'all_sessions_complete': bool(receipts) and all(r['complete'] for r in receipts), 'rows': rows,
            'cpu_wins_at_tested_tokens': sorted({r['tokens'] for r in winners}),
            'scope': 'Measured points only. CPU winning at one length does not prove a monotonic threshold. Native TTFT excludes HTTP, media, reasoning and first decode.'}


def markdown(value):
    lines = ['# Адаптивный префилл: сохранённые измерения', '',
             '| Сессия | Случай | Режим | N | Префикс | Образцов | Forward мс | Native TTFT мс | t/s | Ускорение | PCIe ГБ | CPU setup мс | CPU compute мс |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in value['rows']:
        speedup = f"{row['speedup_vs_off']:.3f}" if 'speedup_vs_off' in row else '—'
        lines.append(f"| {Path(row['session']).name} | {row['case']} | {row['arm']} | {row['tokens']} | {row['prefix_tokens']} | {row['samples']} | {row['forward_ms']:.2f} | {row['native_ttft_ms']:.2f} | {row['tps']:.1f} | {speedup} | {row['gpu_upload_bytes']/1e9:.3f} | {row['cpu_setup_ms']:.2f} | {row['cpu_compute_ms']:.2f} |")
    lines += ['', 'Native TTFT включает префилл нового куска и last-row head. HTTP/энкодер/reasoning сюда не входят. CPU demand bytes — логический объём весов, а не фактический RAM-трафик.', '',
              'CPU выиграл только в проверенных точках: '+', '.join(map(str, value['cpu_wins_at_tested_tokens']))+'.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', nargs='+', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    value = summarize(args.summary)
    args.out.with_suffix('.json').write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    args.out.with_suffix('.md').write_text(markdown(value))
    print(json.dumps({'complete': value['all_sessions_complete'], 'rows': len(value['rows']),
                      'cpu_wins_at': value['cpu_wins_at_tested_tokens']}))


if __name__ == '__main__':
    main()
