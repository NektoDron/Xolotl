#!/usr/bin/env python3
"""Require exact CPU parity for every stage of the independent expert probe."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('report', type=Path)
p.add_argument('--label', required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
d = json.loads(a.report.read_text())
expected = {(l, n, op) for l in (3, 10, 52) for n in (1, 7, 128) for op in ('gate', 'up', 'down', 'full')}
actual = {(c['layer'], c['tokens'], c['operation']) for c in d['cases']}
if d['format'] != 'm31-expert-check-v1' or actual != expected or len(d['cases']) != 36:
    raise SystemExit('expert probe is incomplete or has unexpected case dimensions')
stages = [r for c in d['cases'] for r in c['results'] if r['mode'] == 'cuda_cpu_quants']
passed = len(stages) == 117 and all(r['metrics']['bitwise'] and r['metrics']['max_abs'] == 0 and
    r['metrics']['nonfinite_reference'] == r['metrics']['nonfinite_candidate'] == 0 for r in stages)
result = {'pass': passed, 'label': a.label, 'cases': len(d['cases']), 'stages': len(stages),
          'f32_values': sum(r['metrics']['count'] for r in stages), 'report': str(a.report)}
a.out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
raise SystemExit(0 if passed else 1)
