#!/usr/bin/env python3
"""Compare actual fixed-slot MoE3 diagnostics with immutable M1 oracle taps."""
import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--reference', required=True, type=Path)
p.add_argument('--debug', required=True, type=Path)
p.add_argument('--out', required=True, type=Path)
a = p.parse_args()
checks = [('input', 'ffn_norm', '<f4'), ('original_ids', 'ffn_moe_topk', '<i4'),
          ('gate', 'ffn_moe_gate', '<f4'), ('up', 'ffn_moe_up', '<f4'),
          ('result', 'ffn_moe_out', '<f4')]
report = {'checks': [], 'pass': True}
for candidate, reference, dtype in checks:
    x = np.fromfile(a.reference / 'taps' / f'0_{reference}-3.bin', dtype=dtype)
    y = np.fromfile(a.debug / f'l3_{candidate}.bin', dtype=dtype)
    row = {'stage': candidate, 'reference_elements': int(x.size), 'candidate_elements': int(y.size)}
    row['bitwise'] = x.shape == y.shape and x.tobytes() == y.tobytes()
    if x.shape == y.shape:
        delta = x.astype(np.float64) - y.astype(np.float64)
        row['max_abs'] = float(np.max(np.abs(delta)))
        row['relative_l2'] = float(np.linalg.norm(delta) / max(float(np.linalg.norm(x.astype(np.float64))), 1e-300))
        bad = np.flatnonzero(x != y)
        row['different_elements'] = int(bad.size)
        row['first_difference'] = int(bad[0]) if bad.size else None
    report['checks'].append(row)
    report['pass'] = report['pass'] and row['bitwise']
ids = np.fromfile(a.debug / 'l3_original_ids.bin', dtype='<i4')
unique, counts = np.unique(ids, return_counts=True)
report['expert_counts'] = {str(int(k)): int(v) for k, v in zip(unique, counts)}
report['tiled_experts_ge8'] = [int(k) for k, v in zip(unique, counts) if v >= 8]
a.out.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
raise SystemExit(0 if report['pass'] else 1)
