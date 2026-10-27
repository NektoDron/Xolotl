#!/usr/bin/env python3
"""Compare existing L3 FFN debug dumps; no GPU/model work, diagnostic scope only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def metric(reference, candidate, dtype):
    a = np.fromfile(reference, dtype=dtype)
    b = np.fromfile(candidate, dtype=dtype)
    if a.shape != b.shape or not a.size:
        raise ValueError('L3 dump shapes differ or are empty')
    x, y = a.astype(np.float64), b.astype(np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('non-finite L3 debug values')
    difference = x - y
    norm = float(np.linalg.norm(x))
    indices = np.flatnonzero(a != b)
    return {'values': int(a.size), 'byte_equal': reference.read_bytes() == candidate.read_bytes(),
            'reference_sha256': hashlib.sha256(reference.read_bytes()).hexdigest(),
            'candidate_sha256': hashlib.sha256(candidate.read_bytes()).hexdigest(),
            'different_values': int(indices.size), 'first_difference': int(indices[0]) if len(indices) else None,
            'max_abs': float(np.abs(difference).max()), 'rms': float(np.sqrt(np.mean(difference ** 2))),
            'relative_l2': float(np.linalg.norm(difference)) / max(norm, 1e-30),
            'reference_l2': norm, 'candidate_l2': float(np.linalg.norm(y)),
            'examples': [{'index': int(i), 'reference': float(x[i]), 'candidate': float(y[i])} for i in indices[:8]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('exact', type=Path)
    parser.add_argument('fast', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = {'format': 'm31-port6-l3-dump-diagnostic-v1', 'scope': 'diagnostic-only',
              'exact': str(args.exact), 'fast': str(args.fast), 'arrays': {}}
    for name, dtype in (('input', '<f4'), ('original_ids', '<i4'), ('mapped_ids', '<i4'),
                        ('route_weights', '<f4'), ('result', '<f4')):
        result['arrays'][name] = metric(args.exact / ('l3_' + name + '.bin'),
                                       args.fast / ('l3_fast_' + name + '.bin'), dtype)
    result['equivalent_inputs'] = all(result['arrays'][name]['byte_equal'] for name in ('input', 'original_ids', 'route_weights'))
    with args.out.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
