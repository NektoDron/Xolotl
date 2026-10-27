#!/usr/bin/env python3
"""Per-position numeric diagnostics; this is never a substitute for final gate20."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import compare_logits


def summarize(rows):
    return {'rows': len(rows), 'first_position': rows[0]['position'], 'last_position': rows[-1]['position'],
            'top1_matches': sum(row['top1_same'] for row in rows),
            'top1': sum(row['top1_same'] for row in rows) / len(rows),
            'mean_kl': sum(row['kl'] for row in rows) / len(rows),
            'max_kl': max(row['kl'] for row in rows),
            'first_mismatch': next((row['position'] for row in rows if not row['top1_same']), None),
            'first_kl_gt_1e_3': next((row['position'] for row in rows if row['kl'] > .001), None),
            'first_kl_gt_0_05': next((row['position'] for row in rows if row['kl'] > .05), None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--decode-tail', type=int, default=32)
    parser.add_argument('--bin', type=int, default=32)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.decode_tail < 0 or args.bin < 1:
        parser.error('invalid tail/bin')
    # Reuse strict ID/geometry/finite checks, with bounded memory.
    whole = compare_logits.compare_case(args.reference, args.candidate)
    manifest, reference = compare_logits.load_case(args.reference)
    _, candidate = compare_logits.load_case(args.candidate)
    rows = []
    for start in range(0, len(reference), 8):
        x = np.asarray(reference[start:start + 8], dtype=np.float64)
        y = np.asarray(candidate[start:start + 8], dtype=np.float64)
        topx, topy = x.argmax(1), y.argmax(1)
        top2 = np.partition(x, -2, axis=1)[:, -2:]
        margins = top2[:, 1] - top2[:, 0]
        max_abs = np.abs(x - y).max(1)
        x -= x.max(1, keepdims=True); y -= y.max(1, keepdims=True)
        x -= np.log(np.exp(x).sum(1, keepdims=True)); y -= np.log(np.exp(y).sum(1, keepdims=True))
        kl = np.maximum((np.exp(x) * (x - y)).sum(1), 0)
        for offset in range(len(x)):
            rows.append({'position': manifest['positions'][start + offset], 'reference_top1': int(topx[offset]),
                         'candidate_top1': int(topy[offset]), 'top1_same': bool(topx[offset] == topy[offset]),
                         'kl': float(kl[offset]), 'max_abs': float(max_abs[offset]),
                         'reference_top1_margin': float(margins[offset]),
                         'reference_top1_probability': float(np.exp(x[offset, topx[offset]]))})
    boundary = manifest['tokens'] - args.decode_tail
    groups = {'prefill': [row for row in rows if row['position'] < boundary],
              'decode': [row for row in rows if row['position'] >= boundary]}
    result = {'format': 'm31-port6-per-position-diagnostic-v1', 'scope': 'diagnostic-only',
              'reference': str(args.reference), 'candidate': str(args.candidate), 'whole': whole,
              'decode_start': boundary, 'phases': {k: summarize(v) for k, v in groups.items() if v},
              'bins': [summarize(rows[start:start + args.bin]) for start in range(0, len(rows), args.bin)],
              'positions': rows, 'first_mismatch': next((row['position'] for row in rows if not row['top1_same']), None),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with args.out.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('positions', 'whole')}, indent=2))


if __name__ == '__main__':
    main()
