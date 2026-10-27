#!/usr/bin/env python3
"""Compare layer debug dumps and routing histograms; diagnostic, no model/GPU work."""
import argparse
import json
from pathlib import Path

import numpy as np
from port6_validate_l3 import metric


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('exact', type=Path)
    parser.add_argument('fast', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = {'format': 'm31-port6-all-expert-layers-diagnostic-v1', 'scope': 'diagnostic-only',
              'exact': str(args.exact), 'fast': str(args.fast), 'layers': []}
    for layer in range(3, 60):
        arrays = {}
        for name, dtype in (('input', '<f4'), ('original_ids', '<i4'), ('mapped_ids', '<i4'),
                            ('route_weights', '<f4'), ('result', '<f4')):
            arrays[name] = metric(args.exact / f'l{layer}_{name}.bin', args.fast / f'l{layer}_fast_{name}.bin', dtype)
        original = np.fromfile(args.exact / f'l{layer}_original_ids.bin', dtype='<i4')
        candidate = np.fromfile(args.fast / f'l{layer}_fast_original_ids.bin', dtype='<i4')
        if original.min() < 0 or candidate.min() < 0 or original.max() >= 128 or candidate.max() >= 128:
            raise ValueError('invalid expert IDs')
        counts_x = np.bincount(original, minlength=128)
        counts_y = np.bincount(candidate, minlength=128)
        crossing = np.flatnonzero((counts_x >= 8) != (counts_y >= 8))
        changed = np.flatnonzero(counts_x != counts_y)
        result['layers'].append({'layer': layer, 'arrays': arrays,
                                  'same_input_routes': all(arrays[key]['byte_equal'] for key in ('input', 'original_ids', 'route_weights')),
                                  'histogram_changed_experts': [{'expert': int(i), 'exact': int(counts_x[i]), 'fast': int(counts_y[i])} for i in changed],
                                  'threshold8_crossings': [{'expert': int(i), 'exact': int(counts_x[i]), 'fast': int(counts_y[i])} for i in crossing]})
    for label, predicate in (
        ('first_input_difference', lambda x: not x['arrays']['input']['byte_equal']),
        ('first_result_difference', lambda x: not x['arrays']['result']['byte_equal']),
        ('first_route_difference', lambda x: not x['arrays']['original_ids']['byte_equal']),
        ('first_histogram_threshold8_crossing', lambda x: bool(x['threshold8_crossings'])),
        ('first_result_maxabs_gt_1e_4', lambda x: x['arrays']['result']['max_abs'] > 1e-4),
        ('first_input_maxabs_gt_1e_4', lambda x: x['arrays']['input']['max_abs'] > 1e-4)):
        result[label] = next((layer['layer'] for layer in result['layers'] if predicate(layer)), None)
    with args.out.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    concise = {key: value for key, value in result.items() if key != 'layers'}
    concise['layers'] = [{'layer': x['layer'], 'input_maxabs': x['arrays']['input']['max_abs'],
                          'result_maxabs': x['arrays']['result']['max_abs'], 'result_rel_l2': x['arrays']['result']['relative_l2'],
                          'different_route_ids': x['arrays']['original_ids']['different_values'],
                          'threshold8_crossings': x['threshold8_crossings']} for x in result['layers']]
    print(json.dumps(concise, indent=2))


if __name__ == '__main__':
    main()
