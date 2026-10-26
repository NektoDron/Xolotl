#!/usr/bin/env python3
"""Check replay invariants and reject incomplete/corrupt calibration traces."""
import json
from pathlib import Path
import tempfile
import unittest
from cache_rank import load_trace, simulate


class CacheReplayTests(unittest.TestCase):
    def test_hot_seed_survives_layer_scan(self):
        traffic = [[(3, 0)], [(4, 0)], [(5, 0)]] * 8
        cold = simulate(traffic, 2)
        seeded = simulate(traffic, 2, [(3, 0)])
        self.assertEqual(cold['hits'], 0)
        self.assertEqual(seeded['hot_hits'], 8)
        self.assertEqual(seeded['misses'], 16)
        self.assertEqual(seeded['requests'], seeded['hits'] + seeded['misses'])

    def test_protected_batch_is_not_evicted_mid_batch(self):
        traffic = [[(3, 0), (3, 1)], [(3, 1), (3, 2)], [(3, 1), (3, 2)]]
        result = simulate(traffic, 2)
        self.assertEqual(result['misses'], 3)
        self.assertEqual(result['hits'], 3)

    def test_stream_reserve_must_fit_batch(self):
        with self.assertRaises(ValueError):
            simulate([[(3, 1), (3, 2)]], 2, [(3, 0)])

    def test_reject_incomplete_and_duplicate_route_trace(self):
        header = {'kind': 'header', 'format': 'm31-routing-trace-v1', 'layers': 4,
                  'dense_lead': 3, 'experts': 4, 'used': 2}
        case = {'kind': 'case', 'case': 'held', 'tokens': 2}
        route = {'kind': 'routes', 'case': 'held', 'position': 0, 'layer': 3,
                 'n': 1, 'ids': [0, 1], 'weights': [0.5, 0.5]}
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'trace.jsonl'
            path.write_text('\n'.join(map(json.dumps, [header, case, route])))
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                load_trace(path)
            path.write_text('\n'.join(map(json.dumps, [header, case, route, route])))
            with self.assertRaisesRegex(ValueError, 'duplicate observed'):
                load_trace(path)
            route['n'] = 2
            route['ids'] *= 2
            route['weights'] *= 2
            path.write_text('\n'.join(map(json.dumps, [header, case, route])))
            h, cases, grouped = load_trace(path)
            self.assertEqual(len(grouped['held']), 2)
            route['ids'] = [0, 0, 1, 2]
            path.write_text('\n'.join(map(json.dumps, [header, case, route])))
            with self.assertRaisesRegex(ValueError, 'duplicate top-k'):
                load_trace(path)


if __name__ == '__main__':
    unittest.main()
