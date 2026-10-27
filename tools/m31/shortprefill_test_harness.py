#!/usr/bin/env python3
"""CPU-only aggregation checks: warmup exclusion, medians and actual KV suffix."""
import json
from pathlib import Path
import tempfile
import unittest

from shortprefill_summarize import markdown, summarize


class AggregateTests(unittest.TestCase):
    def test_samples_warmup_and_prefix(self):
        rows = []
        for arm, times in [('off', [999, 100, 80, 90]), ('cpu', [999, 50, 70, 60])]:
            for index, elapsed in enumerate(times):
                rows.append({'case': 'text128', 'arm': arm, 'warmup': index == 0,
                             'tokens': 128, 'prefix_tokens': 0, 'pass': True,
                             'forward_ms': elapsed, 'native_ttft_ms': elapsed+1,
                             'head_ms': 1, 'tokens_per_second': 128000/elapsed,
                             'counters': {'gpu_upload_bytes': elapsed*1000},
                             'chunks': [{'tokens': 128}]})
        rows.append({**rows[-1], 'case': 'suffix128', 'prefix_tokens': 515})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'summary.json'
            path.write_text(json.dumps({'format': 'm31-shortprefill-result-v1', 'pass': True, 'runs': rows}))
            value = summarize([path])
        found = {r['arm']: r for r in value['rows'] if r['case'] == 'text128'}
        self.assertEqual(found['off']['forward_ms'], 90)
        self.assertEqual(found['cpu']['forward_ms'], 60)
        self.assertEqual(found['cpu']['speedup_vs_off'], 1.5)
        self.assertEqual(found['cpu']['samples'], 3)
        self.assertEqual(value['cpu_wins_at_tested_tokens'], [128])
        self.assertEqual(value['rows'][-1]['prefix_tokens'], 515)
        self.assertIn('suffix128', markdown(value))

    def test_incomplete_does_not_become_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'summary.json'
            path.write_text(json.dumps({'format': 'm31-shortprefill-result-v1', 'pass': False, 'runs': []}))
            self.assertFalse(summarize([path])['all_sessions_complete'])

    def test_different_sessions_do_not_share_baselines(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name, arm in [('old', 'off'), ('new', 'cpu')]:
                root = Path(directory)/name
                root.mkdir()
                path = root/'summary.json'
                path.write_text(json.dumps({'format': 'm31-shortprefill-result-v1', 'pass': True, 'runs': [{
                    'case': 'same', 'arm': arm, 'tokens': 128, 'prefix_tokens': 0, 'pass': True,
                    'forward_ms': 100, 'native_ttft_ms': 101, 'head_ms': 1,
                    'tokens_per_second': 1280, 'chunks': [{'tokens': 128}]}]}))
                paths.append(path)
            rows = summarize(paths)['rows']
            self.assertNotIn('speedup_vs_off', next(r for r in rows if r['arm'] == 'cpu'))


if __name__ == '__main__':
    unittest.main()
