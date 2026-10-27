#!/usr/bin/env python3
"""CPU-only protocol tests for the raw-ID long comparison client."""
import copy
import os
import subprocess
import threading
import unittest
from unittest.mock import patch

import shortprefill_long_rpc as L


class FakeRPC:
    def __init__(self, chunks=None, cache=None):
        self.lock = threading.RLock()
        self.body = None
        timings = {'prompt_n': 3, 'predicted_n': 1, 'prompt_ms': 1,
                   'prompt_per_second': 3000,
                   'actual_chunks': chunks if chunks is not None else [{'position': 0, 'rows': 2}, {'position': 2, 'rows': 1}],
                   'ttft_stages_ms': {'reset_ms': .01, 'prefill_forward_ms': .8, 'prefill_head_ms': .1}}
        if cache is not None:
            timings['prompt_cache'] = cache
        self.frames = iter([{'event': 'prefill', 'prompt_n': 3},
                            {'event': 'token', 'token': 7, 'index': 0},
                            {'event': 'done', 'timings': timings}])
    def write(self, body):
        self.body = copy.deepcopy(body)
    def read(self):
        return next(self.frames)


class ProtocolTests(unittest.TestCase):
    def test_raw_fresh_request_and_all_actual_chunks(self):
        rpc = FakeRPC()
        result = L.raw_request(rpc, [1, 2, 3], 'test')
        self.assertIs(rpc.body['cache_prompt'], False)
        self.assertTrue(rpc.body['ttft_trace'])
        self.assertEqual(rpc.body['max_tokens'], 1)
        self.assertFalse(rpc.body['interactive'])
        self.assertEqual(result['sampled_token'], 7)
        self.assertEqual(result['actual_chunks'], [{'position': 0, 'rows': 2}, {'position': 2, 'rows': 1}])
        self.assertTrue(result['prompt_cache_proof']['all_rows_fresh'])
    def test_cache_hits_and_missing_fresh_chunks_are_rejected(self):
        for rpc in [FakeRPC(cache={'hit': True}), FakeRPC(chunks=[]),
                    FakeRPC(chunks=[{'position': 1, 'rows': 3}])]:
            with self.assertRaises(RuntimeError):
                L.raw_request(rpc, [1, 2, 3], 'bad')
    def test_quant_defaults_unset_and_budget_disabled(self):
        with patch.dict(os.environ, {'M31_SHORT_PREFILL_GPU_QUANT': '0', 'M31_SHORT_PREFILL_GPU_OAI': '0'}):
            argv, env = L.environment_argv({'environment': {}}, {'M31_SHORT_PREFILL_MODE': 'auto'})
        self.assertIn('M31_SHORT_PREFILL_GPU_QUANT', argv)
        self.assertIn('M31_SHORT_PREFILL_GPU_OAI', argv)
        self.assertNotIn('M31_SHORT_PREFILL_GPU_QUANT', env)
        self.assertEqual(env['M31_PROMPT_CACHE_MIB'], '0')
        self.assertEqual(env['M31_GPU_PREFETCH_HISTORY'], '0')

    def test_graceful_shutdown_closes_stdin_then_waits_before_rpc_fallback(self):
        events = []
        class Process:
            returncode = None
            def poll(self):
                return self.returncode
            class Input:
                def close(self):
                    events.append('stdin-close')
            stdin = Input()
            def wait(self, timeout):
                events.append(('wait', timeout))
                self.returncode = 0
        class ClosingRPC:
            process = Process()
            def close(self):
                events.append('rpc-close')
        L.close_gracefully(ClosingRPC())
        self.assertEqual(events, ['stdin-close', ('wait', 300), 'rpc-close'])

    def test_graceful_timeout_keeps_bounded_rpc_cleanup(self):
        events = []
        class Process:
            returncode = None
            def poll(self):
                return None
            class Input:
                def close(self):
                    events.append('stdin-close')
            stdin = Input()
            def wait(self, timeout):
                raise subprocess.TimeoutExpired('native', timeout)
        class ClosingRPC:
            process = Process()
            def close(self):
                events.append('rpc-close')
        with self.assertRaisesRegex(RuntimeError, 'shutdown exceeded'):
            L.close_gracefully(ClosingRPC(), timeout=1)
        self.assertEqual(events, ['stdin-close', 'rpc-close'])


if __name__ == '__main__':
    unittest.main()
