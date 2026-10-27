#!/usr/bin/env python3
"""CPU-only tests of exact token count, same-request cache states and SSE timing."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import ttft_session as T
import ttft_summarize as S


class FakeVocab:
    def call(self, request):
        if request['op'] == 'detokenize':
            return {'content': 'x'*len(request['tokens'])}
        count = request['messages'][0]['content'].count('x') + 25
        return {'tokens': list(range(count))}


class SessionTests(unittest.TestCase):
    def test_prompt_cache_options_are_explicit_and_default_false(self):
        default = T.request_options()
        allowed = T.request_options(allow_prompt_cache=True)
        self.assertIs(default['cache_prompt'], False)
        self.assertIs(allowed['cache_prompt'], True)
        self.assertEqual({k:v for k,v in default.items() if k != 'cache_prompt'},
                         {k:v for k,v in allowed.items() if k != 'cache_prompt'})
        body = {**allowed, 'messages': [{'role': 'user', 'content': 'same prompt'}]}
        optout = T.cache_optout_body(body)
        self.assertIs(body['cache_prompt'], True)
        self.assertIs(optout['cache_prompt'], False)
        self.assertEqual({k:v for k,v in body.items() if k != 'cache_prompt'},
                         {k:v for k,v in optout.items() if k != 'cache_prompt'})

    def test_optout_proves_full_prefill_and_identical_greedy_result(self):
        common = {'request_prompt_sha256': 'same-prompt', 'content': 'OK',
                  'reasoning_content': 'reason', 'finish_reason': 'stop'}
        cached = {**common, 'native_timings': {'predicted_n': 5, 'prompt_cache': {'hit': True}, 'actual_chunks': []}}
        fresh = {**common, 'native_timings': {'predicted_n': 5, 'actual_chunks': [{'position': 0, 'rows': 515}]}}
        T.assert_optout(fresh, cached, 515)
        self.assertTrue(all(fresh['prompt_cache_optout_checks'].values()))
        bad = {**common, 'native_timings': {'predicted_n': 5, 'prompt_cache': {'hit': True}, 'actual_chunks': []}}
        with self.assertRaises(RuntimeError):
            T.assert_optout(bad, cached, 515)
        with self.assertRaises(RuntimeError):
            T.assert_optout(fresh, fresh, 515)

    def test_exact_text_counts(self):
        for target in (515, 3863):
            messages, ids = T.exact_text_chat(FakeVocab(), list(range(5000)), target)
            self.assertEqual(len(ids), target)
            self.assertEqual(messages[0]['role'], 'user')
            self.assertTrue(messages[0]['content'].endswith('Reply only OK.'))

    def test_too_short_corpus_rejected(self):
        with self.assertRaises(ValueError):
            T.exact_text_chat(FakeVocab(), [1, 2], 515)

    def test_archive_only_complete_matching_owned_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root/'cache'
            entry = cache/('a'*64)
            (entry/'batch').mkdir(parents=True)
            (entry/'batch/manifest.json').write_text(json.dumps({'rows': 515, 'backend': 'cuda'}))
            (cache/'sources').mkdir()
            result = T.archive_batches(cache, root/'archive', 'image', 515)
            self.assertEqual(len(result), 1)
            self.assertFalse(entry.exists())
            self.assertTrue((cache/'sources').is_dir())
            self.assertTrue((Path(result[0]['archive'])/'batch/manifest.json').is_file())

    def test_trace_correlation_skips_partial_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trace.jsonl'
            path.write_text('{"id":"other"}\n{"id":"wanted","points_ms":{"first_generated_delta":1}}\n{')
            self.assertEqual(T.trace_for(path, 'wanted', 0)['points_ms']['first_generated_delta'], 1)

    def test_partition_does_not_add_nested_or_decode_timers(self):
        result = {'ttft_s': 1, 'trace': {'points_ms': {
            'http_body_parsed': 10, 'prepared': 100, 'native_rpc_start': 102,
            'first_native_token': 900, 'first_generated_delta': 950,
            'first_content_delta': 1200}, 'stages_ms': {'encoder_python_total_ms': 80}},
            'native_timings': {'ttft_stages_ms': {'prefill_forward_ms': 700,
                'reset_ms': 2, 'first_token_sent_from_request_ms': 795,
                'first_decode_forward_ms': 30}}}
        value = T.breakdown(result)
        self.assertEqual(value['critical_path_sum_ms'], 1000)
        self.assertEqual(value['partition_residual_ms'], 0)
        self.assertEqual(value['native_before_first_token_leaf_sum_ms'], 702)
        self.assertEqual(value['native_transport_ms'], 3)
        self.assertEqual(value['reasoning_until_content_ms'], 250)
        self.assertEqual(value['after_first_token_ms']['first_decode_forward_ms'], 30)

    def test_postprocess_uses_saved_trace_without_mutating_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native = {'prompt_n': 515, 'ttft_stages_ms': {'prefill_forward_ms': 700,
                'first_token_sent_from_request_ms': 795}}
            trace = {'id': 'one', 'points_ms': {'http_body_parsed': 10, 'prepared': 100,
                     'native_rpc_start': 102, 'first_native_token': 900,
                     'first_generated_delta': 950, 'first_content_delta': 1200},
                     'stages_ms': {'prepare_total_ms': 90},
                     'details': {'encoder': {'worker_started': False}, 'native_timings': native}}
            original = json.dumps({'pass': True, 'server_returncode': 0,
                'trace_file': str(root/'trace.jsonl'), 'cases': [{'name': 'test',
                'id': 'one', 'pass': True, 'ttft_s': 1, 'visible_ttft_s': 1.2}]})
            (root/'results.json').write_text(original)
            (root/'trace.jsonl').write_text(json.dumps(trace)+'\n')
            summary = S.summarize(root)
            self.assertTrue(summary['session_complete'])
            self.assertFalse(summary['cases'][0]['encoder_worker_started'])
            self.assertEqual(summary['cases'][0]['native_stages_ms']['prefill_forward_ms'], 700)
            self.assertEqual((root/'results.json').read_text(), original)

    def test_prompt_cache_leaf_timers_are_accounted_once(self):
        result = {'ttft_s': .02, 'trace': {'points_ms': {
            'http_body_parsed': 1, 'prepared': 2, 'native_rpc_start': 3,
            'first_native_token': 15, 'first_generated_delta': 19}, 'details': {}},
            'native_timings': {'ttft_stages_ms': {'reset_ms': 1,
                'first_token_sent_from_request_ms': 11},
                'prompt_cache': {'key_ms': 4, 'restore_ms': 5}}}
        value = T.breakdown(result)
        self.assertEqual(value['native_prompt_cache_ms'], {'key_ms': 4, 'restore_ms': 5})
        self.assertEqual(value['native_before_first_token_leaf_sum_ms'], 10)
        self.assertEqual(value['native_unattributed_before_first_token_ms'], 1)

    def test_profile_traffic_differences_cumulative_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'server.log'
            path.write_text('''M31 GPU route demand layer=3 n=515 required=2 required_bytes=20 predicted_used=2 predicted_unused=1 predicted_bytes=30 history=0
M31 forward pos=0 n=515 seconds=1 tps=515
M31 GPU expert profile label=forward forwards=1 bytes=30
M31 GPU prefetch profile label=forward batches=1 bytes=20
M31 GPU upload profile label=forward bytes=10
M31 GPU expert profile label=forward forwards=1 bytes=30
M31 request timings {}
M31 GPU route demand layer=3 n=515 required=2 required_bytes=20 predicted_used=2 predicted_unused=0 predicted_bytes=20 history=1
M31 forward pos=0 n=515 seconds=1 tps=515
M31 GPU expert profile label=forward forwards=2 bytes=50
M31 GPU prefetch profile label=forward batches=2 bytes=40
M31 GPU upload profile label=forward bytes=10
M31 request timings {}
M31 forward pos=515 n=1 seconds=.1 tps=10
M31 GPU expert profile label=forward forwards=2 bytes=50
M31 GPU prefetch profile label=forward batches=2 bytes=40
M31 GPU upload profile label=forward bytes=10
M31 request timings {}
''')
            rows = S.profile_traffic(path, [{'name': str(i), 'expected_rows': 515} for i in range(3)])
            self.assertEqual([r['prefill_dma_bytes'] for r in rows], [30, 20, 0])
            self.assertEqual([r['unused_prefetch_bytes'] for r in rows], [10, 0, 0])

    def test_sse_records_reasoning_content_and_native_length(self):
        captured = {}
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                captured['id'] = self.headers['X-M31-TTFT-ID']
                captured['body'] = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                for event in [
                    {'choices': [{'delta': {'role': 'assistant'}, 'finish_reason': None}]},
                    {'choices': [{'delta': {'reasoning_content': 'think'}, 'finish_reason': None}]},
                    {'choices': [{'delta': {'content': 'OK'}, 'finish_reason': None}]},
                    {'choices': [{'delta': {}, 'finish_reason': 'stop'}], 'timings': {'prompt_n': 515}},
                ]:
                    self.wfile.write(('data: '+json.dumps(event)+'\n\n').encode())
                    self.wfile.flush()
                self.wfile.write(b'data: [DONE]\n\n')
                self.wfile.flush()
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = T.request(server.server_port, '/v1/chat/completions', {'test': True}, 'probe-id', 2)
            self.assertEqual(captured, {'id': 'probe-id', 'body': {'test': True}})
            self.assertEqual(result['content'], 'OK')
            self.assertEqual(result['reasoning_content'], 'think')
            self.assertEqual(result['native_timings']['prompt_n'], 515)
            self.assertGreaterEqual(result['visible_ttft_s'], result['ttft_s'])
            self.assertEqual(result['finish_reason'], 'stop')
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
