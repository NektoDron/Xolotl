#!/usr/bin/env python3
"""CPU-only traced HTTP request verifies correlation and first-token/content order."""
import http.client
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import serve_m31 as S

class Engine:
    ready = {'vocab':64, 'context':4096}
    def __init__(self):
        self.writes = []
        self.events = [
            {'event':'prefill','prompt_n':1,'prompt_ms':10,'ttft_stages_ms':{'prefill_forward_ms':9}},
            {'event':'token','token':10}, {'event':'token','token':11},
            {'event':'done','finish_reason':'length','timings':{
                'prompt_n':1,'predicted_n':2,'prompt_ms':10,'predicted_ms':1,
                'ttft_stages_ms':{'prefill_forward_ms':9, 'first_decode_forward_ms':1}}}]
    def write(self, value): self.writes.append(value)
    def read(self, **kwargs): return self.events.pop(0)

class Helper:
    ready = {'vocab':64,'eog':[2]}
    def call(self, request):
        if request['op']=='chat': return {'tokens':[1], 'prompt':'hi', 'grammar_active':False}
        if request['op']=='piece': return {'bytes':list(b'R' if request['token']==10 else b'C')}
        if request['op']=='parse':
            return {'message':{'role':'assistant','reasoning_content':'R',
                               'content':'C' if request['content'].endswith('C') else ''}}
        raise AssertionError(request)

class TraceHTTPTest(unittest.TestCase):
    def run_request(self, trace_path):
        engine = Engine()
        server = ThreadingHTTPServer(('127.0.0.1',0), S.Handler)
        server.app = S.Application(engine, Helper())
        thread = threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        connection = http.client.HTTPConnection('127.0.0.1', server.server_port,timeout=5)
        env = dict(os.environ)
        if trace_path: env['M31_TTFT_TRACE'] = str(trace_path)
        else: env.pop('M31_TTFT_TRACE', None)
        try:
            with patch.dict(os.environ, env, clear=True):
                connection.request('POST','/v1/chat/completions', json.dumps({
                    'messages':[{'role':'user','content':'hi'}], 'max_tokens':2,'stream':True}),
                    {'Content-Type':'application/json','X-M31-TTFT-ID':'same-request-515'})
                response = connection.getresponse(); body = response.read().decode()
                self.assertEqual(response.status,200,body)
                self.assertIn('[DONE]',body)
                # Handler finishes tracing after the final flush.
                if trace_path:
                    for _ in range(100):
                        if trace_path.exists() and trace_path.stat().st_size: break
                        threading.Event().wait(.001)
            return engine, body
        finally:
            connection.close();server.shutdown();server.server_close();thread.join()

    def test_same_request_trace_correlates_native_and_visible_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trace.jsonl'; engine, body = self.run_request(path)
            value = json.loads(path.read_text())
            self.assertEqual(value['id'],'same-request-515')
            points = value['points_ms']
            self.assertLessEqual(points['first_native_token'],points['first_generated_delta'])
            self.assertLessEqual(points['first_generated_delta'],points['first_content_delta'])
            self.assertEqual(value['details']['native_timings']['ttft_stages_ms']['prefill_forward_ms'],9)
            self.assertTrue(engine.writes[0]['ttft_trace'])
            self.assertNotIn('same-request-515',body)

    def test_disabled_trace_does_not_change_native_request_or_wire(self):
        engine, body = self.run_request(None)
        self.assertNotIn('ttft_trace',engine.writes[0])
        self.assertNotIn('points_ms',body)

if __name__=='__main__': unittest.main()
