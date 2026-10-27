#!/usr/bin/env python3
"""Prewarm must observe loaded LLM memory before HTTP becomes ready; no GPU."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import serve_m31 as S
from vision_encoder import VisionEncoder

class PrewarmTests(unittest.TestCase):
    def test_start_keeps_existing_worker_without_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['adapter','model','mmproj']:
                (root/name).write_text('fixture')
            e = VisionEncoder(root/'adapter',root/'model',root/'mmproj',root/'cache',backend='cpu')
            worker = Mock();worker.poll.return_value = None;e.process = worker
            with patch('vision_encoder.subprocess.Popen',side_effect=AssertionError('restarted live worker')):
                first, second = e.start(), e.start()
                self.assertFalse(first['worker_started']);self.assertFalse(second['worker_started'])
            self.assertIs(e.process,worker)
            worker.terminate.assert_not_called()

    def run_main(self, prewarm):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {'adapter':'unused-adapter','cache_dir':str(root/'cache'),
                      'mmproj':'unused-mmproj','backend':'auto','prewarm':prewarm}
            path = root/'vision.json';path.write_text(json.dumps(config))
            events = []
            def rpc(argv, **kwargs):
                name = 'tokenizer' if argv[0]=='tokenizer' else 'engine'
                events.append(name+'_ready')
                ready = {'vocab':64,'context':4096,'eog':[2]}
                return Mock(ready=ready)
            encoder = Mock()
            def start():
                self.assertIn('engine_ready',events, 'VRAM probe happened before LLM allocations')
                events.append('worker_start')
                return {'worker_started':True, 'backend':'cpu', 'milliseconds':0}
            encoder.start.side_effect = start
            def server(*args, **kwargs):
                events.append('http_constructed')
                http = Mock()
                http.serve_forever.side_effect = lambda: events.append('http_ready')
                return http
            # main imports the module by name after setting its own path.
            with patch('vision_encoder.VisionEncoder',return_value=encoder) as constructor, \
                 patch.object(S,'RPC',side_effect=rpc) as rpc_mock, \
                 patch.object(S,'ThreadingHTTPServer',side_effect=server), \
                 patch.object(S.signal,'signal'), \
                 patch.object(sys,'argv',['serve_m31.py','--engine','engine','--tokenizer','tokenizer',
                       '--model','model','--vision-config',str(path)]):
                if type(prewarm) is not bool:
                    with self.assertRaisesRegex(ValueError,'prewarm must be boolean'):S.main()
                    constructor.assert_not_called();rpc_mock.assert_not_called()
                else:
                    S.main()
                    self.assertNotIn('prewarm',constructor.call_args.kwargs)
            return events, encoder

    def test_prewarm_runs_after_engine_ready_before_http_ready(self):
        events, encoder = self.run_main(True)
        self.assertLess(events.index('engine_ready'),events.index('worker_start'))
        self.assertLess(events.index('worker_start'),events.index('http_constructed'))
        self.assertLess(events.index('worker_start'),events.index('http_ready'))
        encoder.start.assert_called_once()

    def test_default_false_does_not_start_encoder(self):
        events, encoder = self.run_main(False)
        self.assertNotIn('worker_start',events)
        encoder.start.assert_not_called()

    def test_non_boolean_is_rejected_before_any_rpc_or_worker(self):
        for value in [1,0,'true',None]:
            with self.subTest(value=value):self.run_main(value)

if __name__=='__main__':unittest.main()
