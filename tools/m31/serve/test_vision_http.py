#!/usr/bin/env python3
"""CPU tests: real bounded media ingestion and mixed JSONL handoff, without model/GPU."""
import base64
import importlib.util
import io
import http.client
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vision_http import MARKER, MediaInput, install_video_reader
from test_common import FakeRPC, Helper, done, shim
from engine_adapter import M31Engine, PreparedIds, standalone
import numpy as np
from PIL import Image


class Encoder:
    def __init__(self, directory):
        self.dir = Path(directory)
        self.calls = []

    def encode(self, prompt, media, video_fps=1):
        self.calls.append((prompt, list(media), video_fps))
        ids = np.array([1, -1, -1, 3], dtype='<i4')
        ids.tofile(self.dir/'text_ids.i32')
        path = self.dir/'manifest.json'
        path.write_text(json.dumps({'format':'m31-vision-batch-v1', 'encoded':True, 'rows':4}))
        return path


class VisionHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='m31-vision-http-')
        self.media = MediaInput(self.temp.name)
        raw = io.BytesIO()
        Image.new('RGB', (8,8), 'red').save(raw, format='PNG')
        self.png = raw.getvalue()
        self.url = 'data:image/png;base64,'+base64.b64encode(self.png).decode()
        self.previous_parser = shim.S.OutputParser

    def tearDown(self):
        shim.S.OutputParser = self.previous_parser
        self.temp.cleanup()

    def messages(self):
        return [{'role':'user', 'content':[{'type':'text','text':'before '},
                {'type':'image_url','image_url':{'url':self.url}}, {'type':'text','text':' after'}]}]

    def test_multiple_images_preserve_conversation_order_without_mutating_input(self):
        messages = self.messages()+[{'role':'assistant','content':'seen'}, *self.messages()]
        copied, paths, markers, fps = self.media.prepare(messages)
        self.assertEqual(len(paths), 2)
        self.assertEqual(paths[0], paths[1])
        self.assertEqual(paths[0].read_bytes(), self.png)
        self.assertEqual(copied[0]['content'], 'before '+markers[0]+' after')
        self.assertEqual(copied[2]['content'], 'before '+markers[1]+' after')
        self.assertEqual(messages[0]['content'][1]['image_url']['url'], self.url)
        self.assertEqual(fps, 1)

    def test_reserved_mtmd_text_marker_is_rejected(self):
        messages = self.messages()
        messages[0]['content'][0]['text'] = MARKER
        with self.assertRaisesRegex(ValueError, 'reserved'):
            self.media.prepare(messages)

    def test_invalid_base64_missing_file_and_disabled_http_are_bounded_errors(self):
        for source in ('data:image/png;base64,!!!', '/nonexistent-m31-file', 'https://example.invalid/image.png'):
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.media._read(source)

    def test_media_bytes_total_count_and_pixels_are_enforced(self):
        for limits in ({'max_media_bytes':10}, {'max_total_bytes':10}, {'max_image_pixels':4}):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                MediaInput(self.temp.name, **limits).prepare(self.messages())
        messages = self.messages()
        messages[0]['content'] *= 2
        with self.assertRaisesRegex(ValueError, 'max_media'):
            MediaInput(self.temp.name, max_media=1).prepare(messages)

    def test_video_openai_frontend_preserves_file_order_and_fps(self):
        install_video_reader()
        from serve.frontend import openai_to_messages
        request = {'messages':[{'role':'user','content':[
            {'type':'text','text':'before'}, {'type':'video_url','video_url':{'url':'/tmp/clip.mp4', 'fps':2}},
            {'type':'text','text':'after'}]}]}
        messages, _, _ = openai_to_messages(request)
        self.assertEqual(messages[0]['content'][1], {'type':'image','source':'/tmp/clip.mp4','media_type':'video','fps':2})
        self.assertEqual(messages[0]['content'][2]['text'], 'after')

    def test_real_reference_video_frame_limit_and_selected_fps(self):
        video = Path('/mnt/ai/logs/m31-ref/media/shapes-12s.mp4')
        if not video.exists():
            self.skipTest('reference video absent')
        messages = [{'role':'user','content':[{'type':'video_url', 'video_url':{'url':str(video)}}]}]
        _, paths, _, fps = self.media.prepare(messages)
        self.assertEqual(fps, 1)
        self.assertEqual(paths[0].read_bytes(), video.read_bytes())
        with self.assertRaisesRegex(ValueError, 'max_video_frames'):
            MediaInput(self.temp.name, max_video_frames=12).prepare(messages, requested_fps=2)
        for bad in (0, -1, float('nan'), True, 31):
            with self.subTest(fps=bad), self.assertRaisesRegex(ValueError, 'video_fps'):
                self.media.prepare(messages, requested_fps=bad)

    def test_standalone_handoff_counts_visual_rows_and_keeps_text_helper_ids(self):
        helper = Helper()
        original = helper.call
        def render(request):
            result = original(request)
            if request['op'] == 'chat':
                result['prompt'] = request['messages'][0]['content']
            return result
        helper.call = render
        encoder = Encoder(self.temp.name)
        engine = FakeRPC()
        app = standalone.Application(engine, helper, vision=encoder, media_input=self.media)
        params, _ = app.prepare({'messages':self.messages(), 'max_tokens':126})
        self.assertEqual(params['tokens'], [1])
        self.assertEqual(params['max_tokens'], 124)  # ctx128 minus all four mixed rows
        self.assertEqual(params['mixed_manifest'], str(Path(self.temp.name)/'manifest.json'))
        self.assertEqual(encoder.calls[0][0], 'before '+MARKER+' after')
        self.assertEqual(engine.writes, [])

    def test_common_service_handoff_grammar_replay_compares_original_helper_ids(self):
        helper = Helper()
        original = helper.call
        def render(request):
            result = original(request)
            if request['op'] == 'chat':
                result['prompt'] = request['messages'][0]['content']
            return result
        helper.call = render
        rpc = FakeRPC([done()])
        service = shim.make_service(rpc, helper, Helper(), vision=Encoder(self.temp.name), media_input=self.media)
        ids, _, maximum = service.prepare(self.messages(), [], {}, max_new=126)
        self.assertEqual(ids, [1,-1,-1,3])
        self.assertEqual(maximum,124)
        list(service.engine.generate(ids, maximum, {}, threading.Event()))
        self.assertEqual(rpc.writes[0]['tokens'], [1])
        self.assertEqual(rpc.writes[0]['mixed_manifest'], ids.mixed_manifest)

    def test_real_http_image_request_reaches_native_mixed_protocol(self):
        helper = Helper()
        original = helper.call
        def render(request):
            result = original(request)
            if request['op'] == 'chat':
                result['prompt'] = request['messages'][0]['content']
            return result
        helper.call = render
        rpc = FakeRPC([{'event':'token','token':10}, done()])
        server = ThreadingHTTPServer(('127.0.0.1', 0), standalone.Handler)
        server.daemon_threads = True
        server.app = standalone.Application(rpc, helper, vision=Encoder(self.temp.name), media_input=self.media)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = http.client.HTTPConnection('127.0.0.1', server.server_address[1], timeout=5)
        try:
            connection.request('POST', '/v1/chat/completions',
                               json.dumps({'messages':self.messages(), 'max_tokens':1}),
                               {'Content-Type':'application/json'})
            response = connection.getresponse()
            result = json.loads(response.read())
            self.assertEqual(response.status, 200, result)
            self.assertEqual(result['choices'][0]['message']['content'], 'hello')
            self.assertIn('mixed_manifest', rpc.writes[0])
            self.assertEqual(rpc.writes[0]['tokens'], [1])
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_template_dropping_media_marker_fails_before_engine_send(self):
        app = standalone.Application(FakeRPC(), Helper(), vision=Encoder(self.temp.name), media_input=self.media)
        with self.assertRaisesRegex(ValueError, 'media order'):
            app.prepare({'messages':self.messages()})
        self.assertEqual(app.engine.writes, [])


if __name__ == '__main__':
    unittest.main()
