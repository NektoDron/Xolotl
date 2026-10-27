#!/usr/bin/env python3
"""CPU-only checks for ownership, deferred selection and backend cache identity."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from vision_batch import PR_SHA
from vision_encoder import VisionEncoder


class EncoderSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='m31-vision-selection-')
        self.root = Path(self.temp.name)
        for name in ('adapter', 'model', 'mmproj', 'media'):
            (self.root/name).write_bytes(name.encode())
        self.clean_env = patch.dict(os.environ, {}, clear=True)
        self.clean_env.start()

    def tearDown(self):
        self.clean_env.stop()
        self.temp.cleanup()

    def encoder(self, backend='auto', **kwargs):
        return VisionEncoder(self.root/'adapter', self.root/'model', self.root/'mmproj',
                             self.root/'cache', backend=backend, **kwargs)

    def owned(self):
        os.environ.update(M31_CARD_SESSION='1', CUDA_VISIBLE_DEVICES='0')

    def test_selection_is_deferred(self):
        with patch.object(VisionEncoder, '_cuda_free_mib', side_effect=AssertionError('early probe')):
            e = self.encoder()
            self.assertIsNone(e.selected_backend)
            self.assertEqual(e.backend_selection['reason'], 'deferred')

    def test_no_owned_card_never_probes_gpu(self):
        with patch.object(VisionEncoder, '_cuda_free_mib', side_effect=AssertionError('unowned probe')):
            e = self.encoder(); e._resolve_backend()
            self.assertEqual(e.selected_backend, 'cpu')
            self.assertEqual(e.backend_selection['reason'], 'no_owned_card0')

    def test_wrong_device_is_not_owned(self):
        self.owned(); os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        with patch.object(VisionEncoder, '_cuda_free_mib', side_effect=AssertionError('device1 probe')):
            e = self.encoder(); e._resolve_backend()
            self.assertEqual(e.selected_backend, 'cpu')

    def test_vram_threshold_and_worker_lifetime(self):
        self.owned()
        for free, expected in [(4095, 'cpu'), (4096, 'cuda'), (8192, 'cuda')]:
            with self.subTest(free=free), patch.object(VisionEncoder, '_cuda_free_mib', return_value=free) as probe:
                e = self.encoder(); e._resolve_backend(); e._resolve_backend()
                self.assertEqual(e.selected_backend, expected)
                probe.assert_called_once()
                self.assertEqual(json.loads(e.config_path.read_text())['backend'], expected)

    def test_probe_failure_falls_back(self):
        self.owned()
        with patch.object(VisionEncoder, '_cuda_free_mib', side_effect=subprocess.TimeoutExpired('nvidia-smi', 5)):
            e = self.encoder(); e._resolve_backend()
            self.assertEqual(e.selected_backend, 'cpu')
            self.assertEqual(e.backend_selection['reason'], 'free_vram_probe_failed')

    def test_configurable_reserve(self):
        self.owned()
        with patch.object(VisionEncoder, '_cuda_free_mib', return_value=5000):
            e = self.encoder(gpu_min_free_mib=6000); e._resolve_backend()
            self.assertEqual(e.selected_backend, 'cpu')

    def test_explicit_cuda_requires_owner(self):
        with self.assertRaises(ValueError): self.encoder('cuda')
        e = self.encoder('cpu'); e._resolve_backend()
        self.assertEqual(e.selected_backend, 'cpu')

    def test_unload_reselects_after_llm_reload(self):
        self.owned()
        with patch.object(VisionEncoder, '_cuda_free_mib', return_value=8192):
            e = self.encoder(); e._resolve_backend()
        self.assertEqual(e.selected_backend, 'cuda')
        e.unload()
        self.assertIsNone(e.selected_backend)
        with patch.object(VisionEncoder, '_cuda_free_mib', return_value=1000): e._resolve_backend()
        self.assertEqual(e.selected_backend, 'cpu')

    def test_actual_backend_separates_cache_keys(self):
        # Stop before spawning: the real encode cache key and lock have already
        # been created, but this test never invokes a model or GPU subprocess.
        def attempt(e):
            with patch.object(e, '_start', side_effect=RuntimeError('stop before subprocess')):
                with self.assertRaisesRegex(RuntimeError, 'stop before subprocess'):
                    e.encode('<__media__>', [self.root/'media'])
            return {p.stem for p in (self.root/'cache').glob('*.lock')}
        cpu_keys = attempt(self.encoder('cpu'))
        self.assertEqual(len(cpu_keys), 1)
        self.assertEqual(attempt(self.encoder()), cpu_keys)
        self.owned()
        with patch.object(VisionEncoder, '_cuda_free_mib', return_value=8192):
            both_keys = attempt(self.encoder())
        self.assertEqual(len(both_keys), 2)
        self.assertEqual(len(both_keys-cpu_keys), 1)

    def test_precision_environment_separates_actual_cache_keys(self):
        def attempt(e):
            with patch.object(e, '_start', side_effect=RuntimeError('stop')):
                with self.assertRaises(RuntimeError):
                    e.encode('prompt', [self.root/'media'])
            return {p.stem for p in (self.root/'cache').glob('*.lock')}
        before = attempt(self.encoder('cpu'))
        os.environ['GGML_CUDA_CUBLAS_COMPUTE_TYPE'] = 'f32'
        after = attempt(self.encoder('cpu'))
        self.assertEqual(len(after-before), 1)

    def test_worker_uses_constructor_environment_snapshot(self):
        os.environ['GGML_CUDA_CUBLAS_COMPUTE_TYPE'] = 'f32'
        e = self.encoder('cpu')
        os.environ['GGML_CUDA_CUBLAS_COMPUTE_TYPE'] = 'bf16'
        os.environ['GGML_NEW_PARENT_FLAG'] = '1'
        with patch('vision_encoder.subprocess.Popen') as spawn, patch.object(e, '_read', return_value={'ready': True}):
            e._start()
            env = spawn.call_args.kwargs['env']
            self.assertEqual(env['GGML_CUDA_CUBLAS_COMPUTE_TYPE'], 'f32')
            self.assertNotIn('GGML_NEW_PARENT_FLAG', env)
        e.process = None
        e.close()

    def test_cache_manifest_backend_must_match_selection(self):
        e = self.encoder('cpu')
        with patch.object(e, '_start', side_effect=RuntimeError('stop')):
            with self.assertRaises(RuntimeError): e.encode('prompt', [self.root/'media'])
        key = next((self.root/'cache').glob('*.lock')).stem
        entry = self.root/'cache'/key/'batch'; entry.mkdir(parents=True)
        (entry/'manifest.json').write_text(json.dumps(dict(format='m31-vision-batch-v1',
            source_pr=PR_SHA, encoded=True, backend='cuda', rows=1, visual_rows=0)))
        with self.assertRaisesRegex(RuntimeError, 'invalid vision cache manifest'):
            e.encode('prompt', [self.root/'media'])


if __name__ == '__main__':
    unittest.main()
