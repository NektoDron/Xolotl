"""Persistent, isolated PR mtmd encoder for native Xolotl mixed prefill.

The subprocess owns a PR-specific ggml ABI. Only manifests and F32 arrays cross
this boundary; it never executes an LLM. CPU is the safe default. Automatic CUDA selection requires card ownership and conservative free VRAM.
"""
from __future__ import annotations
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import select
import subprocess
import threading
import time
import uuid

from vision_batch import PR_SHA, assemble, load


class VisionEncoder:
    def __init__(self, adapter, model, mmproj, cache_dir, *, backend='cpu',
                 threads=16, cpu_workers=6, timeout=180, embedding_cache_mb=256, gpu_min_free_mib=4096):
        if backend == 'cuda' and (os.environ.get('M31_CARD_SESSION') != '1' or os.environ.get('CUDA_VISIBLE_DEVICES') != '0'):
            raise ValueError('CUDA vision requires an aictl card-0 session')
        if backend not in ('cpu', 'cuda', 'auto'):
            raise ValueError('vision backend must be cpu, cuda or auto')
        if not 1 <= int(cpu_workers) <= 16 or not 1 <= int(threads) <= 128:
            raise ValueError('invalid vision worker/thread count')
        if int(gpu_min_free_mib) < 1:
            raise ValueError('GPU minimum free memory must be positive')
        # Match cache identity to the exact environment inherited by the worker.
        # A later parent profile change must not silently reuse old embeddings.
        self._compute_env = {key: value for key, value in os.environ.items()
                             if key.startswith('GGML_')}
        self.requested_backend = backend
        self.selected_backend = None if backend == 'auto' else backend
        self.gpu_min_free_mib = int(gpu_min_free_mib)
        self.backend_selection = dict(requested=backend, backend=self.selected_backend,
                                      min_free_mib=self.gpu_min_free_mib,
                                      reason='deferred' if backend == 'auto' else 'explicit')
        self.adapter, self.cache_dir = str(Path(adapter).resolve()), Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout, self.lock = float(timeout), threading.RLock()
        self.config = dict(model=str(Path(model).resolve()), mmproj=str(Path(mmproj).resolve()),
                           backend='cpu' if backend == 'auto' else backend, threads=int(threads), cpu_workers=int(cpu_workers),
                           encode=True, encoder_mode='single', add_special=False,
                           embedding_cache_mb=int(embedding_cache_mb))
        self.config_path = self.cache_dir / ('worker-' + uuid.uuid4().hex + '.json')
        self.config_path.write_text(json.dumps(self.config) + '\n')
        self.process = None
        self.log = None
        self.last_stats = {}

    @staticmethod
    def _owns_cuda_card0():
        return os.environ.get('M31_CARD_SESSION') == '1' and os.environ.get('CUDA_VISIBLE_DEVICES') == '0'

    @staticmethod
    def _cuda_free_mib():
        result = subprocess.run(['nvidia-smi', '-i', '0', '--query-gpu=memory.free',
                                 '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=5, check=True)
        value = int(result.stdout.strip())
        if value < 0:
            raise ValueError('invalid GPU free memory')
        return value

    def _resolve_backend(self):
        # Call only after the LLM has loaded. Keep this choice for the worker
        # lifetime: its own allocations must not change a later selection.
        if self.selected_backend is not None:
            return
        selected, free, reason = 'cpu', None, 'no_owned_card0'
        if self._owns_cuda_card0():
            try:
                free = self._cuda_free_mib()
                if free >= self.gpu_min_free_mib:
                    selected, reason = 'cuda', 'sufficient_free_vram'
                else:
                    reason = 'insufficient_free_vram'
            except (OSError, ValueError, subprocess.SubprocessError):
                reason = 'free_vram_probe_failed'
        self.selected_backend = selected
        self.config['backend'] = selected
        self.backend_selection = dict(requested=self.requested_backend, backend=selected,
                                      free_mib=free, min_free_mib=self.gpu_min_free_mib, reason=reason)
        self.config_path.write_text(json.dumps(self.config) + '\n')

    def _read(self):
        if not select.select([self.process.stdout], [], [], self.timeout)[0]:
            self.close()
            raise TimeoutError('vision encoder timed out')
        line = self.process.stdout.readline()
        if not line:
            code = self.process.poll()
            self.close()
            raise RuntimeError(f'vision encoder exited ({code}); see encoder.log')
        try:
            return json.loads(line)
        except ValueError as exc:
            self.close()
            raise RuntimeError('invalid vision encoder response') from exc

    def _start(self):
        self._resolve_backend()
        if self.process is not None and self.process.poll() is None:
            return
        self.log = (self.cache_dir / 'encoder.log').open('ab', buffering=0)
        env = {key: value for key, value in os.environ.items()
               if not key.startswith('GGML_')}
        env.update(self._compute_env)
        env['OPENBLAS_NUM_THREADS'] = '1'
        env['OMP_NUM_THREADS'] = str(self.config['threads'])
        if self.config['backend'] == 'cpu':
            env['CUDA_VISIBLE_DEVICES'] = ''
        if self.config['backend'] == 'cuda' and (env.get('M31_CARD_SESSION') != '1' or env.get('CUDA_VISIBLE_DEVICES') != '0'):
            self.log.close(); self.log = None
            raise ValueError('CUDA vision requires an aictl card-0 session')
        self.process = subprocess.Popen([self.adapter, '--worker', str(self.config_path)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, text=True, bufsize=1, env=env)
        reply = self._read()
        if not reply.get('ready'):
            self.close()
            raise RuntimeError('vision encoder failed to initialize')

    def prepare(self, prompt, media, **kwargs):
        return self.encode(prompt, media, **kwargs)

    def encode(self, prompt, media, *, video_fps=1.0, timestamp_interval_ms=5000):
        with self.lock:
            return self._encode(prompt, media, video_fps=video_fps,
                                timestamp_interval_ms=timestamp_interval_ms)

    def _encode(self, prompt, media, *, video_fps, timestamp_interval_ms):
        if not isinstance(prompt, str) or not media:
            raise ValueError('vision requires a prompt and media')
        if not math.isfinite(video_fps) or not 0 < video_fps <= 30:
            raise ValueError('video_fps must be in (0, 30]')
        self._resolve_backend()
        paths = [Path(p).resolve(strict=True) for p in media]
        # Hash bytes rather than paths/mtime: the same upload can be re-used across
        # requests, and an overwritten local file must never hit an old entry.
        media_hashes = []
        for path in paths:
            h = hashlib.sha256()
            with path.open('rb') as source:
                for block in iter(lambda: source.read(1024*1024), b''):
                    h.update(block)
            media_hashes.append(h.hexdigest())
        identity = dict(format='m31-vision-cache-v3', source_pr=PR_SHA,
                        encoder_environment=self._compute_env,
                        config=self.config, prompt=prompt, media=media_hashes,
                        video_fps=video_fps, timestamp_interval_ms=timestamp_interval_ms,
                        adapter_stat=[os.stat(self.adapter).st_size, os.stat(self.adapter).st_mtime_ns],
                        mmproj_stat=[os.stat(self.config['mmproj']).st_size,
                                     os.stat(self.config['mmproj']).st_mtime_ns])
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        entry = self.cache_dir / key
        manifest = entry / 'batch' / 'manifest.json'
        started = time.monotonic()
        with (self.cache_dir/(key+'.lock')).open('a+b') as cache_lock:
            fcntl.flock(cache_lock.fileno(), fcntl.LOCK_EX)
            if manifest.is_file():
                m = json.loads(manifest.read_text())
                if m.get('format') != 'm31-vision-batch-v1' or m.get('source_pr') != PR_SHA or m.get('backend') != self.selected_backend or m.get('encoder_environment', {}) != self._compute_env or not m.get('encoded'):
                    raise RuntimeError('invalid vision cache manifest')
                for filename, size in [('text_ids.i32', m['rows']*4), ('visual_rows.i32', m['rows']*4), ('positions.i32', m['rows']*4), ('visual.f32', m['visual_rows']*6144*4)]:
                    if (manifest.parent/filename).stat().st_size != size:
                        raise RuntimeError('invalid vision cache array size')
                self.last_stats = dict(cache_hit=True, backend=self.selected_backend, encoder_environment=dict(self._compute_env), seconds=time.monotonic()-started,
                                       rows=m['rows'], visual_rows=m['visual_rows'])
                return manifest
            self._start()
            # Failed/interrupted attempts stay available for diagnosis. A fresh
            # unique directory avoids exposing a partly written cache entry.
            attempt = self.cache_dir / ('attempt-' + key + '-' + uuid.uuid4().hex)
            attempt.mkdir()
            request = dict(prompt=prompt, media=[str(p) for p in paths], encode=True,
                           out=str(attempt/'ipc'), video_fps=video_fps,
                           timestamp_interval_ms=int(timestamp_interval_ms))
            self.process.stdin.write(json.dumps(request) + '\n')
            self.process.stdin.flush()
            reply = self._read()
            if not reply.get('ok'):
                raise ValueError(reply.get('error', 'vision encoding failed'))
            ipc = Path(reply['manifest'])
            assemble(ipc, attempt/'batch')
            m = load(ipc)
            # Artifact manifest retains source metadata filenames and source
            # manifest path. Relocation keeps both source and batch together.
            source_manifest = json.loads((attempt/'batch'/'manifest.json').read_text())
            source_manifest['encoder_environment'] = dict(self._compute_env)
            source_manifest['source_manifest'] = str(entry/'ipc'/'manifest.json')
            (attempt/'batch'/'manifest.json').write_text(json.dumps(source_manifest) + '\n')
            attempt.rename(entry)
            self.last_stats = dict(cache_hit=False, backend=self.selected_backend, encoder_environment=dict(self._compute_env), seconds=time.monotonic()-started,
                                   rows=m['rows'], visual_rows=m['visual_rows'],
                                   encode_seconds=m['encode_seconds'],
                                   cached_image_chunks=m.get('cached_image_chunks', 0))
            return manifest

    def close(self):
        # Read failures close from inside encode/restart while this lock is
        # already held. RLock also makes external unload wait for that request.
        with self.lock:
            self._close_unlocked()

    def _close_unlocked(self):
        process, self.process = self.process, None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            if process.stdin:
                process.stdin.close()
            if process.stdout:
                process.stdout.close()
        if self.log:
            self.log.close()
            self.log = None
        if self.requested_backend == 'auto':
            self.selected_backend = None
            self.backend_selection = dict(requested='auto', backend=None,
                                          min_free_mib=self.gpu_min_free_mib, reason='deferred')

    unload = close

    def restart(self):
        with self.lock:
            self.unload()
            self._start()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
