#!/usr/bin/env python3
"""Check persistent MoE graphs across changed routes, chunk sizes and reset()."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
sys.path.insert(0, str(Path(__file__).parent / 'tests'))
import make_fixture as fixture
from make_fixture import make_fixture

p = argparse.ArgumentParser()
p.add_argument('--engine', type=Path, required=True)
a = p.parse_args()
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2')
with tempfile.TemporaryDirectory(prefix='m31-cpu-decode-') as tmp:
    root = Path(tmp)
    model = root / 'tiny.gguf'
    make_fixture(model)
    cases = root / 'ids'
    cases.mkdir()
    rng = np.random.default_rng(1009)
    for label, count in [('first', 29), ('reset', 17)]:
        rng.integers(0, 64, count, dtype=np.int32).astype('<i4').tofile(cases / (label + '.i32'))
    for chunk, tail, taps in [(1, 0, 'none'), (7, 11, 'none'), (1, 0, 'all')]:
        tag = f'{chunk}-{tail}-{taps}'
        for cached, dense in ((0, 0), (1, 0), (1, 1)):
            out = root / (tag + f'-cache{cached}-dense{dense}')
            cmd = [str(a.engine), '-m', str(model), '--cases', str(cases), '--out', str(out),
                   '--backend', 'cpu', '--ctx', '32', '--chunk', str(chunk), '--threads', '2',
                   '--reuse-allocations', '1', '--persistent-cpu-pool', '1', '--cpu-residual', '1',
                   '--cpu-decode-cache', str(cached), '--dense-decode-cache', str(dense), '--taps', taps]
            if tail:
                cmd += ['--decode-tail', str(tail)]
            subprocess.run(cmd, check=True, env=env)
        for case in ('first', 'reset'):
            base = root / (tag + '-cache0-dense0') / case
            for dense in (0, 1):
                opt = root / (tag + f'-cache1-dense{dense}') / case
                assert (base / 'logits.f32').read_bytes() == (opt / 'logits.f32').read_bytes(), (tag, case, dense)
                assert json.loads((opt / 'manifest.json').read_text())['cpu_decode_cache']
        print(f'CPU decode graph cache bit-identical: chunk={chunk} tail={tail} taps={taps}, both reset cases')

    # Cross both 256-cell buckets on one engine and then reset to a shorter prompt.
    fixture.CTX = 640
    long_model = root / 'long.gguf'
    make_fixture(long_model)
    long_cases = root / 'long-ids'
    long_cases.mkdir()
    for label, count in [('first', 520), ('reset', 259)]:
        rng.integers(0, 64, count, dtype=np.int32).astype('<i4').tofile(long_cases / (label + '.i32'))
    for dense in (0, 1):
        cmd = [str(a.engine), '-m', str(long_model), '--cases', str(long_cases), '--out', str(root / f'long-dense{dense}'),
               '--backend', 'cpu', '--ctx', '640', '--chunk', '128', '--threads', '2', '--decode-tail', '300',
               '--reuse-allocations', '1', '--persistent-cpu-pool', '1', '--cpu-residual', '1',
               '--cpu-decode-cache', '1', '--dense-decode-cache', str(dense)]
        subprocess.run(cmd, check=True, env=env)
    for case in ('first', 'reset'):
        assert (root / 'long-dense0' / case / 'logits.f32').read_bytes() == (root / 'long-dense1' / case / 'logits.f32').read_bytes(), case
    print('Dense/indexed KV cache: bit-identical across 256/512 buckets and reset')
