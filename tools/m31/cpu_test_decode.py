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
        for cached in (0, 1):
            out = root / (tag + f'-cache{cached}')
            cmd = [str(a.engine), '-m', str(model), '--cases', str(cases), '--out', str(out),
                   '--backend', 'cpu', '--ctx', '32', '--chunk', str(chunk), '--threads', '2',
                   '--reuse-allocations', '1', '--persistent-cpu-pool', '1', '--cpu-residual', '1',
                   '--cpu-decode-cache', str(cached), '--taps', taps]
            if tail:
                cmd += ['--decode-tail', str(tail)]
            subprocess.run(cmd, check=True, env=env)
        for case in ('first', 'reset'):
            base = root / (tag + '-cache0') / case
            opt = root / (tag + '-cache1') / case
            assert (base / 'logits.f32').read_bytes() == (opt / 'logits.f32').read_bytes(), (tag, case)
            assert json.loads((opt / 'manifest.json').read_text())['cpu_decode_cache']
        print(f'CPU decode graph cache bit-identical: chunk={chunk} tail={tail} taps={taps}, both reset cases')
