#!/usr/bin/env python3
"""Generate a supported CUDA MSA fixture for decode-cache bucket/reset gates.

Original tiny HD16/block4 is deliberately retained by CPU math tests. CUDA FA
needs a supported head size and gathered-key shape, so this fixture uses HD64,
block128 and two selected blocks. Weights remain deterministic synthetic F32.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).parent / 'tests'))
import make_fixture as fixture

p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
a.out.mkdir(exist_ok=False)
fixture.CTX = 640
fixture.HD = 64
fixture.ROT = 32
fixture.BLOCK = 128
fixture.TOP_BLOCKS = 2
fixture.make_fixture(a.out / 'tiny.gguf')
ids = a.out / 'ids'
ids.mkdir()
rng = np.random.default_rng(901)
for case, count in [('first', 520), ('reset', 259)]:
    rng.integers(0, 64, count, dtype=np.int32).astype('<i4').tofile(ids / (case + '.i32'))
print(a.out)
