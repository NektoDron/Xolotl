#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).parent/'tests'))
import make_fixture as fixture
p=argparse.ArgumentParser()
p.add_argument('--check',type=Path,required=True)
a=p.parse_args()
with tempfile.TemporaryDirectory(prefix='m31-shared-overlap-') as temp:
    model=Path(temp)/'tiny.gguf'
    fixture.CTX=640;fixture.HD=64;fixture.ROT=32;fixture.BLOCK=128;fixture.TOP_BLOCKS=2
    fixture.make_fixture(model)
    result=subprocess.run([str(a.check),str(model),'cpu'],capture_output=True,text=True,
                          env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2'},timeout=180)
    sys.stdout.write(result.stdout);sys.stderr.write(result.stderr)
    if result.returncode:raise SystemExit(result.returncode)
    assert 'M31 CPU/shared overlap: enabled=0' in result.stderr
    assert 'M31 shared-overlap pos=' not in result.stderr
    print('PASS CPU fallback has no async shared launches')
