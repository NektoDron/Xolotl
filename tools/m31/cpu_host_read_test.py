#!/usr/bin/env python3
"""Generate a small CPU model, compare bits, and verify actual buffer growth reuse."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).parent/'tests'))
import make_fixture as fixture
p=argparse.ArgumentParser()
p.add_argument('--engine-check',type=Path,required=True)
a=p.parse_args()
with tempfile.TemporaryDirectory(prefix='m31-host-reads-') as temp:
    model=Path(temp)/'tiny.gguf'
    fixture.CTX=4096;fixture.D=16;fixture.L=3;fixture.H=2;fixture.HKV=1
    fixture.HD=16;fixture.ROT=8;fixture.FF=32;fixture.EF=32
    fixture.BLOCK=128;fixture.TOP_BLOCKS=2
    fixture.make_fixture(model)
    result=subprocess.run([str(a.engine_check),str(model)],capture_output=True,text=True,
                          env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2'},timeout=240)
    sys.stdout.write(result.stdout);sys.stderr.write(result.stderr)
    if result.returncode:raise SystemExit(result.returncode)
    rows=[dict(re.findall(r'(\w+)=(-?\d+)',line)) for line in result.stderr.splitlines() if line.startswith('M31 host-reads ')]
    assert rows,'missing reuse evidence'
    assert any(r['n']=='2048' and int(r['growths'])>0 for r in rows),'shape growth was not observed'
    assert any(r['n']=='2048' and r['growths']=='0' and int(r['reuses'])>0 for r in rows),'no capacity reuse across reset'
    assert all(r['growths']=='0' for r in rows if r['n']=='1'),'decode unexpectedly allocated after prefill'
    assert 'M31 host read reuse: enabled=0' in result.stderr,'missing taps/no-reuse fallback evidence'
    print('PASS allocation evidence: initial/growing vectors allocate; same-capacity layers/reset reuse; decode no growth')
