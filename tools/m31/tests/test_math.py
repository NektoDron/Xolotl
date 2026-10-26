#!/usr/bin/env python3
"""Small independent math/loader/chunk check. Defers while the real reference unit is active."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from make_fixture import make_fixture
from numpy_reference import forward


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--inspect", type=Path, required=True)
    a = p.parse_args()
    if Path("/run/systemd/system").exists():
        ref = subprocess.run(["systemctl", "is-active", "-q", "codex-m31-ref"], check=False)
        if ref.returncode == 0:
            print("codex-m31-ref active: no tests or inference")
            return 77
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2"}
    with tempfile.TemporaryDirectory(prefix="m31-math-") as tmp:
        root = Path(tmp)
        model = root / "tiny.gguf"
        make_fixture(model)
        subprocess.run([str(a.inspect), str(model)], check=True, env=env)
        for kind, kwargs in (("missing-index", dict(missing_dense_index=True)),
                             ("unknown-tensor", dict(unknown_tensor=True))):
            bad = root / f"{kind}.gguf"
            make_fixture(bad, **kwargs)
            test = subprocess.run([str(a.inspect), str(bad)], env=env, capture_output=True, text=True)
            if test.returncode == 0:
                raise AssertionError(f"accepted invalid {kind} fixture")
        ids = np.random.default_rng(64).integers(0, 64, 29, dtype=np.int32)
        tokens = root / "tiny.i32"
        ids.astype("<i4").tofile(tokens)
        expected = forward(model, ids)
        got = []
        for chunk in (1, 7, 29):
            out = root / f"chunk{chunk}"
            subprocess.run([str(a.engine), "-m", str(model), "--ids-file", str(tokens), "--out", str(out),
                            "--backend", "cpu", "--ctx", "32", "--chunk", str(chunk), "--threads", "2"],
                           check=True, env=env)
            run = out / "tiny"
            meta = json.loads((run / "manifest.json").read_text())
            if meta["positions"] != list(range(len(ids))):
                raise AssertionError("not all logits were emitted")
            values = np.fromfile(run / "logits.f32", dtype="<f4").reshape(len(ids), 64)
            if not np.isfinite(values).all():
                raise AssertionError("non-finite logits")
            # CPU FA may use F16 query/output accumulation; this checks architecture, not bit parity.
            err = float(np.max(np.abs(values - expected)))
            print(f"chunk={chunk}: max abs versus independent NumPy={err:.6g}", flush=True)
            np.testing.assert_allclose(values, expected, rtol=.005, atol=.005)
            got.append(values)
        for values in got[1:]:
            np.testing.assert_allclose(values, got[0], rtol=.005, atol=.005)
        packed = root / "packed"
        subprocess.run([str(a.engine), "-m", str(model), "--ids-file", str(tokens), "--out", str(packed),
                        "--backend", "cpu", "--ctx", "32", "--chunk", "7", "--threads", "2",
                        "--arena-gib", ".01"], check=True, env=env)
        values = np.fromfile(packed / "tiny/logits.f32", dtype="<f4").reshape(len(ids), 64)
        np.testing.assert_array_equal(values, got[1])
        print("packed arena: logits bit-identical to mapped weights", flush=True)
        insufficient = subprocess.run([str(a.engine), "-m", str(model), "--ids-file", str(tokens),
            "--out", str(root / "insufficient"), "--backend", "cpu", "--ctx", "32", "--arena-gib", ".000001"],
            env=env, capture_output=True, text=True)
        if insufficient.returncode == 0 or "budget is smaller" not in insufficient.stderr:
            raise AssertionError("undersized arena was not rejected before allocation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
