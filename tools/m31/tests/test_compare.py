#!/usr/bin/env python3
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_logits import compare_suite


class CompareTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="m31-compare-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ref, self.cand = self.root / "ref", self.root / "cand"
        for root, engine in ((self.ref, "llama.cpp-PR31192"), (self.cand, "xolotl-m31-correctness")):
            case = root / "tiny"
            case.mkdir(parents=True)
            meta = dict(format="m31-logits-v1", engine=engine, model="same-model", kv="f16",
                        tokens=3, vocab=4, positions=[0, 1, 2])
            (case / "manifest.json").write_text(json.dumps(meta))
            np.array([1, 2, 3], dtype="<i4").tofile(case / "tokens.i32")
            np.array([[1, 0, -1, 2], [3, 0, 1, 2], [1, 5, 0, -1]], dtype="<f4").tofile(case / "logits.f32")

    def test_identity_and_logit_shift(self):
        result = compare_suite(self.ref, self.cand)
        self.assertTrue(result["pass"])
        path = self.cand / "tiny/logits.f32"
        values = np.fromfile(path, dtype="<f4")
        (values + 10).tofile(path)
        result = compare_suite(self.ref, self.cand)
        self.assertTrue(result["pass"])
        self.assertLess(result["mean_kl"], 1e-12)

    def test_token_mismatch_refused(self):
        np.array([1, 2, 4], dtype="<i4").tofile(self.cand / "tiny/tokens.i32")
        with self.assertRaisesRegex(ValueError, "token IDs differ"):
            compare_suite(self.ref, self.cand)

    def test_corruption_and_nonfinite(self):
        path = self.cand / "tiny/logits.f32"
        values = np.fromfile(path, dtype="<f4")
        values[0] = np.nan
        values.tofile(path)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            compare_suite(self.ref, self.cand)
        path.write_bytes(b"short")
        with self.assertRaisesRegex(ValueError, "truncated"):
            compare_suite(self.ref, self.cand)

    def test_wrong_result_fails_and_partial_suite_refused(self):
        path = self.cand / "tiny/logits.f32"
        values = np.fromfile(path, dtype="<f4").reshape(3, 4)
        values[:, 3] += 30
        values.tofile(path)
        self.assertFalse(compare_suite(self.ref, self.cand)["pass"])
        with self.assertRaisesRegex(ValueError, "exactly 20"):
            compare_suite(self.ref, self.cand, require_suite=True)


if __name__ == "__main__":
    unittest.main()
