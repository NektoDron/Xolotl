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

    def test_bounded_head_contract_mismatch_refused(self):
        candidate = self.cand / 'tiny/manifest.json'
        manifest = json.loads(candidate.read_text())
        manifest['head_tail'] = 64
        candidate.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'head_tail differs'):
            compare_suite(self.ref, self.cand)
        reference = self.ref / 'tiny/manifest.json'
        manifest = json.loads(reference.read_text())
        manifest['head_tail'] = 64
        manifest['full_last_layer_before_crop'] = True
        reference.write_text(json.dumps(manifest))
        result = compare_suite(self.ref, self.cand)
        self.assertTrue(result['pass'])
        self.assertEqual(result['scope'], 'diagnostic')
        self.assertEqual(result['cases'][0]['head_tail'], 64)

    def test_bounded_head_chunk_geometry_refused(self):
        for root, chunks in ((self.ref, [{'position': 0, 'tokens': 3}]),
                             (self.cand, [{'position': 0, 'tokens': 2}, {'position': 2, 'tokens': 1}])):
            path = root / 'tiny/manifest.json'
            manifest = json.loads(path.read_text())
            manifest.update(head_tail=64, actual_chunks=chunks, full_last_layer_before_crop=True)
            path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'actual chunk geometry differs'):
            compare_suite(self.ref, self.cand)

    def test_partial_actual_chunk_coverage_refused(self):
        path = self.cand / 'tiny/manifest.json'
        manifest = json.loads(path.read_text())
        manifest['actual_chunks'] = [{'position': 0, 'tokens': 2}]
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'actual chunks do not cover'):
            compare_suite(self.ref, self.cand)

    def test_explicit_last_layer_crop_policy_refused(self):
        for root, full in ((self.ref, True), (self.cand, False)):
            path = root / 'tiny/manifest.json'
            manifest = json.loads(path.read_text())
            manifest['full_last_layer_before_crop'] = full
            path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'last-layer crop policy'):
            compare_suite(self.ref, self.cand)

    def test_legacy_bounded_oracle_crop_is_not_native_full_layer(self):
        for root in (self.ref, self.cand):
            path = root / 'tiny/manifest.json'
            manifest = json.loads(path.read_text())
            manifest['head_tail'] = 64
            path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'last-layer crop policy differs'):
            compare_suite(self.ref, self.cand)

    def test_gpu_offload_oracle_is_not_strict_m1(self):
        path = self.ref / 'tiny/manifest.json'
        manifest = json.loads(path.read_text())
        manifest['oracle_op_offload'] = True
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'strict CPU expert oracle'):
            compare_suite(self.ref, self.cand, require_suite=True)

    def test_full_gate_requires_same_actual_chunks(self):
        for root, chunks in ((self.ref, [{'position': 0, 'tokens': 3}]),
                             (self.cand, [{'position': 0, 'tokens': 2}, {'position': 2, 'tokens': 1}])):
            path = root / 'tiny/manifest.json'
            manifest = json.loads(path.read_text())
            manifest['actual_chunks'] = chunks
            path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'matching actual chunk geometry'):
            compare_suite(self.ref, self.cand, require_suite=True)

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
