"""Synthetic CPU tests for mixed-row preflight; no model/vision inference."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

from vision_protocol import EMBEDDING_WIDTH, PR_SHA, load_mixed_rows


class MixedProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="m31-vision-protocol-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "manifest.json"
        self.manifest = {
            "format": "m31-vision-batch-v1", "source_pr": PR_SHA,
            "n_embd": EMBEDDING_WIDTH, "encoded": True, "llm_inference": False,
            "native_engine_validated": False, "rows": 4, "visual_rows": 2,
            "chunks": [
                {"index": 0, "row": 0, "rows": 1, "position": 0, "n_pos": 1,
                 "type": "text", "tokens": [5]},
                {"index": 1, "row": 1, "rows": 2, "position": 1, "n_pos": 2,
                 "type": "image", "visual_row": 0,
                 "decoder_positions": {"start": 1, "step": 1, "axes": 1}},
                {"index": 2, "row": 3, "rows": 1, "position": 3, "n_pos": 1,
                 "type": "text", "tokens": [17]},
            ],
        }
        self.write_manifest()
        self.write_i32("text_ids.i32", [5, -1, -1, 17])
        self.write_i32("visual_rows.i32", [-1, 0, 1, -1])
        self.write_i32("positions.i32", [0, 1, 2, 3])
        (self.root / "visual.f32").write_bytes(b"\0" * (2 * EMBEDDING_WIDTH * 4))

    def write_manifest(self):
        self.path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def write_i32(self, name, values):
        (self.root / name).write_bytes(struct.pack("<" + "i" * len(values), *values))

    def load(self, **kwargs):
        return load_mixed_rows(self.path, vocab_size=200064, max_context=40000, **kwargs)

    def test_valid_mixed_rows(self):
        batch = self.load()
        self.assertEqual(batch.rows, 4)
        self.assertEqual(batch.visual_count, 2)
        self.assertEqual(batch.token_ids, (5, -1, -1, 17))
        self.assertEqual(batch.embedding_width, 6144)
        self.assertFalse(self.manifest["native_engine_validated"])

    def test_duplicate_visual_mapping(self):
        self.write_i32("visual_rows.i32", [-1, 0, 0, -1])
        with self.assertRaisesRegex(ValueError, "ordered, and unique"):
            self.load()

    def test_out_of_vocabulary(self):
        self.write_i32("text_ids.i32", [5, -1, -1, 200064])
        with self.assertRaisesRegex(ValueError, "in-vocabulary"):
            self.load()

    def test_unmapped_negative_token(self):
        self.write_i32("text_ids.i32", [5, -2, -1, 17])
        with self.assertRaises(ValueError):
            self.load()

    def test_positions_are_not_a_qwen_grid(self):
        self.write_i32("positions.i32", [0, 1, 1, 2])
        with self.assertRaisesRegex(ValueError, "contiguous linear"):
            self.load()

    def test_incomplete_encoder_output(self):
        self.manifest["encoded"] = False
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "encoded input"):
            self.load()

    def test_truncated_embeddings(self):
        (self.root / "visual.f32").write_bytes(b"\0" * 4)
        with self.assertRaisesRegex(ValueError, "exactly"):
            self.load()

    def test_nonfinite_embeddings(self):
        with (self.root / "visual.f32").open("r+b") as stream:
            stream.write(struct.pack("<f", float("nan")))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.load()

    def test_manifest_and_binary_tokens_must_match(self):
        self.manifest["chunks"][0]["tokens"] = [6]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "binary token IDs"):
            self.load()

    def test_non_linear_image_position_contract(self):
        self.manifest["chunks"][1]["decoder_positions"]["axes"] = 4
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "linear visual mapping"):
            self.load()

    def test_context_limit_before_binary_read(self):
        with self.assertRaisesRegex(ValueError, "rows must be"):
            load_mixed_rows(self.path, vocab_size=200064, max_context=3)

    def test_wrong_source_revision(self):
        self.manifest["source_pr"] = "unverified"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "source PR"):
            self.load()

    def test_boolean_is_not_a_row_index(self):
        self.manifest["chunks"][0]["row"] = False
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "chunk row must be"):
            self.load()


if __name__ == "__main__":
    unittest.main()
