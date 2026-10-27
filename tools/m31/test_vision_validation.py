"""Tests for semantic checks and SSE timing without starting model processes."""
import json
import threading
from pathlib import Path
import tempfile
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from vision_validation import check, request
from vision_numeric_suite import mixed_inputs


class ValidationTests(unittest.TestCase):
    def test_gpu_summary_selects_sibling_mixed_batch_and_rejects_cpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            results = []
            for label in ("image", "video"):
                entry = directory / label
                (entry / "ipc").mkdir(parents=True)
                (entry / "batch").mkdir()
                (entry / "ipc/manifest.json").write_text("{}")
                (entry / "batch/manifest.json").write_text('{"backend":"cuda"}')
                results.append({"case": label, "single_ipc": str(entry / "ipc/manifest.json"), "single_vs_batch": {"pass": True}})
            path = directory / "summary.json"
            path.write_text(json.dumps({"single_batch_bitwise_pass": True, "results": results}))
            self.assertEqual(mixed_inputs(path)["video"], directory / "video/batch/manifest.json")
            (directory / "video/batch/manifest.json").write_text('{"backend":"cpu"}')
            with self.assertRaisesRegex(ValueError, "CUDA backend"):
                mixed_inputs(path)

    def test_video_order_cannot_be_satisfied_by_unordered_facts(self):
        case = {"expect": ["red", "green", "blue"], "ordered": ["red", "green", "blue"]}
        self.assertTrue(check(case, "red circle then green square then blue triangle")["pass"])
        self.assertFalse(check(case, "blue triangle then red circle then green square")["pass"])

    def test_digit_is_not_substring(self):
        case = {"expect": [r"\b1\b|\bone\b"]}
        self.assertTrue(check(case, "The digit is 1.")["pass"])
        self.assertFalse(check(case, "There are 12 circles.")["pass"])

    def test_stream_first_reasoning_and_visible_tokens(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.end_headers()
                for delta in ({"role": "assistant"}, {"reasoning_content": "think"}, {"content": "red circle"}):
                    self.wfile.write(("data: " + json.dumps({"choices": [{"delta": delta}]}) + "\n\n").encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
            def log_message(self, *args):
                pass
        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = request(f"http://127.0.0.1:{server.server_port}", {}, 5)
            self.assertEqual(result["content"], "red circle")
            self.assertEqual(result["reasoning_content"], "think")
            self.assertGreaterEqual(result["visible_ttft_s"], result["ttft_s"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
