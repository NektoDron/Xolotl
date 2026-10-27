#!/usr/bin/env python3
"""Small CPU protocol coverage for opt-in cache counters and HTTP usage."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("prefix_api", Path(__file__).parents[1] / "serve_m31.py")
server = importlib.util.module_from_spec(spec); spec.loader.exec_module(server)

class Engine:
    ready = {"context": 128, "vocab": 64}
    def __init__(self, events): self.events = iter(events); self.writes = []
    def write(self, value): self.writes.append(value)
    def read(self, timeout=None): return next(self.events)

class Tokenizer:
    ready = {"vocab": 64}
    def call(self, req):
        if req["op"] == "parse": return {"message": {"role": "assistant", "content": req["content"]}}
        raise AssertionError(req)

class PrefixAPI(unittest.TestCase):
    def test_default_usage_keeps_v8_fields(self):
        self.assertEqual(server.usage_counts({"prompt_n": 8, "predicted_n": 3}),
                         {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11})
    def test_cached_usage_discloses_reuse(self):
        self.assertEqual(server.usage_counts({"prompt_n": 8, "predicted_n": 3, "cached_n": 8})["prompt_tokens_details"],
                         {"cached_tokens": 8})
    def test_reused_logical_tokens_are_not_reported_as_compute(self):
        engine = Engine([{"event": "prefill", "cache_hit": True, "cached_n": 8, "prompt_eval_n": 0},
                         {"event": "done", "finish_reason": "length", "timings": {"prompt_n": 8, "predicted_n": 0,
                          "prompt_ms": .1, "prompt_per_second": 80000}}])
        result = list(server.Application(engine, Tokenizer()).events({"tokens": [1]}, []))[-1]
        self.assertEqual(result["timings"]["prompt_per_second"], 0)
        self.assertEqual(result["timings"]["cached_n"], 8)
        self.assertEqual(result["timings"]["prompt_eval_n"], 0)
        self.assertTrue(result["timings"]["cache_hit"])
    def test_cache_prompt_is_strict_boolean_and_default_omitted(self):
        self.assertNotIn("cache_prompt", server.request_parameters({}, 128, 8)[0])
        self.assertFalse(server.request_parameters({"cache_prompt": False}, 128, 8)[0]["cache_prompt"])
        for value in (0, 1, "false", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                server.request_parameters({"cache_prompt": value}, 128, 8)

if __name__ == "__main__": unittest.main()
