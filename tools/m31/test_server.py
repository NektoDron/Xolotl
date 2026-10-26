#!/usr/bin/env python3
"""CPU-only protocol tests; no subprocess model, GPU or large GGUF needed."""
import importlib.util
import json
import os
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("serve_m31", Path(__file__).with_name("serve_m31.py"))
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class FakeEngine:
    ready = {"context": 128, "vocab": 200064}

    def __init__(self, events):
        self.queue = iter(events)
        self.writes = []

    def write(self, value):
        self.writes.append(value)

    def read(self):
        return next(self.queue)


class FakeTokenizer:
    ready = {"vocab": 200064, "eog": [2]}

    def __init__(self, pieces):
        self.pieces = pieces

    def call(self, req):
        if req["op"] == "piece":
            return {"bytes": list(self.pieces[req["token"]])}
        if req["op"] == "parse":
            return {"message": {"role": "assistant", "content": req["content"]}}
        raise AssertionError(req)


def done():
    return {"event": "done", "finish_reason": "length", "timings": {"prompt_n": 1, "predicted_n": 2}}


class ProtocolTests(unittest.TestCase):
    def test_stop_across_utf8_token_boundary(self):
        engine = FakeEngine([{"event": "token", "token": 10}, {"event": "token", "token": 11}, done()])
        app = server.Application(engine, FakeTokenizer({10: b"abc\xe2\x82", 11: b"\xacSTOPtail"}))
        events = list(app.events({"tokens": [1]}, ["STOP"]))
        self.assertEqual("".join(e["delta"].get("content", "") for e in events if e["event"] == "delta"), "abc€")
        self.assertEqual(events[-1]["message"]["content"], "abc€")
        self.assertEqual(engine.writes[-1], {"continue": False})

    def test_partial_stop_released_at_eof(self):
        engine = FakeEngine([{"event": "token", "token": 10}, done()])
        app = server.Application(engine, FakeTokenizer({10: b"abcST"}))
        events = list(app.events({"tokens": [1]}, ["STOP"]))
        deltas = [e["delta"]["content"] for e in events if e["event"] == "delta"]
        self.assertEqual(deltas, ["abc", "ST"])
        self.assertEqual(events[-1]["message"]["content"], "abcST")

    def test_cancellation_acknowledges_and_drains(self):
        engine = FakeEngine([{"event": "token", "token": 10}, {"event": "token", "token": 11}, done()])
        app = server.Application(engine, FakeTokenizer({10: b"abc", 11: b"def"}))
        events = app.events({"tokens": [1]}, [])
        self.assertEqual(next(events)["event"], "delta")
        events.close()
        self.assertEqual(engine.writes[-1], {"continue": False})
        with self.assertRaises(StopIteration):
            engine.read()

    def test_engine_error_consumed_without_wait_for_done(self):
        engine = FakeEngine([{"event": "error", "error": "bad token"}])
        app = server.Application(engine, FakeTokenizer({}))
        with self.assertRaisesRegex(RuntimeError, "bad token"):
            list(app.events({"tokens": [-1]}, []))
        self.assertEqual(len(engine.writes), 1)

    def test_grammar_proposal_and_resample_handshake(self):
        engine = FakeEngine([{"event": "sample", "token": 99}, {"event": "sample", "token": 10},
                             {"event": "token", "token": 10}, done()])
        class GrammarTokenizer(FakeTokenizer):
            def call(self, req):
                if req["op"] == "grammar_check":
                    return {"accepted": True} if req["token"] == 10 else {"accepted": False, "allowed": [10]}
                return super().call(req)
        app = server.Application(engine, GrammarTokenizer({10: b"ok"}))
        events = list(app.events({"tokens": [1], "grammar_check": True}, []))
        self.assertEqual(engine.writes[1:3], [{"accepted": False, "allowed": [10]}, {"accepted": True}])
        self.assertEqual(events[-1]["message"]["content"], "ok")

    def test_cancellation_drains_pending_grammar_proposal(self):
        engine = FakeEngine([{"event": "prefill"}, {"event": "sample", "token": 10},
                             {"event": "token", "token": 10}, done()])
        class GrammarTokenizer(FakeTokenizer):
            def call(self, req):
                if req["op"] == "grammar_check": return {"accepted": True}
                return super().call(req)
        app = server.Application(engine, GrammarTokenizer({10: b"ok"}))
        events = app.events({"tokens": [1], "grammar_check": True}, [])
        self.assertEqual(next(events)["event"], "prefill")
        events.close()
        self.assertEqual(engine.writes[-2:], [{"accepted": True}, {"continue": False}])

    def test_sampling_rejects_invalid_numbers_and_context_overflow(self):
        for body in ({"top_p": float("nan")}, {"temperature": -1}, {"min_p": 1.1}, {"top_k": True}, {"max_tokens": 129}):
            with self.assertRaises(ValueError):
                server.request_parameters(body, 128, 1)

    def test_multimodal_is_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "vision"):
            server.validate_messages([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png,abc"}}]}])

    def test_max_tokens_zero(self):
        params, _ = server.request_parameters({"max_tokens": 0}, 128, 128)
        self.assertEqual(params["max_tokens"], 0)


@unittest.skipUnless(os.environ.get("M31_TEST_TOKENIZER") and os.environ.get("M31_TEST_MODEL"),
                     "set M31_TEST_TOKENIZER and M31_TEST_MODEL for real vocab-only checks")
class RealTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helper = server.RPC([os.environ["M31_TEST_TOKENIZER"], os.environ["M31_TEST_MODEL"]])

    @classmethod
    def tearDownClass(cls):
        cls.helper.close()

    def chat(self, tools=None, thinking=True, messages=None):
        return self.helper.call({"op": "chat", "messages": messages or [{"role": "user", "content": "Привет!"}],
                                 "tools": tools or [], "enable_thinking": thinking,
                                 "parallel_tool_calls": True})

    def parse(self, content, partial=False):
        return self.helper.call({"op": "parse", "content": content, "partial": partial})["message"]

    def test_nfc_roundtrip_and_literal_special_token(self):
        composed = self.helper.call({"op": "tokenize", "content": "Привет café 🐈"})["tokens"]
        decomposed = self.helper.call({"op": "tokenize", "content": "Привет cafe\u0301 🐈"})["tokens"]
        self.assertEqual(composed, decomposed)
        self.assertEqual(self.helper.call({"op": "detokenize", "tokens": composed})["content"], "Привет café 🐈")
        literal = self.helper.call({"op": "tokenize", "content": "]~b]", "parse_special": False})["tokens"]
        special = self.helper.call({"op": "tokenize", "content": "]~b]", "parse_special": True})["tokens"]
        self.assertNotEqual(literal, special)

    def test_think_content_and_disabled_parser(self):
        rendered = self.chat()
        self.assertTrue(rendered["prompt"].startswith("]~!b[]~b]system\n"))
        self.assertTrue(rendered["prompt"].endswith("]~b]ai\n"))
        self.assertEqual(rendered["format"], "peg-minimax-m3")
        message = self.parse("<mm:think>Проверяю</mm:think>Ответ")
        self.assertEqual(message.get("reasoning_content"), "Проверяю")
        self.assertEqual(message.get("content"), "Ответ")
        self.chat(thinking=False)
        message = self.parse("<mm:think>Hidden</mm:think>Visible")
        self.assertFalse(message.get("reasoning_content"))
        self.assertEqual(message.get("content"), "Visible")

    def test_reasoning_parser_streaming_prefixes(self):
        self.chat()
        text = "<mm:think>тест</mm:think>Ответ"
        previous = {"content": "", "reasoning_content": ""}
        for index in range(1, len(text) + 1):
            message = self.parse(text[:index], partial=True)
            for key in ("content", "reasoning_content"):
                self.assertTrue((message.get(key) or "").startswith(previous.get(key) or ""), (index, previous, message))
            previous = message
        self.assertEqual(previous["content"], "Ответ")

    def grammar_tokens(self, text, accept=True):
        ids = self.helper.call({"op": "tokenize", "content": text, "parse_special": True})["tokens"]
        for token in ids:
            result = self.helper.call({"op": "grammar_check", "token": token, "accept": accept})
            self.assertTrue(result["accepted"], (text, token, result))
        return ids

    def test_lazy_grammar_allows_text_reasoning_and_resets(self):
        tools = [{"type": "function", "function": {"name": "probe", "parameters": {"type": "object", "properties": {}}}}]
        rendered = self.chat(tools)
        self.assertTrue(rendered["grammar_active"])
        self.assertTrue(rendered["grammar_lazy"])
        ns = "]<]minimax[>["
        # A tool trigger within reasoning must not activate tool grammar.
        self.grammar_tokens('<mm:think>' + ns + '<tool_call>' + ns + '<invoke name="missing">рассуждение</mm:think>Ответ без вызова')
        self.chat(tools)
        self.grammar_tokens("Простой ответ 🐈 без инструментов")
        self.assertFalse(self.chat()["grammar_active"])

    def test_lazy_grammar_nested_unicode_and_multiple_calls(self):
        tools = [{"type": "function", "function": {"name": "probe", "parameters": {"type": "object", "properties": {
            "obj": {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]},
            "items": {"type": "array", "items": {"type": "string"}}}, "required": ["obj", "items"]}}}]
        ns = "]<]minimax[>["
        invocation = (ns + '<invoke name="probe">' + ns + '<obj>' + ns + '<value>7' + ns + '</value>' + ns + '</obj>' +
                      ns + '<items>' + ns + '<item>один 🐈' + ns + '</item>' + ns + '</items>' + ns + '</invoke>')
        text = ns + '<tool_call>' + invocation + invocation + ns + '</tool_call>'
        self.chat(tools, thinking=False)
        self.grammar_tokens(text)
        message = self.parse(text)
        self.assertEqual(len(message["tool_calls"]), 2)
        self.assertEqual(json.loads(message["tool_calls"][0]["function"]["arguments"]), {"obj": {"value": 7}, "items": ["один 🐈"]})
        # A new request resets the finished grammar. Reject an unknown function name
        # and return real PR-filtered candidates for resampling.
        self.chat(tools, thinking=False)
        self.grammar_tokens(ns + '<tool_call>' + ns + '<invoke name="')
        bad = self.helper.call({"op": "tokenize", "content": "DefinitelyMissingTool", "parse_special": True})["tokens"][0]
        result = self.helper.call({"op": "grammar_check", "token": bad})
        self.assertFalse(result["accepted"])
        self.assertTrue(result["allowed"])
        self.assertNotIn(bad, result["allowed"])
        self.grammar_tokens('probe">')

    def test_nested_xml_tool_and_continuation_template(self):
        tools = [{"type": "function", "function": {"name": "probe", "description": "test",
                  "parameters": {"type": "object", "properties": {
                      "obj": {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]},
                      "items": {"type": "array", "items": {"type": "string"}}}, "required": ["obj", "items"]}}}]
        self.chat(tools)
        ns = "]<]minimax[>["
        text = ("</mm:think>" + ns + "<tool_call>" + ns + '<invoke name="probe">' +
                ns + "<obj>" + ns + "<value>7" + ns + "</value>" + ns + "</obj>" +
                ns + "<items>" + ns + "<item>one" + ns + "</item>" + ns + "<item>два" + ns + "</item>" +
                ns + "</items>" + ns + "</invoke>" + ns + "</tool_call>")
        message = self.parse(text)
        call = message["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "probe")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"obj": {"value": 7}, "items": ["one", "два"]})
        call["id"] = "call_test"
        messages = [{"role": "user", "content": "Проверь"}, {**message, "role": "assistant"},
                    {"role": "tool", "tool_call_id": "call_test", "content": "ok"}]
        rendered = self.chat(tools, messages=messages)
        self.assertIn("]~b]tool\n<response>ok</response>[e~[", rendered["prompt"])


if __name__ == "__main__":
    unittest.main()
