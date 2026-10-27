#!/usr/bin/env python3
"""CPU-only protocol tests; no subprocess model, GPU or large GGUF needed."""
import importlib.util
import http.client
import json
import os
from pathlib import Path
import threading
import sys
import time
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

    def read(self, timeout=None):
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
        self.assertEqual(engine.writes[-2:], [{"accepted": False, "allowed": []}, {"continue": False}])

    def test_ack_precedes_stream_parse(self):
        engine = FakeEngine([{"event": "token", "token": 10}, done()])
        class CheckingTokenizer(FakeTokenizer):
            def call(self, req):
                if req["op"] == "parse" and req.get("partial"):
                    self_case.assertEqual(engine.writes[-1], {"continue": True})
                return super().call(req)
        self_case = self
        app = server.Application(engine, CheckingTokenizer({10: b"ok"}))
        self.assertEqual(list(app.events({"tokens": [1]}, []))[-1]["message"]["content"], "ok")

    def test_parse_error_after_ack_drains_next_token(self):
        engine = FakeEngine([{"event": "token", "token": 10}, {"event": "token", "token": 11}, done()])
        class BrokenParser(FakeTokenizer):
            def call(self, req):
                if req["op"] == "parse": raise RuntimeError("PEG failed")
                return super().call(req)
        app = server.Application(engine, BrokenParser({10: b"ok", 11: b"next"}))
        with self.assertRaisesRegex(RuntimeError, "PEG failed"):
            list(app.events({"tokens": [1]}, []))
        self.assertEqual(engine.writes[1:], [{"continue": True}, {"continue": False}])
        with self.assertRaises(StopIteration): engine.read()

    def test_nonstream_only_parses_final_and_keeps_stop(self):
        engine = FakeEngine([{"event": "token", "token": 10}, {"event": "token", "token": 11}, done()])
        parses = []
        class CountingParser(FakeTokenizer):
            def call(self, req):
                if req["op"] == "parse": parses.append(req)
                return super().call(req)
        app = server.Application(engine, CountingParser({10: b"abcST", 11: b"OPtail"}))
        events = list(app.events({"tokens": [1]}, ["STOP"], stream_parse=False))
        self.assertEqual(events[-1]["message"]["content"], "abc")
        self.assertEqual(parses, [{"op": "parse", "content": "abc", "partial": False}])
        self.assertEqual(engine.writes[1:], [{"continue": True}, {"continue": False}])

    def test_nonstream_parse_error_after_done_needs_no_drain(self):
        engine = FakeEngine([{"event": "token", "token": 10}, done()])
        class BrokenParser(FakeTokenizer):
            def call(self, req):
                if req["op"] == "parse": raise RuntimeError("PEG failed")
                return super().call(req)
        app = server.Application(engine, BrokenParser({10: b"ok"}))
        with self.assertRaisesRegex(RuntimeError, "PEG failed"):
            list(app.events({"tokens": [1]}, [], stream_parse=False))
        self.assertEqual(engine.writes[1:], [{"continue": True}])
        with self.assertRaises(StopIteration): engine.read()

    def test_reasoning_effort_maps_like_reference_server(self):
        class ChatTokenizer(FakeTokenizer):
            def call(self, req):
                if req["op"] == "chat":
                    self.chat_request = req
                    return {"tokens": [1]}
                return super().call(req)
        tokenizer = ChatTokenizer({})
        app = server.Application(FakeEngine([]), tokenizer)
        base = {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 1}
        original_kwargs = {"reasoning_effort": "high"}
        for effort in ("low", "medium", "high"):
            app.prepare({**base, "reasoning_effort": effort, "chat_template_kwargs": original_kwargs})
            self.assertEqual(tokenizer.chat_request["chat_template_kwargs"]["reasoning_effort"], effort)
        self.assertEqual(original_kwargs, {"reasoning_effort": "high"})
        app.prepare({**base, "reasoning_effort": "", "chat_template_kwargs": original_kwargs})
        self.assertEqual(tokenizer.chat_request["chat_template_kwargs"]["reasoning_effort"], "high")
        app.prepare({**base, "reasoning_effort": "none", "chat_template_kwargs": {"reasoning_effort": "high"}})
        self.assertFalse(tokenizer.chat_request["enable_thinking"])
        self.assertNotIn("reasoning_effort", tokenizer.chat_request["chat_template_kwargs"])
        with self.assertRaises(ValueError): app.prepare({**base, "reasoning_effort": 7})

    def test_sampling_rejects_invalid_numbers_and_context_overflow(self):
        for body in ({"top_p": float("nan")}, {"temperature": -1}, {"min_p": 1.1}, {"top_k": True}, {"max_tokens": -2}, {"max_completion_tokens": True}):
            with self.assertRaises(ValueError):
                server.request_parameters(body, 128, 1)

    def test_prompt_cache_explicit_optout_is_forwarded_when_provided(self):
        with patch.dict(os.environ, {"M31_PROMPT_CACHE_MIB": "1024"}):
            for value in (False, True):
                params, _ = server.request_parameters({"cache_prompt": value}, 128, 1)
                self.assertIs(params["cache_prompt"], value)
            for value in (0, 1, "false", None):
                with self.assertRaisesRegex(ValueError, "cache_prompt"):
                    server.request_parameters({"cache_prompt": value}, 128, 1)
        with patch.dict(os.environ, {"M31_PROMPT_CACHE_MIB": "0"}):
            params, _ = server.request_parameters({"cache_prompt": False}, 128, 1)
            self.assertIs(params["cache_prompt"], False)
            self.assertNotIn("cache_prompt", server.request_parameters({}, 128, 1)[0])

    def test_multimodal_is_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "vision"):
            server.validate_messages([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png,abc"}}]}])

    def test_max_tokens_zero(self):
        params, _ = server.request_parameters({"max_tokens": 0}, 128, 128)
        self.assertEqual(params["max_tokens"], 0)


class SafetyTests(unittest.TestCase):
    def test_grammar_helper_failure_sends_negative_ack_and_drains(self):
        engine = FakeEngine([{"event": "sample", "token": 10},
                             {"event": "error", "error": "empty grammar candidates"},
                             {"event": "token", "token": 11}, done()])
        class BrokenGrammar(FakeTokenizer):
            def call(self, req):
                if req["op"] == "grammar_check":
                    raise RuntimeError("grammar helper failed")
                return super().call(req)
        app = server.Application(engine, BrokenGrammar({11: b"fresh"}))
        with self.assertRaisesRegex(RuntimeError, "grammar helper failed"):
            list(app.events({"tokens": [1]}, []))
        self.assertEqual(engine.writes[-1], {"accepted": False, "allowed": []})
        # The previous terminal error was consumed before unlocking the channel.
        self.assertEqual(list(app.events({"tokens": [2]}, []))[-1]["message"]["content"], "fresh")

    def test_cancel_does_not_call_failed_grammar_helper(self):
        engine = FakeEngine([{"event": "prefill"}, {"event": "sample", "token": 10},
                             {"event": "error", "error": "empty grammar candidates"}])
        class BrokenGrammar(FakeTokenizer):
            def call(self, req):
                raise AssertionError("cancellation must not call tokenizer")
        app = server.Application(engine, BrokenGrammar({}))
        events = app.events({"tokens": [1]}, [])
        next(events)
        events.close()
        self.assertEqual(engine.writes[-1], {"accepted": False, "allowed": []})
        with self.assertRaises(StopIteration): engine.read()

    def test_generation_aliases_and_context_clamping(self):
        cases = [({}, 127), ({"max_tokens": 256}, 127),
                 ({"max_completion_tokens": 7}, 7), ({"max_tokens": -1}, 127),
                 ({"max_tokens": 2, "max_completion_tokens": 3}, 3),
                 ({"max_tokens": 2, "max_completion_tokens": 3, "n_predict": 4}, 4)]
        for body, expected in cases:
            with self.subTest(body=body):
                self.assertEqual(server.request_parameters(body, 128, 1)[0]["max_tokens"], expected)
        with self.assertRaisesRegex(ValueError, "prompt exceeds"):
            server.request_parameters({}, 128, 129)

    def test_disconnected_before_start_does_not_touch_engine(self):
        engine = FakeEngine([])
        app = server.Application(engine, FakeTokenizer({}))
        with self.assertRaises(server.ClientDisconnected):
            list(app.events({"tokens": [1]}, [], cancelled=lambda: True))
        self.assertEqual(engine.writes, [])

    def test_nonstream_disconnect_sends_cancel_and_drains(self):
        engine = FakeEngine([{"event": "token", "token": 10}, done()])
        disconnected = [False]
        class DisconnectingTokenizer(FakeTokenizer):
            def call(self, req):
                disconnected[0] = True
                return super().call(req)
        app = server.Application(engine, DisconnectingTokenizer({10: b"ok"}))
        with self.assertRaises(server.ClientDisconnected):
            list(app.events({"tokens": [1]}, [], stream_parse=False, cancelled=lambda: disconnected[0]))
        self.assertEqual(engine.writes[-1], {"continue": False})
        with self.assertRaises(StopIteration): engine.read()

    def test_inference_queue_wait_is_bounded_and_slot_released(self):
        app = server.Application(FakeEngine([]), FakeTokenizer({}))
        app.queue_timeout = 0.01
        app.lock.acquire()
        try:
            with self.assertRaisesRegex(server.ServerBusy, "timed out"):
                with app.inference_slot(): pass
        finally:
            app.lock.release()
        with app.inference_slot(): pass
        self.assertTrue(app.inference_queue.acquire(blocking=False))
        self.assertTrue(app.inference_queue.acquire(blocking=False))
        try:
            with self.assertRaisesRegex(server.ServerBusy, "full"):
                with app.inference_slot(): pass
        finally:
            app.inference_queue.release()
            app.inference_queue.release()

    def test_rpc_partial_frame_timeout_invalidates_channel(self):
        code = "import sys,time,json;print(json.dumps(dict(event='ready')),flush=True);sys.stdout.write('{');sys.stdout.flush();time.sleep(30)"
        rpc = server.RPC([sys.executable, "-c", code], read_timeout=0.05)
        try:
            start = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                rpc.read()
            self.assertLess(time.monotonic() - start, 1.0)
            self.assertIsNotNone(rpc.failure)
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                rpc.write({"tokens": [1]})
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                rpc.read()
        finally:
            rpc.close()

    def test_rpc_blocked_write_invalidates_channel(self):
        code = "import time,json;print(json.dumps(dict(event='ready')),flush=True);time.sleep(30)"
        rpc = server.RPC([sys.executable, "-c", code], read_timeout=0.05)
        try:
            start = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "write timed out"):
                rpc.write({"content": "x" * 1024 * 1024})
            self.assertLess(time.monotonic() - start, 1.0)
            self.assertIsNotNone(rpc.failure)
        finally:
            rpc.close()

    def test_http_busy_is_503_with_retry_after(self):
        app = server.Application(FakeEngine([]), FakeTokenizer({}))
        app.inference_queue.acquire()
        app.inference_queue.acquire()
        class QuietHandler(server.Handler):
            def log_message(self, *args): pass
        http_server = server.ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
        http_server.app = app
        thread = threading.Thread(target=http_server.handle_request)
        thread.start()
        connection = http.client.HTTPConnection(*http_server.server_address, timeout=1.0)
        try:
            connection.request("POST", "/completion", json.dumps({"prompt": [1]}), {"Connection": "close"})
            response = connection.getresponse()
            self.assertEqual(response.status, 503)
            self.assertEqual(response.getheader("Retry-After"), "1")
            self.assertIn("queue is full", json.loads(response.read())["error"]["message"])
        finally:
            connection.close()
            thread.join(timeout=1)
            http_server.server_close()
            app.inference_queue.release()
            app.inference_queue.release()

    def test_rpc_nonobject_frame_invalidates_channel(self):
        code = "import time,json;print(json.dumps(dict(event='ready')),flush=True);print('[]',flush=True);time.sleep(30)"
        rpc = server.RPC([sys.executable, "-c", code], read_timeout=0.1)
        try:
            with self.assertRaisesRegex(RuntimeError, "must be an object"):
                rpc.read()
            self.assertIsNotNone(rpc.failure)
        finally:
            rpc.close()


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

    def test_stream_nonstream_exact_final_parser_message(self):
        tools = [{"type": "function", "function": {"name": "probe", "parameters": {"type": "object", "properties": {
            "value": {"type": "integer"}}, "required": ["value"]}}}]
        ns = "]<]minimax[>["
        cases = [("<mm:think>Проверяю</mm:think>Ответ café 🐈STOPхвост", ["STOP"], True),
                 (ns + '<tool_call>' + ns + '<invoke name="probe">' + ns + '<value>7' + ns + '</value>' + ns + '</invoke>' + ns + '</tool_call>', [], False)]
        for text, stops, thinking in cases:
            messages = []
            for streaming in (True, False):
                self.chat(tools, thinking=thinking)
                ids = self.helper.call({"op": "tokenize", "content": text, "parse_special": True})["tokens"]
                engine = FakeEngine([*({"event": "token", "token": token} for token in ids), done()])
                app = server.Application(engine, self.helper)
                result = list(app.events({"tokens": [1]}, stops, stream_parse=streaming))[-1]
                message = result["message"]
                for call in message.get("tool_calls", []): call.pop("id", None)
                messages.append((message, result["finish_reason"], result["timings"]))
            self.assertEqual(messages[0], messages[1])

    def test_nonstream_truncated_token_prefixes_match_stream_final(self):
        tools = [{"type": "function", "function": {"name": "probe", "parameters": {"type": "object", "properties": {
            "value": {"type": "integer"}}, "required": ["value"]}}}]
        ns = "]<]minimax[>["
        text = ('<mm:think>Проверяю</mm:think>Ответ ' + ns + '<tool_call>' + ns + '<invoke name="probe">' +
                ns + '<value>7' + ns + '</value>' + ns + '</invoke>' + ns + '</tool_call>')
        ids = self.helper.call({"op": "tokenize", "content": text, "parse_special": True})["tokens"]
        for end in range(1, len(ids) + 1):
            messages = []
            for streaming in (True, False):
                self.chat(tools)
                engine = FakeEngine([*({"event": "token", "token": token} for token in ids[:end]), done()])
                app = server.Application(engine, self.helper)
                result = list(app.events({"tokens": [1]}, [], stream_parse=streaming))[-1]
                message = result["message"]
                for call in message.get("tool_calls", []): call.pop("id", None)
                messages.append((message, result["finish_reason"]))
            self.assertEqual(messages[0], messages[1], (end, ids[:end]))

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
