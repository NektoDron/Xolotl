#!/usr/bin/env python3
"""Small CPU-only tests of the opt-in common-server M31 bridge."""
import contextlib
import http.client
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("m31_common_shim", HERE / "serve_m31.py")
shim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shim)
from engine_adapter import M31Engine, PollingRPC, PreparedIds  # noqa: E402


def done(finish="length"):
    return {"event": "done", "finish_reason": finish,
            "timings": {"prompt_n": 1, "predicted_n": 1, "prompt_ms": 1, "predicted_ms": 2}}


class FakeRPC:
    ready = {"event": "ready", "vocab": 200064, "context": 128}
    failure = None
    process = SimpleNamespace(poll=lambda: None)

    def __init__(self, events=()):
        self.events = iter(events)
        self.writes = []

    def write(self, value):
        self.writes.append(value)

    def read(self, timeout=None):
        return next(self.events)

    def poll(self, timeout):
        return self.read()

    def invalidate(self, reason):
        self.failure = reason


class Helper:
    ready = {"vocab": 200064, "eog": [2], "suppress": [3], "template": "M31 PR"}
    failure = None
    process = SimpleNamespace(poll=lambda: None)

    def __init__(self):
        self.calls = []
        self.pieces = {10: b"hello", 11: b"world"}

    def call(self, request):
        self.calls.append(request)
        op = request["op"]
        if op == "chat":
            return {"tokens": [1], "prompt": "M31 prompt", "grammar_active": bool(request.get("tools"))}
        if op == "grammar_check":
            return {"accepted": True}
        if op == "piece":
            return {"bytes": list(self.pieces[request["token"]])}
        if op == "parse":
            return {"message": {"role": "assistant", "content": request["content"]}}
        if op == "tokenize":
            return {"tokens": [20 + ord(c) for c in request["content"]]}
        if op == "detokenize":
            return {"content": "".join(self.pieces.get(t, b"").decode() for t in request["tokens"])}
        raise AssertionError(request)


def prepared(tools=None):
    request = {"op": "chat", "messages": [{"role": "user", "content": "hi"}], "tools": tools or []}
    return PreparedIds([1], request, {"tokens": [1]})


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.previous_parser = shim.S.OutputParser

    def tearDown(self):
        shim.S.OutputParser = self.previous_parser

    def test_optional_portable_profile_resolves_without_starting_model(self):
        config = {"runtime_profile": "../profiles/native-gpu.json", "engine": "/fake/m31_serve",
                  "model": "/fake/model.gguf"}
        argv, environment, unset = shim.runtime_plan(config, HERE)
        self.assertEqual(argv[:3], ["/fake/m31_serve", "--model", "/fake/model.gguf"])
        self.assertEqual(environment["M31_GPU_FAST_FFN"], "0")
        self.assertEqual(environment["GGML_CUDA_DISABLE_FUSION"], "1")
        self.assertIn("M31_GPU_QUANTS_DEBUG_DIR", unset)
        with self.assertRaisesRegex(ValueError, "not both"):
            shim.runtime_plan({**config, "engine_argv": ["other"]}, HERE)

    def test_anthropic_auto_and_parallel_choice_are_normalized(self):
        req = shim.chat_request([{"role": "user", "content": "hi"}], [], {},
                                {"tool_choice": {"type": "auto", "disable_parallel_tool_use": True}})
        self.assertEqual(req["tool_choice"], "auto")
        self.assertFalse(req["parallel_tool_calls"])
        service = shim.make_service(FakeRPC(), Helper(), Helper())
        self.assertEqual(service.encode_prompt([{"role": "user", "content": "hi"}], [], {}), [1])

    def test_sampling_and_ack_are_preserved_without_gen_translation(self):
        rpc = FakeRPC([{"event": "prefill", "prompt_n": 1, "prompt_ms": 1},
                       {"event": "token", "token": 10}, done()])
        helper = Helper()
        engine = M31Engine(rpc, helper)
        self.assertEqual(list(engine.generate(prepared(), 2, {"temperature": 0, "top_k": 999}, threading.Event())),
                         [None, 10])
        self.assertEqual(rpc.writes[0]["top_k"], 999)
        self.assertEqual(rpc.writes[0]["temperature"], 0)
        self.assertEqual(rpc.writes[1], {"continue": True})
        self.assertEqual(engine.last["reused"], 0)
        self.assertEqual(engine.last["prompt_tokens"], 1)

    def test_normalized_penalty_aliases_are_rejected_without_engine_side_effect(self):
        rpc, helper = FakeRPC(), Helper()
        engine = M31Engine(rpc, helper)
        for sampling in ({"repetition_penalty": 1.2}, {"repeat_penalty": 1.2},
                         {"penalty_last_n": 64}, {"repeat_last_n": -1}):
            with self.subTest(sampling=sampling), self.assertRaisesRegex(ValueError, "not implemented"):
                list(engine.generate(prepared(), 2, sampling, threading.Event()))
        self.assertEqual((rpc.writes, helper.calls), ([], []))

    def test_immutable_request_replay_and_id_mismatch_do_not_send_engine_request(self):
        ids = prepared()
        original = ids.chat_request
        request = {"op": "chat", "messages": [{"role": "user", "content": "before"}]}
        copied = PreparedIds([1], request, {})
        request["messages"][0]["content"] = "after"
        self.assertEqual(copied.chat_request["messages"][0]["content"], "before")
        class Mismatch(Helper):
            def call(self, request): return {"tokens": [7]}
        rpc = FakeRPC()
        with self.assertRaisesRegex(RuntimeError, "IDs changed"):
            list(M31Engine(rpc, Mismatch()).generate(ids, 1, {}, threading.Event()))
        self.assertEqual(rpc.writes, [])
        self.assertEqual(ids.chat_request, original)

    def test_grammar_error_negative_ack_drain_allows_next_request(self):
        rpc = FakeRPC([{"event": "sample", "token": 10}, {"event": "error", "error": "empty support"},
                       {"event": "token", "token": 11}, done()])
        class BrokenOnce(Helper):
            def call(self, request):
                if request["op"] == "grammar_check":
                    raise RuntimeError("PEG helper failure")
                return super().call(request)
        engine = M31Engine(rpc, BrokenOnce())
        with self.assertRaisesRegex(RuntimeError, "PEG helper failure"):
            list(engine.generate(prepared(), 2, {}, threading.Event()))
        self.assertEqual(rpc.writes[-1], {"accepted": False, "allowed": []})
        self.assertEqual(list(engine.generate(prepared(), 2, {}, threading.Event())), [11])

    def test_early_consumer_close_drains_before_reuse(self):
        rpc = FakeRPC([{"event": "token", "token": 10}, {"event": "sample", "token": 11},
                       {"event": "error", "error": "empty support"},
                       {"event": "token", "token": 11}, done()])
        engine = M31Engine(rpc, Helper())
        stream = engine.generate(prepared(), 2, {}, threading.Event())
        self.assertEqual(next(stream), 10)
        stream.close()
        self.assertEqual(rpc.writes[-1], {"accepted": False, "allowed": []})
        self.assertEqual(list(engine.generate(prepared(), 2, {}, threading.Event())), [11])

    def test_native_consumed_eos_is_exposed_to_common_service(self):
        rpc = FakeRPC([done("stop")])
        self.assertEqual(list(M31Engine(rpc, Helper()).generate(prepared(), 2, {}, threading.Event())), [2])

    def test_cancel_before_start_has_no_helper_or_engine_side_effect(self):
        rpc, helper, cancel = FakeRPC(), Helper(), threading.Event()
        cancel.set()
        self.assertEqual(list(M31Engine(rpc, helper).generate(prepared(), 2, {}, cancel)), [])
        self.assertEqual((rpc.writes, helper.calls), ([], []))

    def test_raw_continuations_and_geni_fail_explicitly(self):
        engine = M31Engine(FakeRPC(), Helper())
        with self.assertRaisesRegex(ValueError, "prepared chat IDs"):
            list(engine.generate([1], 2, {}, threading.Event()))
        with self.assertRaisesRegex(ValueError, "GENI"):
            list(engine.generate(prepared(), 2, {}, threading.Event(), embeddings="image.sve"))

    def test_common_service_uses_separate_render_helper_and_clamps_context(self):
        render, grammar = Helper(), Helper()
        service = shim.make_service(FakeRPC(), render, grammar)
        ids, thinking, maximum = service.prepare([{"role": "user", "content": "hi"}], None, {}, 256)
        self.assertEqual(maximum, 127)
        self.assertTrue(thinking)
        self.assertIsInstance(ids, PreparedIds)
        self.assertEqual(grammar.calls, [])
        self.assertEqual(ids.chat_request["messages"][0]["content"], "hi")

    def test_explicit_prompt_cache_optout_survives_common_prepare_and_generate(self):
        rpc = FakeRPC([done()])
        service = shim.make_service(rpc, Helper(), Helper())
        with patch.dict(os.environ, {"M31_PROMPT_CACHE_MIB": "1024"}):
            ids, _, _ = service.prepare([{"role": "user", "content": "hi"}], None, {}, 1,
                                        req={"cache_prompt": False})
            self.assertIs(ids.cache_prompt, False)
            list(service.engine.generate(ids, 1, {"temperature": 0}, threading.Event()))
            self.assertIs(rpc.writes[0]["cache_prompt"], False)
            with self.assertRaisesRegex(ValueError, "cache_prompt"):
                service.prepare([{"role": "user", "content": "hi"}], None, {}, 1,
                                req={"cache_prompt": "false"})

    def test_common_service_run_finalizes_peg_before_releasing_fifo(self):
        render, grammar = Helper(), Helper()
        rpc = FakeRPC([{"event": "token", "token": 10}, done()])
        service = shim.make_service(rpc, render, grammar)
        ids, thinking, maximum = service.prepare([{"role": "user", "content": "hi"}], None, {}, 1)
        stream = service.run(ids, thinking, None, maximum, {"temperature": 0}, threading.Event())
        event = next(stream)
        self.assertEqual((event[0], event[1].kind, event[1].text), ("event", "content", "hello"))
        rest = list(stream)
        self.assertEqual(rest[-1][0], "done")
        self.assertTrue(any(c["op"] == "parse" and not c["partial"] for c in grammar.calls))
        self.assertEqual(service.engine.last["generated"], 1)

    def test_grammar_stop_string_finalizes_even_when_negative_ack_yields_error(self):
        render, grammar = Helper(), Helper()
        grammar.pieces[10] = b"helloSTOPtail"
        render.pieces[10] = b"helloSTOPtail"
        rpc = FakeRPC([{"event": "token", "token": 10}, {"event": "sample", "token": 11},
                       {"event": "error", "error": "empty grammar support"}])
        service = shim.make_service(rpc, render, grammar)
        ids, thinking, maximum = service.prepare([{"role": "user", "content": "hi"}],
                                                [{"name": "probe", "parameters": {}}], {}, 5)
        ids.rendered["additional_stops"] = ["STOP"]
        result = list(service.run(ids, thinking, None, maximum, {}, threading.Event()))
        text = "".join(event.text for kind, event in result if kind == "event" and event.kind == "content")
        self.assertEqual(text, "hello")
        self.assertEqual(result[-1][0], "done")
        self.assertEqual(rpc.writes[-1], {"accepted": False, "allowed": []})

    def test_final_parser_snapshot_survives_next_helper_chat(self):
        helper = Helper()
        parser = shim.HelperOutputParser(helper, thinking=False)
        parser.feed("hello")
        parser.complete()
        before = len(helper.calls)
        helper.call({"op": "chat", "messages": []})
        self.assertEqual(parser.finish(), [])
        self.assertEqual(len(helper.calls), before + 1)

    def test_common_openai_http_nonstream_and_anthropic_nonstream(self):
        for path, body in (("/v1/chat/completions", {"model": "m31", "max_tokens": 1,
                              "messages": [{"role": "user", "content": "hi"}]}),
                           ("/v1/messages", {"model": "m31", "max_tokens": 1,
                              "messages": [{"role": "user", "content": "hi"}]})):
            with self.subTest(path=path):
                service = shim.make_service(FakeRPC([{"event": "token", "token": 10}, done()]),
                                            Helper(), Helper(), "m31")
                http_server = shim.S.Server(("127.0.0.1", 0), shim.S.make_handler(service))
                http_server.svc = service
                thread = threading.Thread(target=http_server.handle_request)
                thread.start()
                conn = http.client.HTTPConnection(*http_server.server_address, timeout=2)
                try:
                    conn.request("POST", path, json.dumps(body), {"Connection": "close"})
                    response = conn.getresponse()
                    payload = json.loads(response.read())
                    self.assertEqual(response.status, 200, payload)
                    self.assertIn("hello", json.dumps(payload))
                finally:
                    conn.close()
                    thread.join(timeout=2)
                    http_server.server_close()

    def test_common_openai_and_anthropic_sse(self):
        for path in ("/v1/chat/completions", "/v1/messages"):
            with self.subTest(path=path):
                service = shim.make_service(FakeRPC([{"event": "token", "token": 10}, done()]),
                                            Helper(), Helper(), "m31")
                http_server = shim.S.Server(("127.0.0.1", 0), shim.S.make_handler(service))
                http_server.svc = service
                thread = threading.Thread(target=http_server.handle_request)
                thread.start()
                conn = http.client.HTTPConnection(*http_server.server_address, timeout=2)
                try:
                    conn.request("POST", path, json.dumps({"model": "m31", "max_tokens": 1, "stream": True,
                                 "messages": [{"role": "user", "content": "hi"}]}), {"Connection": "close"})
                    response = conn.getresponse()
                    payload = response.read().decode()
                    self.assertEqual(response.status, 200, payload)
                    self.assertEqual(response.getheader("Content-Type"), "text/event-stream")
                    self.assertIn("hello", payload)
                    self.assertIn("[DONE]" if path.endswith("completions") else "message_stop", payload)
                finally:
                    conn.close()
                    thread.join(timeout=2)
                    http_server.server_close()


class RealPipeTests(unittest.TestCase):
    def test_json_pipe_two_requests_and_generator_close(self):
        code = '''import json,sys
print(json.dumps(dict(event='ready',context=128,vocab=200064)),flush=True)
for line in sys.stdin:
    req=json.loads(line)
    print(json.dumps(dict(event='token',token=10)),flush=True)
    ack=json.loads(sys.stdin.readline())
    print(json.dumps(dict(event='done',finish_reason='length',timings=dict(prompt_n=1,predicted_n=1,prompt_ms=1,predicted_ms=1))),flush=True)
'''
        rpc = PollingRPC([sys.executable, "-c", code], read_timeout=1)
        try:
            engine = M31Engine(rpc, Helper(), silence_s=1)
            stream = engine.generate(prepared(), 2, {}, threading.Event())
            self.assertEqual(next(stream), 10)
            stream.close()
            self.assertEqual(list(engine.generate(prepared(), 2, {}, threading.Event())), [10])
        finally:
            rpc.close()


@unittest.skipUnless(os.environ.get("M31_TEST_TOKENIZER") and os.environ.get("M31_TEST_MODEL"),
                     "set vocab helper/model for PR template/parser integration checks")
class RealHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        argv = [os.environ["M31_TEST_TOKENIZER"], os.environ["M31_TEST_MODEL"]]
        cls.render = PollingRPC(argv, read_timeout=30)
        cls.grammar = PollingRPC(argv, read_timeout=30)

    @classmethod
    def tearDownClass(cls):
        cls.grammar.close()
        cls.render.close()

    def setUp(self):
        self.previous_parser = shim.S.OutputParser

    def tearDown(self):
        shim.S.OutputParser = self.previous_parser

    def check_reply(self, text, tools=None, thinking=True):
        rpc = FakeRPC()
        rpc.ready = {**rpc.ready, "context": 4096}
        service = shim.make_service(rpc, self.render, self.grammar)
        ids, think, maximum = service.prepare([{"role": "user", "content": "Привет cafe\u0301 🐈"}],
                                             tools, {"enable_thinking": thinking}, 128)
        reference = self.render.call(ids.chat_request)
        self.assertEqual(list(ids), reference["tokens"])
        tokens = self.render.call({"op": "tokenize", "content": text, "parse_special": True})["tokens"]
        rpc.events = iter([*({"event": "token", "token": token} for token in tokens), done("stop")])
        result = list(service.run(ids, think, tools, maximum, {"temperature": 0}, threading.Event()))
        actual = {"content": "", "reasoning_content": "", "tool_calls": []}
        for kind, event in result:
            if kind != "event": continue
            if event.kind in ("content", "reasoning"):
                actual["content" if event.kind == "content" else "reasoning_content"] += event.text
            elif event.kind == "tool_call":
                actual["tool_calls"].append({"name": event.call.name, "arguments": event.call.arguments})
        # Re-render resets the helper's PEG state before an independent full parse.
        self.render.call(ids.chat_request)
        expected = self.render.call({"op": "parse", "content": text, "partial": False})["message"]
        self.assertEqual(actual["content"], expected.get("content") or "")
        self.assertEqual(actual["reasoning_content"], expected.get("reasoning_content") or "")
        self.assertEqual(actual["tool_calls"], [{"name": call["function"]["name"],
                                                "arguments": json.loads(call["function"]["arguments"])}
                                               for call in expected.get("tool_calls", [])])
        return actual

    def test_reasoning_unicode_and_native_eos_common_service(self):
        actual = self.check_reply("<mm:think>Проверяю 🐈</mm:think>Ответ café")
        self.assertEqual(actual["content"], "Ответ café")
        self.assertEqual(actual["reasoning_content"], "Проверяю 🐈")

    def test_nested_tool_xml_common_service(self):
        tools = [{"name": "probe", "parameters": {"type": "object", "properties": {
            "obj": {"type": "object", "properties": {"value": {"type": "integer"}}}}, "required": ["obj"]}}]
        ns = "]<]minimax[>["
        text = (ns + '<tool_call>' + ns + '<invoke name="probe">' + ns + '<obj>' +
                ns + '<value>7' + ns + '</value>' + ns + '</obj>' + ns + '</invoke>' + ns + '</tool_call>')
        actual = self.check_reply(text, tools, thinking=False)
        self.assertEqual(actual["tool_calls"], [{"name": "probe", "arguments": {"obj": {"value": 7}}}])


if __name__ == "__main__":
    unittest.main()
