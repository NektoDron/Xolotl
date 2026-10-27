#!/usr/bin/env python3
"""Short native HTTP integration smoke; not GSM/tools quality acceptance."""
import argparse
import hashlib
import http.client
import importlib.util
import json
from pathlib import Path
import socket
import sys
import time


def read_audit(path):
    if not path.exists():
        return []
    # Consume only complete newline-terminated frames. A concurrent writer may
    # leave a tail ending in a nested '}' or in the middle of a UTF-8 character.
    return [json.loads(line) for line in path.read_bytes().split(b"\n")[:-1] if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8181)
    args = parser.parse_args()
    shim_path = args.tree / "tools/m31/serve/serve_m31.py"
    sys.path.insert(0, str(shim_path.parent))
    spec = importlib.util.spec_from_file_location("m31_client_common", shim_path)
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    config = json.loads(args.config.read_text())
    helper = shim.PollingRPC(config["tokenizer_argv"], read_timeout=30)
    result = {"format": "m31-common-native-http-smoke-v1", "checks": [],
              "quality_acceptance": False, "performance_acceptance": False}
    model = config["model_name"]

    def records():
        return read_audit(args.audit)

    def reference(path, body):
        convert = shim.S.anthropic_to_messages if path == "/v1/messages" else shim.S.openai_to_messages
        messages, tools, kwargs = convert(body)
        request = shim.chat_request(messages, tools, kwargs, body)
        return helper.call(request)["tokens"]

    def validate_request(name, expected, before):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            actual = [row for row in records() if row["kind"] == "request"]
            if len(actual) > before:
                row = actual[before]
                if row["params"]["tokens"] != expected:
                    raise AssertionError(f"{name}: engine IDs differ from independent PR helper")
                return row["request"], hashlib.sha256(json.dumps(expected).encode()).hexdigest()
            time.sleep(.05)
        raise TimeoutError(f"{name}: no audited native request")

    def request(name, path, body, cancel_stream=False):
        expected = reference(path, body)
        before = len([row for row in records() if row["kind"] == "request"])
        connection = http.client.HTTPConnection("127.0.0.1", args.port, timeout=180)
        try:
            connection.request("POST", path, json.dumps(body),
                               {"Content-Type": "application/json", "Connection": "close"})
            response = connection.getresponse()
            if response.status != 200:
                raise AssertionError((name, response.status, response.read().decode()))
            if cancel_stream:
                chunks = []
                while True:
                    line = response.readline().decode()
                    if not line:
                        raise AssertionError("stream ended before first content delta")
                    if line.startswith("data: "):
                        payload = json.loads(line[6:])
                        chunks.append(payload)
                        if payload.get("choices", [{}])[0].get("delta", {}).get("content"):
                            break
                if connection.sock is not None:
                    connection.sock.shutdown(socket.SHUT_RDWR)
                response.close()
                value = {"cancelled_after_content": True, "chunks": chunks}
            else:
                value = json.loads(response.read())
            request_id, ids_sha256 = validate_request(name, expected, before)
            result["checks"].append({"name": name, "request_id": request_id,
                                     "prompt_tokens": len(expected), "ids_sha256": ids_sha256,
                                     "reference_ids_equal": True, "response": value})
            return value, request_id
        finally:
            connection.close()

    def openai(prompt, maximum=32, **extra):
        return {"model": model, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0, "top_p": 1, "top_k": 40, "min_p": 0, "seed": 0,
                "max_tokens": maximum, "chat_template_kwargs": {"enable_thinking": False}, **extra}

    try:
        value, _ = request("openai_greedy", "/v1/chat/completions",
                           openai("Ответь ровно SMOKE_OK без пояснений."))
        assert "SMOKE_OK" in value["choices"][0]["message"]["content"], value
        value, _ = request("anthropic_greedy", "/v1/messages",
                           {"model": model, "messages": [{"role": "user", "content": "Ответь ровно SMOKE_OK без пояснений."}],
                            "max_tokens": 32, "temperature": 0, "top_p": 1, "top_k": 40,
                            "min_p": 0, "seed": 0, "thinking": {"type": "disabled"}})
        assert "SMOKE_OK" in "".join(block.get("text", "") for block in value["content"]), value
        _, cancelled_id = request("stream_cancel", "/v1/chat/completions",
                                  openai("Напечатай числа от 1 до 500, каждое на новой строке, без пояснений.",
                                         maximum=512, stream=True), cancel_stream=True)
        value, _ = request("after_cancel", "/v1/chat/completions",
                           openai("Ответь ровно NEXT_OK без пояснений."))
        assert "NEXT_OK" in value["choices"][0]["message"]["content"], value
        cancelled = [r["event"] for r in records() if r["kind"] == "engine_event" and r["request"] == cancelled_id]
        assert any(e.get("event") in ("done", "error") for e in cancelled), cancelled
        assert sum(e.get("event") == "token" for e in cancelled) < 512, "disconnect did not cancel native generation"
        tools = [{"type": "function", "function": {"name": "record_alpha", "parameters": {
            "type": "object", "properties": {"payload": {"type": "object", "properties": {
                "count": {"type": "integer"}, "tags": {"type": "array", "items": {"type": "string"}}},
                "required": ["count", "tags"]}}, "required": ["payload"]}}},
                 {"type": "function", "function": {"name": "record_beta", "parameters": {
            "type": "object", "properties": {"payload": {"type": "object", "properties": {
                "active": {"type": "boolean"}, "child": {"type": "object", "properties": {
                    "value": {"type": "string"}}, "required": ["value"]}}, "required": ["active", "child"]}},
            "required": ["payload"]}}}]
        prompt = ('Используй только инструменты. В одном ответе вызови оба инструмента, каждый ровно один раз: '
                  'record_alpha с аргументами {"payload":{"count":7,"tags":["один","🐈"]}} и '
                  'record_beta с аргументами {"payload":{"active":true,"child":{"value":"готово"}}}. '
                  'Не отвечай обычным текстом.')
        value, tool_id = request("auto_two_nested_tools", "/v1/chat/completions",
                                 openai(prompt, maximum=384, tools=tools, tool_choice="auto", parallel_tool_calls=True))
        calls = value["choices"][0]["message"].get("tool_calls", [])
        actual = {call["function"]["name"]: json.loads(call["function"]["arguments"]) for call in calls}
        assert len(calls) == 2 and actual == {
            "record_alpha": {"payload": {"count": 7, "tags": ["один", "🐈"]}},
            "record_beta": {"payload": {"active": True, "child": {"value": "готово"}}}}, actual
        grammar = [r for r in records() if r["kind"] == "grammar_ack" and r["request"] == tool_id]
        assert grammar, "native sample/grammar callback was not exercised"
        result.update(pass_smoke=True, grammar_acks=len(grammar), grammar_rejections=sum(not r["accepted"] for r in grammar))
    except BaseException as exc:
        result.update(pass_smoke=False, error=str(exc) or type(exc).__name__)
        raise
    finally:
        helper.close()
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
            stream.write("\n")


if __name__ == "__main__":
    main()
