#!/usr/bin/env python3
"""Opt-in M31 adapter: isolated Xolotl engine + audited PR vocab/chat helper.

Only bind loopback. GPU engine must be started from an aictl-owned card session.
No external Python packages are required. One inference request runs at a time.
"""
from __future__ import annotations

import argparse
import codecs
import contextlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class RPC:
    def __init__(self, argv):
        self.lock = threading.RLock()
        self.process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, encoding="utf-8", bufsize=1)
        self.ready = self.read()
        if self.ready.get("event") != "ready":
            self.close()
            raise RuntimeError(f"helper did not become ready: {self.ready}")

    def write(self, value):
        self.process.stdin.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        self.process.stdin.flush()

    def read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"helper exited: rc={self.process.poll()}")
        value = json.loads(line)
        return value

    def call(self, value):
        with self.lock:
            self.write(value)
            result = self.read()
            if "error" in result:
                raise RuntimeError(result["error"])
            return result

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def validate_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty array")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ("root", "system", "developer", "user", "assistant", "tool"):
            raise ValueError("invalid message role")
        content = message.get("content")
        if content is not None and not isinstance(content, (str, list)):
            raise ValueError("content must be string, text array or null")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str):
                    raise ValueError("M31 text server accepts text parts only; vision is not connected")
    return messages


def request_parameters(body, context, prompt_tokens):
    maximum = body.get("max_tokens", body.get("n_predict", 256))
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0:
        raise ValueError("max_tokens must be a nonnegative integer")
    if prompt_tokens + maximum > context:
        raise ValueError("prompt plus max_tokens exceeds configured context")
    if body.get("n", 1) != 1:
        raise ValueError("only n=1 is supported")
    for key in ("frequency_penalty", "presence_penalty"):
        if body.get(key, 0):
            raise ValueError(f"{key} is not implemented")
    if body.get("repeat_penalty", 1) != 1:
        raise ValueError("repeat_penalty other than 1 is not implemented")
    if body.get("logit_bias") or body.get("response_format") or body.get("grammar"):
        raise ValueError("logit_bias, response_format and grammar constraints are not implemented")
    params = {"max_tokens": maximum, "ignore_eos": bool(body.get("ignore_eos", False)),
              "temperature": body.get("temperature", 1.0), "top_p": body.get("top_p", 0.95),
              "top_k": body.get("top_k", 40), "min_p": body.get("min_p", 0.05), "interactive": True}
    for key in ("temperature", "top_p", "min_p"):
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not float("-inf") < value < float("inf"):
            raise ValueError(f"{key} must be finite")
    if params["temperature"] < 0 or not 0 < params["top_p"] <= 1 or not 0 <= params["min_p"] <= 1:
        raise ValueError("invalid sampling parameters")
    if isinstance(params["top_k"], bool) or not isinstance(params["top_k"], int) or params["top_k"] < 0:
        raise ValueError("top_k must be a nonnegative integer")
    if "seed" in body and body["seed"] is not None and body["seed"] != -1:
        if isinstance(body["seed"], bool) or not isinstance(body["seed"], int) or not 0 <= body["seed"] < 2**64:
            raise ValueError("seed must be uint64 or -1")
        params["seed"] = body["seed"]
    stop = body.get("stop", [])
    if isinstance(stop, str):
        stop = [stop]
    if stop is None:
        stop = []
    if not isinstance(stop, list) or any(not isinstance(s, str) or not s for s in stop):
        raise ValueError("stop must contain nonempty strings")
    return params, stop


def clipped_text(text, stops):
    """Return safe text, withholding a suffix that may become a stop string."""
    found = [text.find(stop) for stop in stops if stop in text]
    if found:
        return text[:min(found)], True
    hold = 0
    for stop in stops:
        for n in range(1, min(len(stop), len(text) + 1)):
            if text.endswith(stop[:n]):
                hold = max(hold, n)
    return (text[:-hold] if hold else text), False


class Application:
    def __init__(self, engine, tokenizer, model="m31-xolotl"):
        self.engine, self.tokenizer, self.model = engine, tokenizer, model
        self.lock = threading.Lock()
        self.context = int(engine.ready["context"])
        if int(engine.ready["vocab"]) != int(tokenizer.ready["vocab"]):
            raise ValueError("engine/tokenizer vocabulary mismatch")
        self.pieces = {}

    def tokenize(self, text, **kwargs):
        return self.tokenizer.call({"op": "tokenize", "content": text, **kwargs})["tokens"]

    def prepare(self, body, chat=True):
        if chat:
            messages = validate_messages(body.get("messages"))
            choice = body.get("tool_choice", "auto")
            if not isinstance(choice, str) or choice not in ("auto", "none"):
                raise ValueError("supported tool_choice values: auto, none; forced tool grammar is not connected")
            tools = body.get("tools", [])
            if not isinstance(tools, list):
                raise ValueError("tools must be an array")
            kwargs = body.get("chat_template_kwargs", {})
            if not isinstance(kwargs, dict):
                raise ValueError("chat_template_kwargs must be an object")
            rendered = self.tokenizer.call({"op": "chat", "messages": messages, "tools": tools,
                                           "tool_choice": choice, "parallel_tool_calls": body.get("parallel_tool_calls", True),
                                           "enable_thinking": kwargs.get("enable_thinking", True), "chat_template_kwargs": kwargs})
            ids = rendered["tokens"]
            extra_stops = rendered.get("additional_stops", [])
            grammar_active = rendered.get("grammar_active", False)
        else:
            prompt = body.get("prompt", "")
            if isinstance(prompt, str):
                ids = self.tokenize(prompt, add_special=True, parse_special=True)
            elif isinstance(prompt, list) and all(isinstance(v, int) and not isinstance(v, bool) for v in prompt):
                ids = prompt
            else:
                raise ValueError("prompt must be a string or token ID array")
            extra_stops = []
            grammar_active = False
        if not ids:
            raise ValueError("empty prompt")
        params, stops = request_parameters(body, self.context, len(ids))
        params.update(grammar_check=grammar_active, tokens=ids, stop_ids=self.tokenizer.ready["eog"], suppress_ids=self.tokenizer.ready.get("suppress", []))
        return params, stops + extra_stops

    def piece(self, token):
        if token not in self.pieces:
            self.pieces[token] = bytes(self.tokenizer.call({"op": "piece", "token": token})["bytes"])
        return self.pieces[token]

    def events(self, params, stops, chat=True):
        """Caller holds self.lock. Consume fully, or close generator to cancel/drain."""
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        text = ""
        previous = {"role": "assistant", "content": "", "reasoning_content": ""}
        waiting_ack = False
        completed = False
        tool_ids = {}
        try:
            self.engine.write(params)
            while True:
                event = self.engine.read()
                if event.get("event") == "error":
                    completed = True
                    raise RuntimeError(event["error"])
                if event["event"] == "sample":
                    self.engine.write(self.tokenizer.call({"op": "grammar_check", "token": event["token"]}))
                elif event["event"] == "prefill":
                    yield event
                elif event["event"] == "token":
                    waiting_ack = True
                    text += decoder.decode(self.piece(event["token"]))
                    safe_text, stop = clipped_text(text, stops)
                    # The audited PEG mapper holds incomplete tags and emits reasoning separately.
                    if chat:
                        message = self.tokenizer.call({"op": "parse", "content": safe_text, "partial": True})["message"]
                    else:
                        message = {"role": "assistant", "content": safe_text}
                    delta = {}
                    for key in ("content", "reasoning_content"):
                        current, old = message.get(key) or "", previous.get(key) or ""
                        if not current.startswith(old):
                            raise RuntimeError(f"chat parser retracted streamed {key}")
                        if current != old:
                            delta[key] = current[len(old):]
                    previous = message
                    self.engine.write({"continue": not stop})
                    waiting_ack = False
                    if delta:
                        yield {"event": "delta", "delta": delta}
                elif event["event"] == "done":
                    completed = True
                    text += decoder.decode(b"", final=True)
                    safe_text, stopped = clipped_text(text, stops)
                    # If generation ended before completing a requested stop, retain its text.
                    if not stopped:
                        safe_text = text
                    if chat:
                        message = self.tokenizer.call({"op": "parse", "content": safe_text, "partial": False})["message"]
                    else:
                        message = {"role": "assistant", "content": safe_text}
                    for i, call in enumerate(message.get("tool_calls", [])):
                        call["id"] = tool_ids.setdefault(i, call.get("id") or "call_" + uuid.uuid4().hex[:24])
                    finish = "tool_calls" if message.get("tool_calls") else event["finish_reason"]
                    delta = {}
                    for key in ("content", "reasoning_content"):
                        current, old = message.get(key) or "", previous.get(key) or ""
                        if not current.startswith(old):
                            raise RuntimeError(f"chat parser retracted final {key}")
                        if current != old:
                            delta[key] = current[len(old):]
                    if message.get("tool_calls"):
                        delta["tool_calls"] = [{"index": i, **call} for i, call in enumerate(message["tool_calls"])]
                    if delta:
                        yield {"event": "delta", "delta": delta}
                    yield {**event, "message": message, "finish_reason": finish}
                    return
                else:
                    raise RuntimeError(f"unexpected engine event: {event}")
        finally:
            if not completed:
                # Cancel at the next token boundary and consume its final event before reusing stdin.
                if not waiting_ack:
                    while True:
                        event = self.engine.read()
                        if event["event"] == "sample":
                            self.engine.write(self.tokenizer.call({"op": "grammar_check", "token": event["token"]}))
                        elif event["event"] == "token":
                            waiting_ack = True
                            break
                        if event["event"] in ("done", "error"):
                            completed = True
                            break
                if waiting_ack:
                    self.engine.write({"continue": False})
                    event = self.engine.read()
                    if event["event"] not in ("done", "error"):
                        raise RuntimeError("engine did not acknowledge cancellation")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def app(self):
        return self.server.app

    def json(self, status, value):
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path in ("/health", "/healthz"):
            alive = self.app.engine.process.poll() is None and self.app.tokenizer.process.poll() is None
            self.json(200 if alive else 503, {"status": "ok" if alive else "error"})
        elif self.path == "/v1/models":
            self.json(200, {"object": "list", "data": [{"id": self.app.model, "object": "model", "created": 0, "owned_by": "local"}]})
        else:
            self.json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        streaming = False
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 64 * 1024 * 1024:
                raise ValueError("invalid request size")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("request body must be an object")
            if self.path == "/tokenize":
                content = body.get("content", "")
                if not isinstance(content, str):
                    raise ValueError("content must be a string")
                self.json(200, {"tokens": self.app.tokenize(content, add_special=body.get("add_special", False), parse_special=body.get("parse_special", False))})
                return
            if self.path == "/detokenize":
                self.json(200, self.app.tokenizer.call({"op": "detokenize", "tokens": body.get("tokens", [])}))
                return
            if self.path not in ("/v1/chat/completions", "/v1/completions", "/completion"):
                self.json(404, {"error": {"message": "not found"}})
                return
            chat = self.path == "/v1/chat/completions"
            native = self.path == "/completion"
            with self.app.lock:
                params, stops = self.app.prepare(body, chat)
                request_id, created = "chatcmpl-" + uuid.uuid4().hex, int(time.time())
                if body.get("stream", False):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    streaming = True

                    def emit(value):
                        self.wfile.write(("data: " + (value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)) + "\n\n").encode())
                        self.wfile.flush()

                    def chunk(delta, finish=None, timings=None, usage=None):
                        choice = {"index": 0, "finish_reason": finish}
                        choice["delta" if chat else "text"] = delta if chat else delta.get("content", "")
                        value = {"id": request_id, "object": "chat.completion.chunk" if chat else "text_completion",
                                 "created": created, "model": self.app.model, "choices": [choice]}
                        if timings is not None:
                            value["timings"] = timings
                        if usage is not None:
                            value["usage"] = usage
                        return value

                    emit(chunk({"role": "assistant"} if chat else {}))
                    with contextlib.closing(self.app.events(params, stops, chat)) as events:
                        for event in events:
                            if event["event"] == "delta":
                                emit(chunk(event["delta"]))
                            elif event["event"] == "done":
                                timings = event["timings"]
                                usage = {"prompt_tokens": timings["prompt_n"], "completion_tokens": timings["predicted_n"],
                                         "total_tokens": timings["prompt_n"] + timings["predicted_n"]}
                                emit(chunk({}, event["finish_reason"], timings, usage))
                    emit("[DONE]")
                else:
                    done = None
                    with contextlib.closing(self.app.events(params, stops, chat)) as events:
                        for event in events:
                            if event["event"] == "done":
                                done = event
                    timings = done["timings"]
                    usage = {"prompt_tokens": timings["prompt_n"], "completion_tokens": timings["predicted_n"],
                             "total_tokens": timings["prompt_n"] + timings["predicted_n"]}
                    if native:
                        value = {"content": done["message"].get("content", ""), "stop": True,
                                 "stopped_eos": done["finish_reason"] == "stop", "tokens_predicted": timings["predicted_n"],
                                 "tokens_evaluated": timings["prompt_n"], "timings": timings}
                    else:
                        choice = {"index": 0, "finish_reason": done["finish_reason"]}
                        choice["message" if chat else "text"] = done["message"] if chat else done["message"].get("content", "")
                        value = {"id": request_id, "object": "chat.completion" if chat else "text_completion",
                                 "created": created, "model": self.app.model, "choices": [choice], "usage": usage, "timings": timings}
                    self.json(200, value)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as exc:
            if streaming:
                try:
                    self.wfile.write(("data: " + json.dumps({"error": {"message": str(exc)}}) + "\n\ndata: [DONE]\n\n").encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.json(400 if isinstance(exc, (ValueError, KeyError, TypeError)) else 500,
                          {"error": {"message": str(exc), "type": "invalid_request_error" if isinstance(exc, ValueError) else "server_error"}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--alias", default="m31-xolotl")
    parser.add_argument("--port", type=int, default=8180)
    parser.add_argument("--host", choices=["127.0.0.1", "localhost"], default="127.0.0.1")
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--arena-gib", type=float, default=0)
    parser.add_argument("--gpu-experts-gib", type=float, default=0)
    parser.add_argument("--gpu-experts-min-tokens", type=int, default=32)
    for flag in ("reuse-allocations", "persistent-cpu-pool", "cpu-residual", "cpu-decode-cache", "profile"):
        parser.add_argument("--" + flag, action="store_true")
    args = parser.parse_args()
    if args.backend == "cuda" and os.environ.get("M31_CARD_SESSION") != "1":
        parser.error("CUDA requires an aictl-owned card session")
    argv = [args.engine, "--model", args.model, "--backend", args.backend, "--ctx", str(args.ctx),
            "--chunk", str(args.chunk), "--threads", str(args.threads), "--arena-gib", str(args.arena_gib),
            "--gpu-experts-gib", str(args.gpu_experts_gib), "--gpu-experts-min-tokens", str(args.gpu_experts_min_tokens)]
    for flag in ("reuse-allocations", "persistent-cpu-pool", "cpu-residual", "cpu-decode-cache", "profile"):
        argv += ["--" + flag, "1" if getattr(args, flag.replace("-", "_")) else "0"]
    tokenizer = RPC([args.tokenizer, args.model])
    engine = None
    server = None
    try:
        engine = RPC(argv)
        app = Application(engine, tokenizer, args.alias)
        server = ThreadingHTTPServer((args.host, args.port), Handler)
        server.daemon_threads = True
        server.app = app
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        print(f"M31 server ready: http://{args.host}:{args.port} model={args.alias}", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server:
            server.server_close()
        if engine:
            engine.close()
        tokenizer.close()


if __name__ == "__main__":
    main()
