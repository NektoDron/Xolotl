#!/usr/bin/env python3
"""Opt-in M31 adapter: isolated Xolotl engine + audited PR vocab/chat helper.

Only bind loopback. GPU engine must be started from an aictl-owned card session.
Text needs no external Python packages. Optional vision uses NumPy/Pillow.
One inference request runs at a time.
"""
from __future__ import annotations

import argparse
import codecs
import contextlib
import json
import os
import queue
import select
import socket
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# This file is also loaded by path from the common bridge.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parent))
from ttft_trace import TTFTTrace


class RPC:
    """Line RPC with bounded waits; a timed-out channel is never reused."""
    def __init__(self, argv, read_timeout=300.0, startup_timeout=600.0):
        if read_timeout <= 0 or startup_timeout <= 0:
            raise ValueError("RPC timeouts must be positive")
        self.lock = threading.RLock()
        self.read_timeout = read_timeout
        self.failure = None
        self.lines = queue.Queue(maxsize=8)
        self.stopped = threading.Event()
        self.process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, encoding="utf-8", bufsize=1)
        os.set_blocking(self.process.stdin.fileno(), False)
        self.reader = threading.Thread(target=self._receive, daemon=True, name="m31-rpc-reader")
        self.reader.start()
        try:
            self.ready = self.read(timeout=startup_timeout)
            if self.ready.get("event") != "ready":
                raise RuntimeError(f"helper did not become ready: {self.ready}")
        except BaseException:
            self.close()
            raise

    def _receive(self):
        try:
            while not self.stopped.is_set():
                # A missing newline cannot make the request thread wait forever.
                line = self.process.stdout.readline(64 * 1024 * 1024 + 1)
                if len(line) > 64 * 1024 * 1024:
                    raise RuntimeError("helper RPC frame exceeds 64 MiB")
                item = line if line else EOFError("helper closed stdout")
                while not self.stopped.is_set():
                    try:
                        self.lines.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        pass
                if not line:
                    return
        except Exception as exc:
            while not self.stopped.is_set():
                try:
                    self.lines.put(exc, timeout=0.1)
                    return
                except queue.Full:
                    pass

    def invalidate(self, reason):
        self.failure = self.failure or str(reason)
        self.stopped.set()
        if self.process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                self.process.terminate()

    def _healthy(self):
        if self.failure is not None:
            raise RuntimeError(f"helper RPC unavailable: {self.failure}")

    def write(self, value):
        self._healthy()
        # Nonblocking pipe writes also bound requests whose helper stopped
        # consuming stdin. A timeout invalidates the whole framed channel.
        data = memoryview((json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
        deadline = time.monotonic() + self.read_timeout
        try:
            while data:
                try:
                    written = os.write(self.process.stdin.fileno(), data)
                    if written <= 0:
                        raise BrokenPipeError("helper closed stdin")
                    data = data[written:]
                except BlockingIOError:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not select.select([], [self.process.stdin], [], remaining)[1]:
                        raise TimeoutError("helper RPC write timed out")
        except (OSError, ValueError) as exc:
            self.invalidate(exc)
            raise RuntimeError(f"helper RPC write failed: {exc}") from exc

    def read(self, timeout=None):
        self._healthy()
        try:
            item = self.lines.get(timeout=self.read_timeout if timeout is None else timeout)
            if isinstance(item, Exception):
                raise item
            value = json.loads(item)
            if not isinstance(value, dict):
                raise ValueError("helper RPC response must be an object")
            return value
        except (queue.Empty, ValueError, EOFError, OSError) as exc:
            reason = "helper RPC timed out" if isinstance(exc, queue.Empty) else str(exc)
            self.invalidate(reason)
            raise RuntimeError(reason) from exc
        except Exception as exc:
            self.invalidate(exc)
            raise RuntimeError(f"helper RPC read failed: {exc}") from exc

    def call(self, value):
        if not self.lock.acquire(timeout=self.read_timeout):
            raise RuntimeError("helper RPC is busy")
        try:
            self.write(value)
            result = self.read()
            if "error" in result:
                raise RuntimeError(result["error"])
            return result
        finally:
            self.lock.release()

    def close(self):
        self.stopped.set()
        if self.process.poll() is None:
            # EOF lets a healthy, idle engine flush profiling/cache counters.
            with contextlib.suppress(BrokenPipeError, OSError, ValueError):
                self.process.stdin.close()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self.reader.join(timeout=1)
        with contextlib.suppress(BrokenPipeError, OSError, ValueError):
            self.process.stdin.close()
        self.process.stdout.close()


class ServerBusy(RuntimeError):
    pass


class ClientDisconnected(ConnectionError):
    pass


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
    # Match native/OpenAI aliases; omitted or -1 means the remaining context.
    maximum = body.get("n_predict", body.get("max_completion_tokens", body.get("max_tokens", -1)))
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < -1:
        raise ValueError("generation limit must be an integer >= -1")
    if prompt_tokens > context:
        raise ValueError("prompt exceeds configured context")
    available = context - prompt_tokens
    maximum = available if maximum == -1 else min(maximum, available)
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
    if "cache_prompt" in body:
        if not isinstance(body["cache_prompt"], bool):
            raise ValueError("cache_prompt must be boolean")
        params["cache_prompt"] = body["cache_prompt"]
    stop = body.get("stop", [])
    if isinstance(stop, str):
        stop = [stop]
    if stop is None:
        stop = []
    if not isinstance(stop, list) or any(not isinstance(s, str) or not s for s in stop):
        raise ValueError("stop must contain nonempty strings")
    return params, stop


def usage_counts(timings):
    value = {"prompt_tokens": timings["prompt_n"], "completion_tokens": timings["predicted_n"],
             "total_tokens": timings["prompt_n"] + timings["predicted_n"]}
    if "cached_n" in timings:
        value["prompt_tokens_details"] = {"cached_tokens": timings["cached_n"]}
    return value


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
    def __init__(self, engine, tokenizer, model="m31-xolotl", profile=False, vision=None, media_input=None):
        self.engine, self.tokenizer, self.model = engine, tokenizer, model
        self.profile = profile
        self.vision, self.media_input = vision, media_input
        self.lock = threading.Lock()
        self.inference_queue = threading.BoundedSemaphore(2)
        self.queue_timeout = 1.0
        self.context = int(engine.ready["context"])
        if int(engine.ready["vocab"]) != int(tokenizer.ready["vocab"]):
            raise ValueError("engine/tokenizer vocabulary mismatch")
        self.pieces = {}

    @contextlib.contextmanager
    def inference_slot(self):
        # One running request and one bounded waiter; reject further clients.
        if not self.inference_queue.acquire(blocking=False):
            raise ServerBusy("inference queue is full")
        acquired = False
        try:
            acquired = self.lock.acquire(timeout=self.queue_timeout)
            if not acquired:
                raise ServerBusy("inference queue wait timed out")
            yield
        finally:
            if acquired:
                self.lock.release()
            self.inference_queue.release()

    def tokenize(self, text, **kwargs):
        return self.tokenizer.call({"op": "tokenize", "content": text, **kwargs})["tokens"]

    def prepare(self, body, chat=True, trace=None):
        prepare_started = time.perf_counter_ns()
        manifest = None
        prompt_rows = None
        if chat:
            messages = body.get("messages")
            media, markers, fps = [], [], 1.0
            if self.media_input is not None:
                # Validate outer structure before the media reader visits content.
                if not isinstance(messages, list) or not messages or any(not isinstance(m, dict) for m in messages):
                    raise ValueError("messages must be a nonempty array of objects")
                media_started = time.perf_counter_ns()
                messages, media, markers, fps = self.media_input.prepare(messages, body.get("video_fps"))
                if trace:
                    trace.add("media_prepare_ms", (time.perf_counter_ns()-media_started)/1e6)
                    trace.details["media"] = dict(getattr(self.media_input, "last_timings", {}))
            messages = validate_messages(messages)
            choice = body.get("tool_choice", "auto")
            if not isinstance(choice, str) or choice not in ("auto", "none"):
                raise ValueError("supported tool_choice values: auto, none; forced tool grammar is not connected")
            tools = body.get("tools", [])
            if not isinstance(tools, list):
                raise ValueError("tools must be an array")
            for tool in tools:
                function = tool.get("function") if isinstance(tool, dict) else None
                if (not isinstance(tool, dict) or tool.get("type") != "function" or
                        not isinstance(function, dict) or not isinstance(function.get("name"), str) or
                        not function["name"] or not isinstance(function.get("parameters", {}), dict)):
                    raise ValueError("tools must contain function objects with a name and parameters object")
            if not isinstance(body.get("parallel_tool_calls", True), bool):
                raise ValueError("parallel_tool_calls must be boolean")
            kwargs = body.get("chat_template_kwargs", {})
            if not isinstance(kwargs, dict):
                raise ValueError("chat_template_kwargs must be an object")
            kwargs = dict(kwargs)
            enable_thinking = kwargs.get("enable_thinking", True)
            if not isinstance(enable_thinking, bool):
                raise ValueError("enable_thinking must be boolean")
            if "reasoning_effort" in body:
                effort = body["reasoning_effort"]
                if not isinstance(effort, str):
                    raise ValueError("reasoning_effort must be a string")
                if effort == "none":
                    enable_thinking = False
                    kwargs.pop("reasoning_effort", None)
                elif effort:
                    kwargs["reasoning_effort"] = effort
            template_started = time.perf_counter_ns()
            rendered = self.tokenizer.call({"op": "chat", "messages": messages, "tools": tools,
                                           "tool_choice": choice, "parallel_tool_calls": body.get("parallel_tool_calls", True),
                                           "enable_thinking": enable_thinking, "chat_template_kwargs": kwargs})
            ids = rendered["tokens"]
            if trace:
                trace.add("chat_template_rpc_ms", (time.perf_counter_ns()-template_started)/1e6)
            if media:
                if self.vision is None:
                    raise ValueError("vision encoder is not configured")
                media_marker = self.media_input.marker
                prompt = rendered["prompt"]
                if media_marker in prompt:
                    raise ValueError("literal mtmd media marker is reserved in multimodal requests")
                for marker in markers:
                    if prompt.count(marker) != 1:
                        raise ValueError("chat template did not preserve media order/markers")
                    prompt = prompt.replace(marker, media_marker)
                encoder_started = time.perf_counter_ns()
                manifest = self.vision.encode(prompt, media, video_fps=fps)
                if trace:
                    trace.add("encoder_python_total_ms", (time.perf_counter_ns()-encoder_started)/1e6)
                    trace.details["encoder"] = dict(getattr(self.vision, "last_stats", {}))
                metadata = json.loads(Path(manifest).read_text())
                if metadata.get("format") != "m31-vision-batch-v1" or not metadata.get("encoded"):
                    raise ValueError("encoder returned an invalid mixed manifest")
                prompt_rows = metadata["rows"]
                if type(prompt_rows) is not int or prompt_rows < 1:
                    raise ValueError("encoder returned invalid mixed row count")
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
        params, stops = request_parameters(body, self.context, prompt_rows if prompt_rows is not None else len(ids))
        params.update(grammar_check=grammar_active, tokens=ids, stop_ids=self.tokenizer.ready["eog"], suppress_ids=self.tokenizer.ready.get("suppress", []))
        if manifest is not None:
            params["mixed_manifest"] = str(Path(manifest).resolve())
        if trace:
            params["ttft_trace"] = True
            trace.details["prompt_rows"] = prompt_rows if prompt_rows is not None else len(ids)
            trace.add("prepare_total_ms", (time.perf_counter_ns()-prepare_started)/1e6)
            trace.point("prepared")
        return params, stops + extra_stops

    def piece(self, token):
        if token not in self.pieces:
            self.pieces[token] = bytes(self.tokenizer.call({"op": "piece", "token": token})["bytes"])
        return self.pieces[token]

    def events(self, params, stops, chat=True, stream_parse=True, cancelled=lambda: False, trace=None):
        """Caller holds self.lock. Consume fully, or close generator to cancel/drain."""
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        text = ""
        previous = {"role": "assistant", "content": "", "reasoning_content": ""}
        waiting_ack = False
        waiting_grammar = False
        started = False
        completed = False
        tool_ids = {}
        prefill_cache = {}
        try:
            if cancelled():
                raise ClientDisconnected("client disconnected")
            if trace:
                trace.point("native_rpc_start")
            self.engine.write(params)
            started = True
            while True:
                event = self.engine.read()
                if event.get("event") == "error":
                    completed = True
                    raise RuntimeError(event["error"])
                if event["event"] == "sample":
                    waiting_grammar = True
                    if cancelled():
                        raise ClientDisconnected("client disconnected")
                    self.engine.write(self.tokenizer.call({"op": "grammar_check", "token": event["token"]}))
                    waiting_grammar = False
                elif event["event"] == "prefill":
                    prefill_cache = {key: event[key] for key in ("cached_n", "prompt_eval_n", "cache_hit") if key in event}
                    if trace:
                        trace.point("native_prefill_received")
                        trace.details["native_prefill"] = dict(event)
                    if cancelled():
                        raise ClientDisconnected("client disconnected")
                    yield event
                elif event["event"] == "token":
                    if trace:
                        trace.point("first_native_token")
                    waiting_ack = True
                    text += decoder.decode(self.piece(event["token"]))
                    safe_text, stop = clipped_text(text, stops)
                    if cancelled():
                        raise ClientDisconnected("client disconnected")
                    # Stop decisions precede ACK. The native next forward can overlap
                    # PEG work; cancellation still drains at most one token boundary.
                    self.engine.write({"continue": not stop})
                    waiting_ack = False
                    if not stream_parse:
                        continue
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
                    if delta:
                        yield {"event": "delta", "delta": delta}
                elif event["event"] == "done":
                    if prefill_cache:
                        event = {**event, "timings": {**event["timings"], **prefill_cache}}
                        # Report measured computation separately from reused logical tokens.
                        if "prompt_eval_n" in prefill_cache:
                            elapsed_ms = event["timings"].get("prompt_ms", 0)
                            event["timings"]["prompt_per_second"] = 1000.0 * prefill_cache["prompt_eval_n"] / elapsed_ms if elapsed_ms > 0 else 0
                    if trace:
                        trace.point("native_done")
                        trace.details["native_timings"] = dict(event.get("timings", {}))
                    if self.profile:
                        print("M31 request timings " + json.dumps(event.get("timings", {}), separators=(",", ":")), file=sys.stderr, flush=True)
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
            if started and not completed:
                # Grammar failure must unblock the native sample handshake before
                # waiting for its terminal error. Never invoke the failed helper
                # while draining, and never reuse an undrained channel.
                unwinding = sys.exc_info()[0] is not None
                try:
                    if waiting_grammar:
                        self.engine.write({"accepted": False, "allowed": []})
                    elif waiting_ack:
                        self.engine.write({"continue": False})
                    deadline = time.monotonic() + 30.0
                    for _ in range(16):
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise RuntimeError("engine cancellation timed out")
                        event = self.engine.read(timeout=remaining)
                        kind = event.get("event")
                        if kind in ("done", "error"):
                            break
                        if kind == "sample":
                            self.engine.write({"accepted": False, "allowed": []})
                        elif kind == "token":
                            self.engine.write({"continue": False})
                        elif kind != "prefill":
                            raise RuntimeError("unexpected engine event during cancellation")
                    else:
                        raise RuntimeError("engine cancellation exceeded event limit")
                except Exception as exc:
                    if hasattr(self.engine, "invalidate"):
                        self.engine.invalidate(f"cancellation failed: {exc}")
                    if not unwinding:
                        raise


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 60.0

    def disconnected(self):
        try:
            if not select.select([self.connection], [], [], 0)[0]:
                return False
            return not self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
        except (OSError, ValueError):
            return True

    @property
    def app(self):
        return self.server.app

    def json(self, status, value):
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        if status == 503:
            self.send_header("Retry-After", "1")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path in ("/health", "/healthz"):
            alive = all(rpc.process.poll() is None and rpc.failure is None for rpc in (self.app.engine, self.app.tokenizer))
            self.json(200 if alive else 503, {"status": "ok" if alive else "error"})
        elif self.path == "/v1/models":
            self.json(200, {"object": "list", "data": [{"id": self.app.model, "object": "model", "created": 0, "owned_by": "local"}]})
        else:
            self.json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        streaming = False
        trace = TTFTTrace(self.headers.get("X-M31-TTFT-ID")) if os.environ.get("M31_TTFT_TRACE") else None
        trace_error = None
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 64 * 1024 * 1024:
                raise ValueError("invalid request size")
            body_started = time.perf_counter_ns()
            raw_body = self.rfile.read(length)
            body_read = time.perf_counter_ns()
            body = json.loads(raw_body)
            if trace:
                trace.add("http_body_read_ms", (body_read-body_started)/1e6)
                trace.add("http_json_parse_ms", (time.perf_counter_ns()-body_read)/1e6)
                trace.details["http_body_bytes"] = length
                trace.point("http_body_parsed")
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
            queue_started = time.perf_counter_ns()
            with self.app.inference_slot():
                if trace:
                    trace.add("inference_queue_wait_ms", (time.perf_counter_ns()-queue_started)/1e6)
                params, stops = self.app.prepare(body, chat, trace=trace)
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
                        if trace and isinstance(value, dict):
                            delta = value.get("choices", [{}])[0].get("delta", {})
                            if not chat:
                                delta = {"content": value.get("choices", [{}])[0].get("text", "")}
                            if delta.get("content") or delta.get("reasoning_content"):
                                trace.point("first_generated_delta")
                            if delta.get("content"):
                                trace.point("first_content_delta")
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
                    with contextlib.closing(self.app.events(params, stops, chat, cancelled=self.disconnected, trace=trace)) as events:
                        for event in events:
                            if event["event"] == "delta":
                                emit(chunk(event["delta"]))
                            elif event["event"] == "done":
                                timings = event["timings"]
                                usage = usage_counts(timings)
                                emit(chunk({}, event["finish_reason"], timings, usage))
                    emit("[DONE]")
                else:
                    done = None
                    with contextlib.closing(self.app.events(params, stops, chat, stream_parse=False, cancelled=self.disconnected, trace=trace)) as events:
                        for event in events:
                            if event["event"] == "done":
                                done = event
                    timings = done["timings"]
                    usage = usage_counts(timings)
                    if native:
                        value = {"content": done["message"].get("content", ""), "stop": True,
                                 "stopped_eos": done["finish_reason"] == "stop", "tokens_predicted": timings["predicted_n"],
                                 "tokens_evaluated": timings.get("prompt_eval_n", timings["prompt_n"]), "timings": timings}
                    else:
                        choice = {"index": 0, "finish_reason": done["finish_reason"]}
                        choice["message" if chat else "text"] = done["message"] if chat else done["message"].get("content", "")
                        value = {"id": request_id, "object": "chat.completion" if chat else "text_completion",
                                 "created": created, "model": self.app.model, "choices": [choice], "usage": usage, "timings": timings}
                    self.json(200, value)
        except (BrokenPipeError, ConnectionResetError, ClientDisconnected, TimeoutError) as exc:
            trace_error = exc
            self.close_connection = True
        except Exception as exc:
            trace_error = exc
            if streaming:
                try:
                    self.wfile.write(("data: " + json.dumps({"error": {"message": str(exc)}}) + "\n\ndata: [DONE]\n\n").encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    pass
            else:
                status = 503 if isinstance(exc, ServerBusy) else 400 if isinstance(exc, (ValueError, KeyError, TypeError)) else 500
                with contextlib.suppress(BrokenPipeError, ConnectionResetError, TimeoutError):
                    self.json(status, {"error": {"message": str(exc), "type": "invalid_request_error" if status == 400 else "server_error"}})
        finally:
            if trace:
                try:
                    trace.finish(trace_error)
                except OSError as exc:
                    print("M31 TTFT trace write failed: " + str(exc), file=sys.stderr, flush=True)


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
    for flag in ("reuse-allocations", "persistent-cpu-pool", "cpu-residual", "cpu-decode-cache", "dense-decode-cache", "profile"):
        parser.add_argument("--" + flag, action="store_true")
    parser.add_argument("--vision-config", help="JSON: adapter/model/mmproj/cache_dir/backend/threads and optional limits")
    parser.add_argument("--prefix-cache", action="store_true", help="opt-in exact full-prompt reuse in the native worker")
    parser.add_argument("--dspark-model", default="", help="opt-in DSpark sidecar; greedy only, ordinary sampling fallback")
    parser.add_argument("--dspark-depth", type=int, default=3)
    parser.add_argument("--dspark-p-min", type=float, default=0.0)
    args = parser.parse_args()
    if not 1 <= args.dspark_depth <= 7 or not 0 <= args.dspark_p_min <= 1:
        parser.error("DSpark depth must be1..7 and p_min in0..1")
    if args.backend == "cuda" and os.environ.get("M31_CARD_SESSION") != "1":
        parser.error("CUDA requires an aictl-owned card session")
    argv = [args.engine, "--model", args.model, "--backend", args.backend, "--ctx", str(args.ctx),
            "--chunk", str(args.chunk), "--threads", str(args.threads), "--arena-gib", str(args.arena_gib),
            "--gpu-experts-gib", str(args.gpu_experts_gib), "--gpu-experts-min-tokens", str(args.gpu_experts_min_tokens)]
    for flag in ("reuse-allocations", "persistent-cpu-pool", "cpu-residual", "cpu-decode-cache", "dense-decode-cache", "profile"):
        argv += ["--" + flag, "1" if getattr(args, flag.replace("-", "_")) else "0"]
    if args.prefix_cache:
        argv += ["--prefix-cache", "1"]
    if args.dspark_model:
        argv += ["--dspark-model", args.dspark_model, "--dspark-depth", str(args.dspark_depth),
                 "--dspark-p-min", str(args.dspark_p_min)]
    vision = media_input = None
    prewarm_vision = False
    if args.vision_config:
        sys.path.insert(0, str(Path(__file__).resolve().with_name("serve")))
        from vision_http import MediaInput
        from vision_encoder import VisionEncoder
        options = json.loads(Path(args.vision_config).read_text())
        prewarm_vision = options.pop("prewarm", False)
        if not isinstance(prewarm_vision, bool):
            raise ValueError("vision prewarm must be boolean")
        limits = options.pop("limits", {})
        options.setdefault("model", args.model)
        media_input = MediaInput(options["cache_dir"], **limits)
        vision = VisionEncoder(**options)
    tokenizer = RPC([args.tokenizer, args.model], read_timeout=30.0)
    engine = None
    server = None
    try:
        engine = RPC(argv)
        if vision is not None and prewarm_vision:
            prewarm = vision.start()
            print("M31 vision prewarm " + json.dumps(prewarm), file=sys.stderr, flush=True)
        app = Application(engine, tokenizer, args.alias, profile=args.profile, vision=vision, media_input=media_input)
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
        if vision is not None:
            vision.close()


if __name__ == "__main__":
    main()
