#!/usr/bin/env python3
"""Opt-in M31 shim for the repository's common OpenAI/Anthropic server.

Text prototype; the accepted standalone tools/m31/serve_m31.py stays intact.
The tokenizer, template, and PEG parser remain the pinned PR vocab helper.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import threading

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from engine_adapter import M31Engine, PollingRPC, PreparedIds, standalone  # noqa: E402
from serve import server as S  # noqa: E402
from serve.frontend import Event, ToolCall, images_of  # noqa: E402


def helper_call(rpc, request):
    try:
        return rpc.call(request)
    except RuntimeError as exc:
        if getattr(rpc, "failure", None) is not None or rpc.process.poll() is not None:
            raise S.EngineDied(str(exc)) from exc
        raise ValueError(str(exc)) from exc


def openai_tools(tools):
    return [tool if tool.get("type") == "function" else {"type": "function", "function": tool}
            for tool in (tools or [])]


def chat_request(messages, tools, kwargs, req=None):
    req = req or {}
    standalone.validate_messages(messages)
    if images_of(messages):
        raise ValueError("M31 common text prototype does not support images/video yet")
    raw_choice = req.get("tool_choice", "auto")
    choice = S.tool_choice_of(raw_choice)[0]
    if choice not in ("auto", "none"):
        raise ValueError("M31 common prototype supports tool_choice auto/none only")
    if not isinstance(kwargs.get("enable_thinking", True), bool):
        raise ValueError("enable_thinking must be boolean")
    parallel = req.get("parallel_tool_calls", True)
    if isinstance(raw_choice, dict) and "disable_parallel_tool_use" in raw_choice:
        if not isinstance(raw_choice["disable_parallel_tool_use"], bool):
            raise ValueError("disable_parallel_tool_use must be boolean")
        parallel = not raw_choice["disable_parallel_tool_use"]
    if not isinstance(parallel, bool):
        raise ValueError("parallel_tool_calls must be boolean")
    return {"op": "chat", "messages": messages, "tools": openai_tools(tools),
            "tool_choice": choice, "enable_thinking": kwargs.get("enable_thinking", True),
            "parallel_tool_calls": parallel,
            "chat_template_kwargs": dict(kwargs)}


class HelperTokenizer:
    def __init__(self, rpc):
        self.rpc = rpc
        self.vocab_size = int(rpc.ready["vocab"])
        self.control_tokens = ()
        self.pieces = {}

    def encode(self, text, parse_special=False, plain=None):
        if plain:
            raise ValueError("raw plain-span encoding is not connected; M31 chat uses the PR renderer")
        return helper_call(self.rpc, {"op": "tokenize", "content": text,
                                      "add_special": False, "parse_special": parse_special})["tokens"]

    def decode(self, ids):
        return helper_call(self.rpc, {"op": "detokenize", "tokens": list(ids)})["content"]

    def token_bytes(self, token):
        if token not in self.pieces:
            self.pieces[token] = bytes(helper_call(self.rpc, {"op": "piece", "token": token})["bytes"])
        return self.pieces[token]


class HelperTemplate:
    # These describe the PR helper, not Qwen syntax detection in ChatTemplate.
    caps = {"supports_tools": True, "supports_tool_calls": True, "supports_system_role": True,
            "supports_parallel_tool_calls": True, "supports_preserve_reasoning": True}

    def __init__(self, rpc):
        self.rpc = rpc
        self.source = rpc.ready.get("template", "")

    def render(self, messages, tools=None, add_generation_prompt=True, **kwargs):
        if not add_generation_prompt:
            raise ValueError("M31 helper does not expose history-only/assistant-prefill rendering")
        return helper_call(self.rpc, chat_request(messages, tools, kwargs))["prompt"]


class HelperOutputParser:
    """Translate audited PEG message deltas into common frontend Events."""
    def __init__(self, rpc, thinking=True, tools=None, stream_tools=False, recover=False):
        if recover:
            raise ValueError("Qwen tool recovery is not connected for M31")
        self.rpc = rpc
        self.text = ""
        self.previous = {"content": "", "reasoning_content": ""}
        self.state = "reasoning" if thinking else "content"
        self.buf, self.pending = "", []
        self.rescued = self.refused = 0
        self.final_events = None

    def _events(self, partial):
        message = helper_call(self.rpc, {"op": "parse", "content": self.text, "partial": partial})["message"]
        events = []
        for key, kind in (("reasoning_content", "reasoning"), ("content", "content")):
            current, previous = message.get(key) or "", self.previous.get(key) or ""
            if not current.startswith(previous):
                raise S.EngineDied(f"M31 PEG parser retracted streamed {key}")
            if current != previous:
                events.append(Event(kind, current[len(previous):]))
        self.previous = message
        if message.get("content") or "</mm:think>" in self.text:
            self.state = "content"
        if not partial:
            for call in message.get("tool_calls", []):
                function = call["function"]
                arguments = function.get("arguments", "{}")
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("M31 function arguments must be a JSON object")
                fields = {"name": function["name"], "arguments": arguments}
                if call.get("id"):
                    fields["id"] = call["id"]
                events.append(Event("tool_call", call=ToolCall(**fields)))
        return events

    def feed(self, delta):
        self.text += delta
        return self._events(partial=True)

    def finish(self, reason=None):
        if self.final_events is None:
            raise RuntimeError("M31 parser was not finalized under the engine controller")
        return self.final_events

    def complete(self):
        if self.final_events is None:
            self.final_events = self._events(partial=False)


class M31Service(S.Service):
    def __init__(self, engine, render_rpc, grammar_rpc, model_name="m31-xolotl-common"):
        self.render_rpc = render_rpc
        tokenizer = HelperTokenizer(render_rpc)
        super().__init__(engine, tokenizer, HelperTemplate(render_rpc), model_name=model_name,
                         fit_max_tokens=True)
        self.stop_ids = set(grammar_rpc.ready.get("eog", []))
        self.render_lock = threading.Lock()

    def encode_prompt(self, messages, tools, kwargs):
        # Token-count endpoints must use the same PR chat token IDs, including
        # its own special-token escaping, rather than Qwen's literal-tag shim.
        with self.render_lock:
            return helper_call(self.render_rpc, chat_request(messages, tools, kwargs))["tokens"]

    def prepare(self, messages, tools, kwargs, max_new=None, force=None, req=None, cancel=None):
        req = req or {}
        if force:
            raise ValueError("forced Qwen-style tool openings are not supported by M31")
        for key in ("response_format", "grammar", "strata_prefix", "reasoning_budget_tokens",
                    "thinking_budget", "reasoning_loop_recovery"):
            if req.get(key):
                raise ValueError(f"{key} is not connected in the M31 common prototype")
        if isinstance(req.get("thinking"), dict) and req["thinking"].get("budget_tokens"):
            raise ValueError("thinking.budget_tokens continuation is not connected for M31")
        if cancel is not None and cancel.is_set():
            raise S.ClientGone()
        request = chat_request(messages, tools, kwargs, req)
        # This helper is separate from the active generation helper. Queueing a
        # new HTTP request cannot reset the active request's PEG/grammar state.
        with self.render_lock:
            rendered = helper_call(self.render_rpc, request)
        ids = PreparedIds(rendered["tokens"], request, rendered)
        if not ids:
            raise ValueError("empty M31 prompt")
        # Common server's convention: omitted/0/-1 output limit means remaining
        # context. Explicit positive limits are clamped, as the native adapter.
        limit = -1 if max_new is None or max_new <= 0 else max_new
        params, _ = standalone.request_parameters({"n_predict": limit}, self.engine.max_context, len(ids))
        self.embeddings.path = self.embeddings.keys = None
        return ids, kwargs.get("enable_thinking", True), params["max_tokens"]

    def run(self, ids, thinking, tools, max_new, sampling, cancel, force=None):
        additional = ids.rendered.get("additional_stops", []) if isinstance(ids, PreparedIds) else []
        if additional:
            sampling = {**(sampling or {}),
                        "stop": list(dict.fromkeys([*S.stop_strings(sampling), *additional]))}
        yield from super().run(ids, thinking, tools, max_new, sampling, cancel, force)

    def resolve_prefix(self, *args, **kwargs):
        raise ValueError("M31 prefix caching is not connected in this prototype")


def make_service(engine_rpc, render_rpc, grammar_rpc, model_name="m31-xolotl-common"):
    parser_context = threading.local()

    def parser_factory(**kwargs):
        parser = HelperOutputParser(grammar_rpc, **kwargs)
        parser_context.parser = parser
        return parser

    def finalize_parser():
        parser = getattr(parser_context, "parser", None)
        if parser is not None:
            parser.complete()

    engine = M31Engine(engine_rpc, grammar_rpc, finalize_parser=finalize_parser)
    # Like the model-specific GLM shim, this is process-local registration. No
    # common source or other model's server process is modified.
    S.OutputParser = parser_factory
    return M31Service(engine, render_rpc, grammar_rpc, model_name)


def runtime_plan(config, config_directory):
    """Resolve an optional portable recipe without starting any process."""
    profile = config.get("runtime_profile")
    if not profile:
        return config["engine_argv"], {}, []
    if "engine_argv" in config:
        raise ValueError("use runtime_profile with engine/model, or engine_argv, not both")
    from tools.m31.profile_plan import load_profile, make_plan
    path = Path(profile)
    if not path.is_absolute():
        path = Path(config_directory) / path
    plan = make_plan(load_profile(path), config["engine"], config["model"])
    return plan["argv"], plan["environment"], plan["unset_environment"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True,
                        help="JSON with engine_argv, tokenizer_argv, optional model_name/API key")
    parser.add_argument("--host", choices=["127.0.0.1", "localhost"], default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8181)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    argv, environment, unset_environment = runtime_plan(config, config_path.parent)
    tokenizer_argv = config["tokenizer_argv"]
    if (not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv) or
            not isinstance(tokenizer_argv, list) or not tokenizer_argv or
            not all(isinstance(x, str) for x in tokenizer_argv)):
        parser.error("engine_argv/tokenizer_argv must be nonempty arrays of strings")
    if "cuda" in argv and os.environ.get("M31_CARD_SESSION") != "1":
        parser.error("CUDA requires an aictl-owned card session")
    # This is an isolated launcher process, before any helper has started. Its
    # children inherit exactly the selected native recipe; no shell is involved.
    for key in unset_environment:
        os.environ.pop(key, None)
    os.environ.update(environment)
    render = grammar = engine_rpc = http = None
    stopped = threading.Event()
    try:
        render = PollingRPC(tokenizer_argv, read_timeout=30)
        grammar = PollingRPC(tokenizer_argv, read_timeout=30)
        engine_rpc = PollingRPC(argv)
        service = make_service(engine_rpc, render, grammar, config.get("model_name", "m31-xolotl-common"))
        service.api_key = S.api_key_of(config.get("api_key", ""))
        http = S.serve(service, args.host, args.port)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: stopped.set())
        print(f"M31 common server ready: http://{args.host}:{http.server_address[1]}", flush=True)
        stopped.wait()
    finally:
        if http:
            http.shutdown()
            http.server_close()
        for rpc in (engine_rpc, grammar, render):
            if rpc:
                rpc.close()


if __name__ == "__main__":
    main()
