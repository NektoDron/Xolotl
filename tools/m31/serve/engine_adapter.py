"""Opt-in M31 JSON controller implementing serve.server's Python Engine boundary.

This does not pretend that m31_serve speaks StrataEngine's GEN/GENI protocol.
The accepted standalone adapter owns the shared bounded RPC transport.
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
from pathlib import Path
import queue
import sys
import threading
import time

_spec = importlib.util.spec_from_file_location(
    "m31_standalone_transport", Path(__file__).resolve().parents[1] / "serve_m31.py")
standalone = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(standalone)


class PreparedIds(list):
    """Prompt IDs plus the immutable helper request to replay under the FIFO."""
    def __init__(self, ids, request, rendered, mixed_manifest=None):
        super().__init__(ids)
        # Caller dictionaries may be reused or mutated after prepare().
        self.chat_request = json.loads(json.dumps(request))
        self.rendered = json.loads(json.dumps(rendered))
        self.mixed_manifest = str(Path(mixed_manifest).resolve()) if mixed_manifest else None


class PollingRPC(standalone.RPC):
    def poll(self, timeout):
        """Empty is a heartbeat, not a transport timeout; controller owns deadline."""
        self._healthy()
        try:
            item = self.lines.get(timeout=timeout)
        except queue.Empty:
            return None
        try:
            if isinstance(item, Exception):
                raise item
            value = json.loads(item)
            if not isinstance(value, dict):
                raise ValueError("helper RPC response must be an object")
            return value
        except Exception as exc:
            self.invalidate(exc)
            raise RuntimeError(f"M31 RPC frame failed: {exc}") from exc


class M31Engine:
    batch = 0
    starting = False
    unloaded = False

    def __init__(self, rpc, grammar_helper, silence_s=300.0, finalize_parser=None):
        self.rpc, self.grammar = rpc, grammar_helper
        self.max_context = self.known_ctx = int(rpc.ready["context"])
        if int(rpc.ready["vocab"]) != int(grammar_helper.ready["vocab"]):
            raise ValueError("M31 engine/helper vocabulary mismatch")
        self.silence_s = silence_s
        self.last = {}
        self.progress, self.progress_ms, self.prefill_tok_s_mean = None, 0, None
        self.reused = 0
        self.info = {"model": "minimax-m3.1", "protocol": "m31-json-python-adapter-v1",
                     "prefix_cache": False, "vision": False, "grammar": True}
        self.controller = threading.Lock()
        self.finalize_parser = finalize_parser

    def alive(self):
        return (getattr(self.rpc, "failure", None) is None and
                self.rpc.process.poll() is None and
                getattr(self.grammar, "failure", None) is None and
                self.grammar.process.poll() is None)

    def exit_code(self):
        return self.rpc.process.poll()

    def death_note(self):
        return getattr(self.rpc, "failure", None) or getattr(self.grammar, "failure", None) or ""

    def restart(self):
        from serve.server import EngineDied
        raise EngineDied("M31 common adapter prototype needs an explicit process restart")

    def close(self):
        self.rpc.close()

    unload = close

    def _done(self, event, cancelled=False, finalize=True):
        # Common Service calls parser.finish() after releasing its FIFO. Capture
        # the PEG final parse now, while this request still owns helper state.
        if finalize and self.finalize_parser is not None:
            self.finalize_parser()
        timings = event.get("timings", {})
        self.last = {"generated": timings.get("predicted_n", 0),
                     "prompt_tokens": timings.get("prompt_n", 0),
                     "prompt_read": timings.get("prompt_n", 0),
                     "prompt_ms": timings.get("prompt_ms", 0),
                     "decode_ms": timings.get("predicted_ms", 0),
                     "finish": "cancel" if cancelled else event.get("finish_reason", "length"),
                     "reused": 0}

    def _drain(self, waiting_grammar, waiting_ack, finalize=True):
        try:
            if waiting_grammar:
                self.rpc.write({"accepted": False, "allowed": []})
            elif waiting_ack:
                self.rpc.write({"continue": False})
            deadline = time.monotonic() + 30.0
            for _ in range(16):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("M31 cancellation timed out")
                event = self.rpc.read(timeout=remaining)
                kind = event.get("event")
                if kind == "done":
                    self._done(event, cancelled=True, finalize=finalize)
                    return
                if kind == "error":
                    if finalize and self.finalize_parser is not None:
                        self.finalize_parser()
                    return
                if kind == "sample":
                    self.rpc.write({"accepted": False, "allowed": []})
                elif kind == "token":
                    self.rpc.write({"continue": False})
                elif kind != "prefill":
                    raise RuntimeError("unexpected M31 cancellation event")
            raise RuntimeError("M31 cancellation exceeded event limit")
        except Exception as exc:
            self.rpc.invalidate(f"M31 cancellation failed: {exc}")
            from serve.server import EngineDied
            raise EngineDied(str(exc)) from exc

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        from serve.server import EngineDied
        if embeddings is not None:
            raise ValueError("M31 mixed embeddings/GENI are not connected in this prototype")
        if not isinstance(ids, PreparedIds):
            raise ValueError("M31 generation needs prepared chat IDs; raw continuations are not supported")
        if not self.alive():
            raise EngineDied("M31 helper or engine is unavailable")
        for key in ("logit_bias", "strata_prefix", "strata_score", "strata_img_keys"):
            if (sampling or {}).get(key):
                raise ValueError(f"{key} is not connected in the M31 common prototype")
        for key in ("repetition_penalty", "repeat_penalty"):
            value = (sampling or {}).get(key)
            if value is not None and value != 1:
                raise ValueError(f"{key} other than 1 is not implemented for M31")
        for key in ("penalty_last_n", "repeat_last_n"):
            if (sampling or {}).get(key) is not None:
                raise ValueError(f"{key} is not implemented for M31")
        if not self.controller.acquire(timeout=1.0):
            raise RuntimeError("M31 controller is already active")
        started = completed = waiting_ack = waiting_grammar = False
        try:
            if cancel.is_set():
                return
            # Render-helper activity from another queued request must never reset
            # the grammar/parser of this generation. Replay on the second helper.
            rendered = self.grammar.call(ids.chat_request)
            expected_tokens = ids.rendered.get("tokens", list(ids)) if ids.mixed_manifest else list(ids)
            if rendered["tokens"] != expected_tokens:
                raise RuntimeError("M31 template/token IDs changed between prepare and generation")
            params, _ = standalone.request_parameters(
                {**(sampling or {}), "n_predict": int(max_new)}, self.max_context, len(ids))
            params.update(tokens=expected_tokens, interactive=True,
                          grammar_check=bool(rendered.get("grammar_active")),
                          stop_ids=self.grammar.ready.get("eog", []),
                          suppress_ids=self.grammar.ready.get("suppress", []))
            if ids.mixed_manifest:
                params["mixed_manifest"] = ids.mixed_manifest
            self.progress, self.progress_ms = None, 0
            self.rpc.write(params)
            started = True
            heard = beat = time.monotonic()
            while True:
                if cancel.is_set():
                    return
                event = self.rpc.poll(0.25)
                now = time.monotonic()
                if event is None:
                    if now - heard >= self.silence_s:
                        self.rpc.invalidate("M31 engine RPC timed out")
                        raise EngineDied("M31 engine RPC timed out")
                    if now - beat >= 10:
                        beat = now
                        yield None
                    continue
                heard = beat = now
                kind = event.get("event")
                if kind == "sample":
                    waiting_grammar = True
                    self.rpc.write(self.grammar.call({"op": "grammar_check", "token": event["token"]}))
                    waiting_grammar = False
                elif kind == "prefill":
                    self.progress = (event["prompt_n"], len(ids))
                    self.progress_ms = event.get("prompt_ms", 0)
                    self.prefill_tok_s_mean = 1000 * len(ids) / max(self.progress_ms, 1e-6)
                    yield None
                elif kind == "token":
                    waiting_ack = True
                    if cancel.is_set():
                        return
                    # Let the next forward overlap the frontend's PEG parsing.
                    self.rpc.write({"continue": True})
                    waiting_ack = False
                    yield int(event["token"])
                elif kind == "done":
                    completed = True
                    self._done(event)
                    # Native M31 consumes EOS internally; common Service counts
                    # and recognizes it at the Engine.generate token boundary.
                    if event.get("finish_reason") == "stop" and self.grammar.ready.get("eog"):
                        yield int(self.grammar.ready["eog"][0])
                    return
                elif kind == "error":
                    completed = True
                    raise ValueError(event.get("error", "M31 engine error"))
                else:
                    raise RuntimeError(f"unexpected M31 event: {event}")
        except EngineDied:
            raise
        except RuntimeError as exc:
            # The common HTTP/SSE layer handles EngineDied as a backend error;
            # an unclassified RuntimeError would otherwise drop its connection.
            raise EngineDied(str(exc)) from exc
        finally:
            try:
                if started and not completed:
                    # Transport failures are already invalidated; never consume a
                    # stale frame or replace their original exception by a drain.
                    if getattr(self.rpc, "failure", None) is None:
                        exception = sys.exc_info()[0]
                        self._drain(waiting_grammar, waiting_ack,
                                    finalize=exception in (None, GeneratorExit))
            finally:
                self.controller.release()
