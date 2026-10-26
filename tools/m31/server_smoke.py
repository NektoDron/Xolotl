#!/usr/bin/env python3
"""Short real M31 API smoke. Run from the coordinator's aictl-owned session.

Stores every HTTP request, raw response, SSE event and timing in a NEW JSON file.
No GPU allocation, model launch or aictl action is performed by this client.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request


class Smoke:
    def __init__(self, base, timeout):
        self.base, self.timeout = base, timeout
        self.requests = []
        self.checks = []

    def check(self, name, okay, detail=None):
        item = {"check": name, "ok": bool(okay)}
        if detail is not None:
            item["detail"] = detail
        self.checks.append(item)
        if not okay:
            raise AssertionError(f"{name}: {detail}")

    def request(self, name, path, body=None, stream=False, cancel=False):
        record = {"name": name, "method": "GET" if body is None else "POST", "path": path, "request": body}
        self.requests.append(record)
        payload = None if body is None else json.dumps(body, ensure_ascii=False).encode()
        request = urllib.request.Request(self.base + path, payload, {"Content-Type": "application/json"})
        start = time.monotonic()
        response = None
        try:
            try:
                response = urllib.request.urlopen(request, timeout=self.timeout)
            except urllib.error.HTTPError as exc:
                response = exc
            record["status"] = response.status
            record["headers"] = dict(response.headers.items())
            if stream and response.status == 200:
                events, lines = [], []
                record.update(events=events, raw_lines=lines)
                done = False
                cancelled = False
                first_delta_ms = None
                for raw in response:
                    line = raw.decode("utf-8").rstrip("\r\n")
                    lines.append(line)
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        done = True
                        break
                    if not data:
                        continue
                    event = json.loads(data)
                    events.append(event)
                    if "error" in event:
                        raise AssertionError(f"stream error: {event['error']}")
                    choices = event.get("choices", [])
                    delta = choices[0].get("delta", {}) if choices else {}
                    has_text = bool(delta.get("content") or delta.get("reasoning_content"))
                    if has_text and first_delta_ms is None:
                        first_delta_ms = 1000 * (time.monotonic() - start)
                    if cancel and has_text:
                        cancelled = True
                        break
                record.update(events=events, raw_lines=lines, done=done, cancelled=cancelled, first_delta_ms=first_delta_ms)
                return record
            raw = response.read().decode("utf-8")
            record["raw_response"] = raw
            try:
                record["response"] = json.loads(raw)
            except json.JSONDecodeError:
                record["response"] = None
            return record
        except Exception as exc:
            record["exception"] = str(exc)
            raise
        finally:
            if response is not None:
                response.close()
            record["wall_ms"] = 1000 * (time.monotonic() - start)

    def post_ok(self, name, path, body):
        record = self.request(name, path, body)
        self.check(name + ": HTTP200", record["status"] == 200, record.get("response"))
        self.check(name + ": JSON object", isinstance(record.get("response"), dict))
        return record["response"]

    def text_fields(self, name, message):
        self.check(name + ": assistant role", message.get("role") == "assistant", message)
        text = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        self.check(name + ": text fields", isinstance(text, str) and isinstance(reasoning, str), message)
        leaks = ("<mm:think>", "</mm:think>", "]~b]", "[e~[", "]<]minimax[>[")
        self.check(name + ": no service marker leak", not any(marker in text or marker in reasoning for marker in leaks), message)
        self.check(name + ": no unexpected tool calls", not message.get("tool_calls"), message)
        return text, reasoning

    def timings(self, name, response, maximum):
        usage, timings = response.get("usage", {}), response.get("timings", {})
        for field in ("prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms", "predicted_per_second"):
            self.check(name + ": timing " + field, isinstance(timings.get(field), (int, float)) and timings[field] >= 0, timings)
        self.check(name + ": prompt count", timings["prompt_n"] > 0 and usage.get("prompt_tokens") == timings["prompt_n"], response)
        self.check(name + ": generation count", 0 <= timings["predicted_n"] <= maximum and usage.get("completion_tokens") == timings["predicted_n"], response)
        self.check(name + ": total count", usage.get("total_tokens") == usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0), usage)
        expected_rate = 1000 * max(0, timings["predicted_n"] - 1) / timings["predicted_ms"] if timings["predicted_ms"] else 0
        self.check(name + ": reference TG numerator", abs(timings["predicted_per_second"] - expected_rate) < 1e-6 * max(1, expected_rate), timings)

    def body(self, text, maximum=64, stream=False):
        return {"messages": [{"role": "user", "content": text}], "temperature": 0, "max_tokens": maximum,
                "stream": stream, "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}

    def nonstream(self, name, text, answer, maximum=64):
        response = self.post_ok(name, "/v1/chat/completions", self.body(text, maximum))
        self.check(name + ": one choice", len(response.get("choices", [])) == 1, response)
        choice = response["choices"][0]
        content, _ = self.text_fields(name, choice.get("message", {}))
        self.check(name + ": finish reason", choice.get("finish_reason") in ("stop", "length"), choice)
        self.timings(name, response, maximum)
        if answer:
            self.check(name + ": expected answer", re.search(answer, content, re.I) is not None, content)
        else:
            self.check(name + ": empty zero-token response", not content and response["usage"]["completion_tokens"] == 0, response)
        return response

    def run(self):
        health = self.request("health", "/health")
        self.check("health HTTP200", health["status"] == 200, health.get("response"))
        self.check("health ready", health.get("response", {}).get("status") == "ok", health.get("response"))
        models = self.request("models", "/v1/models")
        self.check("models HTTP200", models["status"] == 200, models.get("response"))
        self.check("models populated", any(item.get("id") for item in models.get("response", {}).get("data", [])), models.get("response"))

        composed = self.post_ok("tokenize-NFC", "/tokenize", {"content": "Привет café 🐈"})["tokens"]
        decomposed = self.post_ok("tokenize-NFD", "/tokenize", {"content": "Привет cafe\u0301 🐈"})["tokens"]
        self.check("NFC token equality", composed == decomposed and bool(composed), {"NFC": composed, "NFD": decomposed})
        detokenized = self.post_ok("detokenize", "/detokenize", {"tokens": composed})["content"]
        self.check("Unicode roundtrip", detokenized == "Привет café 🐈", detokenized)

        self.nonstream("zero-token", "What is 17*23?", None, maximum=0)
        self.nonstream("nonstream-after-zero", "Сколько будет 17*23? Ответь только числом.", r"\b391\b")

        streamed = self.request("Canberra-SSE", "/v1/chat/completions",
                                self.body("What is the capital of Australia? Respond with the city name only.", stream=True), stream=True)
        self.check("SSE HTTP200", streamed["status"] == 200, streamed)
        self.check("SSE content type", "text/event-stream" in streamed["headers"].get("Content-Type", ""), streamed["headers"])
        self.check("SSE DONE", streamed.get("done"), streamed)
        message = {"role": "assistant", "content": "", "reasoning_content": ""}
        final = None
        for event in streamed.get("events", []):
            self.check("SSE one choice", len(event.get("choices", [])) == 1, event)
            choice = event["choices"][0]
            delta = choice.get("delta", {})
            for key in ("content", "reasoning_content"):
                value = delta.get(key) or ""
                self.check("SSE string delta", isinstance(value, str), event)
                message[key] += value
            self.check("SSE no tools", not delta.get("tool_calls"), event)
            if choice.get("finish_reason") is not None:
                self.check("SSE final reason", choice["finish_reason"] in ("stop", "length"), event)
                final = event
        self.check("SSE final event", final is not None, streamed)
        content, _ = self.text_fields("SSE assembled", message)
        self.check("SSE Canberra answer", "canberra" in content.lower(), message)
        self.timings("SSE", final, 64)
        streamed["assembled_message"] = message

        cancelled = self.request("client-cancel", "/v1/chat/completions",
                                 self.body("Сколько будет 17*23? Ответь только числом.", stream=True), stream=True, cancel=True)
        self.check("cancel response HTTP200", cancelled["status"] == 200, cancelled)
        self.check("cancel after text delta", cancelled.get("cancelled"), cancelled)
        self.nonstream("request-after-cancel", "What is 17 times 23? Reply with only the number.", r"\b391\b")

        vision = self.request("reject-vision", "/v1/chat/completions", {
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}],
            "max_tokens": 1})
        self.check("vision explicitly rejected", vision["status"] == 400 and "vision" in str(vision.get("response", {})).lower(), vision)
        last_health = self.request("health-after-smoke", "/health")
        self.check("health after smoke", last_health["status"] == 200 and last_health.get("response", {}).get("status") == "ok", last_health)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8180)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be a NEW file")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the path atomically before any requests; never overwrite a previous report.
    with args.out.open("x", encoding="utf-8") as output:
        smoke = Smoke(f"http://127.0.0.1:{args.port}", args.timeout)
        error = None
        try:
            smoke.run()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        result = {"format": "m31-api-smoke-v1", "base": smoke.base, "ok": error is None,
                  "error": error, "checks": smoke.checks, "requests": smoke.requests}
        json.dump(result, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
        output.flush()
    print(json.dumps({"ok": result["ok"], "checks": len(smoke.checks), "requests": len(smoke.requests), "error": error, "out": str(args.out)}, ensure_ascii=False))
    return 0 if error is None else 1


if __name__ == "__main__":
    sys.exit(main())
