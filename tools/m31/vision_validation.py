#!/usr/bin/env python3
"""Deterministic media fixtures and actual HTTP/SSE vision acceptance probes.

prepare only writes owned fixtures/plans. http measures end-to-end first SSE
token latency, saves complete responses and checks visible facts. No GPU launch.
"""
from __future__ import annotations

import argparse
import base64
import http.client
import json
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

MODEL = "/mnt/ai/models/MiniMax-M3.1-Flash-GGUF/UD-IQ3_S/MiniMax-M3.1-Flash-UD-IQ3_S-00001-of-00005.gguf"
MMPROJ = "/mnt/ai/models/MiniMax-M3.1-Flash-GGUF/mmproj-BF16.gguf"
REFERENCE = Path("/mnt/ai/logs/m31-ref/media")


def image(path):
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()}}


def text(value):
    return {"type": "text", "text": value}


def prepare(directory):
    from PIL import Image, ImageDraw, ImageFont
    directory.mkdir(parents=True, exist_ok=False)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 72)
    canvas = Image.new("RGB", (640, 360), "white")
    ImageDraw.Draw(canvas).text((65, 135), "M31-7429", fill="black", font=font)
    canvas.save(directory / "ocr.png")
    canvas = Image.new("RGB", (640, 360), "white")
    ImageDraw.Draw(canvas).rectangle((240, 100, 400, 260), fill=(0, 180, 0))
    canvas.save(directory / "green-square.png")
    canvas = Image.new("RGB", (640, 360), "white")
    drawing = ImageDraw.Draw(canvas)
    for x in (80, 260, 440):
        drawing.ellipse((x, 120, x + 100, 220), fill="orange")
    canvas.save(directory / "count-three.png")
    ref = image(REFERENCE / "0.png")
    cases = [
        {"name": "ref-image", "messages": [{"role": "user", "content": [ref, text("Describe the shape, its color, location, and the digit shown. Answer in English.")]}],
         "expect": ["red", "circle", r"\b1\b|\bone\b"], "ref_ttft_s": 5.238},
        {"name": "ref-video", "messages": [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "file://" + str(REFERENCE / "shapes-12s.mp4")}}, text("Briefly describe all three stages in chronological order: colors, shapes and positions. Answer in English.")]}],
         "expect": ["red", "circle", "green", "square", "blue", "triangle"], "ordered": ["red", "green", "blue"], "ref_ttft_s": 36.171},
        {"name": "ocr", "messages": [{"role": "user", "content": [image(directory / "ocr.png"), text("Read the text exactly. Give only the text.")]}], "expect": ["M31-7429"]},
        {"name": "multiple-images", "messages": [{"role": "user", "content": [ref, image(directory / "green-square.png"), text("Describe the first and second images separately, in this order. Name colors and shapes. Answer in English.")]}], "expect": ["red", "circle", "green", "square"], "ordered": ["red", "green"]},
        {"name": "image-in-dialogue", "messages": [{"role": "user", "content": "I will send a picture and later ask about it."}, {"role": "assistant", "content": "Ready."}, {"role": "user", "content": [image(directory / "green-square.png"), text("Remember this picture for my next question.")]}, {"role": "assistant", "content": "I have seen the picture. What would you like to know?"}, {"role": "user", "content": "What color and shape was shown in the earlier picture? Answer in English."}], "expect": ["green", "square"]},
        {"name": "count-shapes", "messages": [{"role": "user", "content": [image(directory / "count-three.png"), text("How many orange circles are visible? Answer with the number and shape in English.")]}], "expect": [r"\b3\b|\bthree\b", "circle"]},
    ]
    (directory / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")
    return {"directory": str(directory), "cases": [case["name"] for case in cases]}


def check(case, content):
    missing = [pattern for pattern in case["expect"] if not re.search(pattern, content, re.I)]
    ordered = case.get("ordered", [])
    indices = [content.casefold().find(word) for word in ordered]
    order_ok = all(index >= 0 for index in indices) and indices == sorted(indices)
    return {"pass": not missing and order_ok, "missing": missing, "order_ok": order_ok}


def request(endpoint, body, timeout):
    url = urlsplit(endpoint)
    if url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost"):
        raise ValueError("validation endpoint must be a local HTTP server")
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=timeout)
    started = time.perf_counter()
    first = visible_first = None
    content, reasoning, events = [], [], []
    try:
        connection.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {response.read().decode(errors='replace')}")
        done = False
        while line := response.readline():
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                done = True
                break
            event = json.loads(payload)
            events.append(event)
            if event.get("error"):
                raise RuntimeError(str(event["error"]))
            delta = event.get("choices", [{}])[0].get("delta", {})
            if delta.get("content") or delta.get("reasoning_content"):
                if first is None:
                    first = time.perf_counter() - started
            if delta.get("content"):
                if visible_first is None:
                    visible_first = time.perf_counter() - started
                content.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning.append(delta["reasoning_content"])
        if not done or first is None:
            raise RuntimeError("SSE stream missing DONE or generated text")
        return {"ttft_s": first, "visible_ttft_s": visible_first, "wall_s": time.perf_counter() - started,
                "content": "".join(content), "reasoning_content": "".join(reasoning), "events": events}
    finally:
        connection.close()


def run_http(cases_path, endpoint, model, output, timeout, only):
    cases = json.loads(cases_path.read_text())
    results = []
    for case in cases:
        if only and case["name"] not in only:
            continue
        body = {"model": model, "messages": case["messages"], "temperature": 0, "top_p": 1,
                "seed": 0, "stream": True, "max_tokens": 1200,
                "reasoning_effort": "low", "video_fps": 1}
        result = {"case": case["name"]}
        try:
            result.update(request(endpoint, body, timeout))
            result.update(check(case, result["content"]))
            if "ref_ttft_s" in case:
                result.update(ref_ttft_s=case["ref_ttft_s"], speedup=case["ref_ttft_s"] / result["ttft_s"])
        except Exception as error:
            result.update({"pass": False, "error": f"{type(error).__name__}: {error}"})
        results.append(result)
        # Preserve partial results after every request, including failed requests.
        output.write_text(json.dumps({"format": "m31-vision-http-validation-v1", "pass": all(r["pass"] for r in results), "cases": results}, ensure_ascii=False, indent=2) + "\n")
    if not results:
        raise ValueError("no cases selected")
    return {"pass": all(r["pass"] for r in results), "cases": len(results), "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("directory", type=Path)
    http = commands.add_parser("http")
    http.add_argument("cases", type=Path)
    http.add_argument("--endpoint", default="http://127.0.0.1:8181")
    http.add_argument("--model", default="m31-vision")
    http.add_argument("--output", required=True, type=Path)
    http.add_argument("--timeout", type=float, default=600)
    http.add_argument("--only", action="append")
    args = parser.parse_args()
    result = prepare(args.directory.resolve()) if args.mode == "prepare" else run_http(args.cases, args.endpoint, args.model, args.output, args.timeout, args.only)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("pass", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
