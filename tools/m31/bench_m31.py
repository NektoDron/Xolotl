#!/usr/bin/env python3
"""Prepare the existing aiserver benchmark commands; summarize their saved outputs.

No inference, aictl changes, shell execution, or model reads happen in this helper.
The coordinator runs the plan's argv entries inside the authorized card session.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


REFERENCE = {
    "source": "/mnt/ai/logs/m31-ref/REPORT.md",
    "arm": "m31-ref-1g-q8-64-u2048-t64-e3",
    "tg_tps": {"4K": 23.33, "16K": 22.63, "32K": 21.92},
    "pp_tps": {"4K": 425.0, "32K": 433.0},
    "gsm8k": [96, 100], "tools": [19, 23],
    "configuration": "GPU0, q8_0/64K, ubatch2048, CPU64/96, 3 GPU expert layers",
}
DEPTHS = ("4K", "16K", "32K")
FIELDS = ("arm", "depth", "prompt_n", "prefill_tps", "prefill_s", "gen_n", "gen_tps", "wall_s", "finish")


def fingerprint(path):
    path = Path(path)
    if not path.is_file():
        return {"path": str(path), "exists": False}
    # Inputs here are scripts, corpus, and reports, never multi-GB model weights.
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def plan(args):
    if not args.arm or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.arm):
        raise ValueError("arm must contain only letters, digits, '-' and '_'")
    if not 1 <= args.port <= 65535:
        raise ValueError("invalid port")
    out = Path(args.directory).resolve()
    speed = Path(args.home) / "om_bench.py"
    quality = Path(args.home) / "eval_duty_1023.py"
    return {
        "format": "m31-bench-plan-v1", "arm": args.arm, "port": args.port,
        "reference": REFERENCE,
        "execution": "Sequentially inside systemd-run User=nekto and the aictl card-session worker; no builds/conversions during measurements.",
        "required_server": {
            "endpoints": ["/tokenize", "/detokenize", "/v1/chat/completions"],
            "timings": ["prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_per_second"],
            "options": ["ignore_eos", "cache_prompt=false", "chat_template_kwargs.enable_thinking=false", "tools"],
        },
        "commands": [
            {"name": "smoke", "env": {}, "argv": [sys.executable, str(speed), "smoke", args.arm, str(args.port), str(out / "smoke.json")]},
            {"name": "speed", "env": {"OM_OFF": '{"chat_template_kwargs":{"enable_thinking":false}}'},
             "argv": [sys.executable, str(speed), "speed", args.arm, str(args.port), str(out / "speed.tsv"),
                      "warmup,4K,16K,32K,4K,16K,32K"]},
            {"name": "quality", "env": {"EVAL_TEMP": "1.0", "EVAL_TOPP": "0.95", "EVAL_MAX": "6000",
                                         "EVAL_TOPK": "", "EVAL_MINP": ""},
             "argv": [sys.executable, str(quality), args.arm, str(args.port), str(out), "100"]},
        ],
        "inputs": [fingerprint(p) for p in [speed, quality, "/mnt/ai/logs/q38bench-corpus.txt", REFERENCE["source"]]],
        "expected_outputs": [str(out / "smoke.json"), str(out / "speed.tsv"), str(out / (args.arm + ".jsonl"))],
        "notes": [
            "Create a fresh output directory before execution; existing scripts append speed rows and overwrite quality output.",
            "The random prefix disables prefix reuse; prompt_n includes chat/template overhead and can exceed the nominal depth.",
            "Raw timings and both repetitions must be retained; reference speed is the report's rounded value.",
            "EVAL_TEMP=1.0 and EVAL_TOPP=0.95 override the eval script's misleading header/default temperature=0.",
            "Tools are checked as JSON; the evaluation script never executes tool calls.",
        ],
    }


def finite(value, field, line):
    x = float(value)
    if not math.isfinite(x) or x <= 0:
        raise ValueError(f"line {line}: {field} must be finite and positive")
    return x


def summarize(args):
    grouped = {}
    rows = []
    for path in args.speed:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if reader.fieldnames != list(FIELDS):
                raise ValueError(f"{path}: unexpected TSV columns")
            for line, row in enumerate(reader, 2):
                if row["depth"] == "warmup":
                    continue
                if row["depth"] not in DEPTHS:
                    raise ValueError(f"{path}:{line}: unsupported depth {row['depth']}")
                if row["finish"].split()[0] != "length":
                    raise ValueError(f"{path}:{line}: ignore_eos benchmark did not finish by length")
                numeric = {k: finite(row[k], k, line) for k in FIELDS[2:-1]}
                if numeric["gen_n"] != 256:
                    raise ValueError(f"{path}:{line}: expected 256 generated tokens")
                row = {**row, **numeric, "source": str(path), "line": line}
                grouped.setdefault(row["arm"], {}).setdefault(row["depth"], []).append(row)
                rows.append(row)
    if not rows:
        raise ValueError("no measured speed rows")
    arms = {}
    for arm, depths in grouped.items():
        summaries = {}
        for depth, values in depths.items():
            pp = statistics.median(v["prefill_tps"] for v in values)
            tg = statistics.median(v["gen_tps"] for v in values)
            summaries[depth] = {"repetitions": len(values), "pp_tps_median": pp, "tg_tps_median": tg,
                                "tg_ratio_to_ref": tg / REFERENCE["tg_tps"][depth],
                                "pp_ratio_to_ref": pp / REFERENCE["pp_tps"][depth] if depth in REFERENCE["pp_tps"] else None,
                                "prompt_tokens": [int(v["prompt_n"]) for v in values]}
        arms[arm] = {"depths": summaries,
                     "complete_two_repetitions": all(len(depths.get(d, [])) >= 2 for d in DEPTHS)}
    quality = []
    for path in args.quality:
        kinds = {"gsm8k": [], "tool": [], "tool_follow": []}
        seen = set()
        with open(path, encoding="utf-8") as f:
            for line, text in enumerate(f, 1):
                row = json.loads(text)
                key = (row["kind"], row["id"])
                if key[0] not in kinds or key in seen or not isinstance(row.get("ok"), bool):
                    raise ValueError(f"{path}:{line}: invalid or duplicated evaluation row")
                seen.add(key)
                kinds[key[0]].append(row)
        expected = {("gsm8k", i) for i in range(100)} | {("tool", i) for i in range(20)} | {("tool_follow", i) for i in range(3)}
        gsm = kinds["gsm8k"]
        tool = kinds["tool"] + kinds["tool_follow"]
        quality.append({"source": str(path), "complete": seen == expected,
                        "gsm8k": [sum(r["ok"] for r in gsm), len(gsm)],
                        "tools": [sum(r["ok"] for r in tool), len(tool)],
                        "length_stops": sum(r.get("finish") == "length" for r in gsm + tool),
                        "marker_leaks": sum(bool(r.get("leak")) for r in gsm + tool),
                        "errors": [{"kind": r["kind"], "id": r["id"]} for r in gsm + tool if not r["ok"]]})
    return {"format": "m31-bench-summary-v1", "reference": REFERENCE, "arms": arms, "quality": quality,
            "raw_speed_rows": rows,
            "limits": "Ratios use rounded report values. A single stochastic quality run does not establish a significant quality difference. No M2 pass/fail decision is inferred."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="write NEW JSON file instead of stdout")
    modes = parser.add_subparsers(dest="mode", required=True)
    prep = modes.add_parser("plan", help="print exact existing benchmark commands; do not execute")
    prep.add_argument("--arm", required=True)
    prep.add_argument("--port", type=int, default=8179)
    prep.add_argument("--directory", required=True)
    prep.add_argument("--home", default="/home/nekto")
    summary = modes.add_parser("summarize", help="read saved om_bench TSV and eval JSONL")
    summary.add_argument("--speed", type=Path, required=True, action="append")
    summary.add_argument("--quality", type=Path, default=[], action="append")
    args = parser.parse_args()
    try:
        result = plan(args) if args.mode == "plan" else summarize(args)
        data = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.out:
            with args.out.open("x", encoding="utf-8") as f:
                f.write(data)
        else:
            print(data, end="")
    except (OSError, ValueError, KeyError, IndexError, TypeError) as e:
        parser.exit(2, f"bench_m31: {e}\n")


if __name__ == "__main__":
    main()
