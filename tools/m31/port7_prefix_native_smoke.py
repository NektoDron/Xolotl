#!/usr/bin/env python3
"""Native exact-prefix worker smoke; run only inside the authorized GPU0 session."""
import argparse
import array
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=Path(__file__).with_name("profiles") / "native-gpu.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("M31_CARD_SESSION") != "1":
        parser.error("aictl/flock session required")
    profile = json.loads(args.profile.read_text())
    for key in profile.get("unset_environment", []):
        os.environ.pop(key, None)
    os.environ.update(profile["environment"])
    spec = importlib.util.spec_from_file_location("m31_prefix_rpc", Path(__file__).with_name("serve_m31.py"))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    ids = array.array("i"); ids.frombytes(args.ids.read_bytes())
    if sys.byteorder != "little":
        ids.byteswap()
    ids = list(ids)
    if not ids:
        raise ValueError("empty input")
    argv = [str(args.engine.resolve()), "--model", str(args.model), *profile["engine_arguments"], "--prefix-cache", "1"]
    engine = module.RPC(argv, read_timeout=600.0, startup_timeout=900.0)
    checks = 0
    requests = []
    try:
        def check(ok, why):
            nonlocal checks
            if not ok:
                raise AssertionError(why)
            checks += 1
        def run(name, **options):
            request = {"tokens": ids, "max_tokens": 8, "temperature": 0, "seed": 42, "ignore_eos": True, **options}
            engine.write(request)
            events = []
            while True:
                event = engine.read(); events.append(event)
                if event["event"] == "error":
                    raise RuntimeError(event["error"])
                if event["event"] == "token" and request.get("interactive"):
                    engine.write({"continue": False})
                if event["event"] == "done":
                    break
            prefill = next(event for event in events if event["event"] == "prefill")
            result = {"name": name, "cache_prompt": request.get("cache_prompt", True), "events": events,
                      "tokens": [event["token"] for event in events if event["event"] == "token"]}
            requests.append(result)
            return result, prefill
        cold, cold_prefill = run("cold")
        reuse, reuse_prefill = run("repeat")
        false, false_prefill = run("cache-false", cache_prompt=False)
        check(not cold_prefill["cache_hit"] and cold_prefill["prompt_eval_n"] == len(ids), "cold request reused")
        check(reuse_prefill["cache_hit"] and reuse_prefill["cached_n"] == len(ids) and reuse_prefill["prompt_eval_n"] == 0, "same prompt missed")
        check(not false_prefill["cache_hit"] and false_prefill["prompt_eval_n"] == len(ids), "cache_prompt=false was not cold")
        check(cold["tokens"] == reuse["tokens"] == false["tokens"], "greedy tokens differ")
        check(cold_prefill["actual_chunks"] == reuse_prefill["actual_chunks"] == false_prefill["actual_chunks"], "geometry changed")
        after_false, after_false_prefill = run("after-cache-false")
        check(not after_false_prefill["cache_hit"], "cache-false retained metadata")
        cancel, cancel_prefill = run("cancel", interactive=True)
        check(cancel_prefill["cache_hit"], "cache failed to repopulate")
        check(len(cancel["tokens"]) == 1 and cancel["events"][-1]["fed_generated_n"] == 0, "cancel counted an unfed token")
        after_cancel, after_cancel_prefill = run("after-cancel")
        check(not after_cancel_prefill["cache_hit"], "cancel retained metadata")
        check(after_cancel["tokens"] == cold["tokens"], "cold after cancel differs")
        for request in requests:
            expected_fed = max(0, len(request["tokens"]) - 1)
            check(request["events"][-1]["kv_position"] == len(ids) + expected_fed, "reported KV position includes final sample")
        receipt = {"pass": True, "checks": checks, "scope": "actual GPU native worker exact prefix smoke; raw rows checked by separate probe",
                   "engine": str(args.engine), "engine_sha256": sha(args.engine), "profile": str(args.profile),
                   "profile_sha256": sha(args.profile), "ids_sha256": sha(args.ids), "prompt_tokens": len(ids),
                   "environment": {key: os.environ[key] for key in profile["environment"]},
                   "all_m31_environment": {key: value for key, value in sorted(os.environ.items()) if key.startswith(("M31_", "GGML_"))}, "argv": argv,
                   "ready": engine.ready, "requests": requests}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"pass": True, "checks": checks, "receipt": str(args.out)}))
    finally:
        engine.close()

if __name__ == "__main__":
    main()
