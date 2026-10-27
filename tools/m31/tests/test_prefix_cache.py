#!/usr/bin/env python3
"""CPU tiny-model protocol coverage for the opt-in exact prompt cache."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
from make_fixture import make_fixture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    args = parser.parse_args()
    checks = 0
    def check(ok, why):
        nonlocal checks
        if not ok:
            raise AssertionError(why)
        checks += 1
    with tempfile.TemporaryDirectory(prefix="m31-prefix-test-") as tmp:
        model = Path(tmp) / "tiny.gguf"
        make_fixture(model)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"}
        argv = [str(args.engine), "--model", str(model), "--backend", "cpu", "--ctx", "32", "--chunk", "16", "--threads", "2"]
        for opt_in in (False, True):
            with subprocess.Popen(argv + (["--prefix-cache", "1"] if opt_in else []), env=env,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
                ready = proc.stdout.readline()
                if not ready:
                    raise RuntimeError("worker startup failed: " + proc.stderr.read())
                check(json.loads(ready)["event"] == "ready", "ready")
                def run(ids, **kwargs):
                    request = {"tokens": ids, "max_tokens": 4, "temperature": 0, "seed": 42, **kwargs}
                    proc.stdin.write(json.dumps(request) + "\n"); proc.stdin.flush()
                    events = []
                    while True:
                        line = proc.stdout.readline()
                        if not line:
                            raise RuntimeError("worker exited")
                        event = json.loads(line); events.append(event)
                        if event["event"] in ("done", "error"):
                            return events
                        if event["event"] == "token" and request.get("interactive"):
                            proc.stdin.write('{"continue":false}\n'); proc.stdin.flush()
                def prefill(events):
                    return next(e for e in events if e["event"] == "prefill")
                def tokens(events):
                    return [e["token"] for e in events if e["event"] == "token"]
                ids = [1, 2, 3, 4, 5, 6, 7, 8]
                cold, repeat = run(ids), run(ids)
                check(tokens(cold) == tokens(repeat), "repeat changed greedy output")
                if not opt_in:
                    check("cache_hit" not in prefill(repeat), "default protocol changed")
                else:
                    check(prefill(cold)["cached_n"] == 0, "first request reused")
                    check(prefill(repeat)["cache_hit"] and prefill(repeat)["prompt_eval_n"] == 0, "exact repeat miss")
                    check(repeat[-1]["kv_position"] == len(ids) + 3 and repeat[-1]["fed_generated_n"] == 3, "KV counted an unfed final token")
                    no_cache = run(ids, cache_prompt=False)
                    check(tokens(no_cache) == tokens(cold) and not prefill(no_cache)["cache_hit"], "cache_prompt=false")
                    check(not prefill(run(ids))["cache_hit"], "cache_prompt=false retained metadata")
                    check(prefill(run(ids))["cache_hit"], "cache failed to repopulate")
                    changed = [9] + ids[1:]
                    check(not prefill(run(changed))["cache_hit"], "different tokens reused")
                    check(not prefill(run(ids, max_tokens=0))["cache_hit"], "zero-max different prompt reuse")
                    check(not prefill(run(ids))["cache_hit"], "zero-max request retained stale head")
                    run(ids, interactive=True)
                    check(not prefill(run(ids))["cache_hit"], "cancellation retained cache")
                    check(run([-1])[-1]["event"] == "error", "invalid ID accepted")
                    check(not prefill(run(ids))["cache_hit"], "error retained cache")
                    multi = ids * 3
                    run(multi)
                    check(not prefill(run(multi))["cache_hit"], "unproved multi-chunk reuse")
                    check(not prefill(run(ids))["cache_hit"], "partial prefix reused")
                proc.stdin.close()
                check(proc.wait(timeout=20) == 0, "worker exit")
    print(json.dumps({"pass": True, "checks": checks, "scope": "CPU tiny-model exact-cache protocol, not production GPU quality"}))

if __name__ == "__main__":
    main()
