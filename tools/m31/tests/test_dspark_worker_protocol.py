#!/usr/bin/env python3
"""Real worker+session control flow against CPU Engine/Draft mocks; no CUDA library."""
import argparse
import json
import os
from pathlib import Path
import subprocess

class Worker:
    def __init__(self, path, enabled, mode="all", backend="cuda", pool=True, threshold=32, fail_once=False):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "M31_CARD_SESSION": "1", "DS_FIXTURE_DRAFT_MODE": mode}
        if fail_once: env["DS_FIXTURE_DRAFT_FAIL_ONCE"] = "1"
        argv = [str(path), "--model", "fixture.gguf", "--backend", backend, "--ctx", "128", "--chunk", "16",
                "--persistent-cpu-pool", str(int(pool)), "--gpu-experts-min-tokens", str(threshold)]
        if enabled:
            argv += ["--dspark-model", "fixture-draft.gguf", "--dspark-depth", "7", "--dspark-p-min", "0"]
        self.proc = subprocess.Popen(argv, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True)
        ready = self.proc.stdout.readline()
        if not ready:
            raise RuntimeError(self.proc.stderr.read())
        self.ready = json.loads(ready)
        self.last_inits = 0
    def run(self, body, cancel_after=None, allow_error=False):
        request = {"tokens": [1,2,3,4,5,6,7,8], "max_tokens": 17, "temperature": 0, "seed": 123,
                   "ignore_eos": True, **body}
        if cancel_after is not None:
            request["interactive"] = True
        self.proc.stdin.write(json.dumps(request) + "\n"); self.proc.stdin.flush()
        events = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(self.proc.stderr.read())
            event = json.loads(line); events.append(event)
            if event["event"] == "sample":
                self.proc.stdin.write('{"accepted":true}\n'); self.proc.stdin.flush()
            if event["event"] == "token" and request.get("interactive"):
                keep = cancel_after is None or event["index"] < cancel_after
                self.proc.stdin.write(json.dumps({"continue": keep}) + "\n"); self.proc.stdin.flush()
            if event["event"] in ("done", "error"):
                break
        if events[-1]["event"] == "error" and not allow_error:
            raise RuntimeError(events[-1]["error"])
        new_inits = events[-1]["fixture"]["draft_inits"]
        delta = new_inits - self.last_inits; self.last_inits = new_inits
        return request, events, delta
    def close(self):
        self.proc.stdin.close()
        if self.proc.wait(timeout=20):
            raise RuntimeError(self.proc.stderr.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    args = parser.parse_args()
    checks = 0; cases = 0
    def check(ok, why):
        nonlocal checks
        if not ok:
            raise AssertionError(why)
        checks += 1
    requests = [(dict(max_tokens=n), None) for n in (0,1,2,8,9,17,40)]
    requests += [({"suppress_ids": [2,3,4,8]}, None),
                 ({"tokens": [1], "stop_ids": [3], "ignore_eos": False}, None),
                 ({"tokens": [1], "stop_ids": [3], "ignore_eos": True}, None),
                 ({"max_tokens": 17, "cache_prompt": False}, None),
                 ({"max_tokens": 8, "temperature": 1, "top_k": 3, "top_p": 1, "min_p": 0}, None),
                 ({"max_tokens": 8, "grammar_check": True}, None)]
    requests += [(dict(max_tokens=17), stop) for stop in (0,1,4,7,8,15)]
    for mode in ("all", "middle", "none"):
        baseline, fast = Worker(args.worker, False), Worker(args.worker, True, mode=mode)
        try:
            check("dspark" not in baseline.ready, "default ready changed")
            check(fast.ready["dspark"]["mode"] == "greedy_only" and fast.ready["dspark"]["sampling_fallback"], "ready overclaims sampling")
            for body, cancel_after in requests:
                request, plain, _ = baseline.run(body, cancel_after)
                _, result, inits = fast.run(body, cancel_after)
                cold, done = plain[-1], result[-1]
                tokens = lambda values: [e["token"] for e in values if e["event"] == "token"]
                check(tokens(plain) == tokens(result), f"greedy/fallback tokens differ mode={mode} body={body}")
                check(cold["timings"]["predicted_n"] == done["timings"]["predicted_n"] and cold["finish_reason"] == done["finish_reason"], "sample count/finish differs")
                expected = len(request["tokens"]) + max(0, done["timings"]["predicted_n"] - 1)
                check(done["kv_position"] == expected and done["fixture"]["position"] == expected, "unemitted speculative tail retained")
                check(done["fed_generated_n"] == max(0, done["timings"]["predicted_n"] - 1), "last emitted sample counted as fed")
                check(done["timings"]["sampling_calls"] == done["timings"]["predicted_n"], "sampler was called ahead")
                eligible = request["max_tokens"] > 1 and request["temperature"] == 0 and not request.get("grammar_check", False)
                check(done["timings"]["dspark_mode"] == ("greedy" if eligible else "target_fallback"), "mode mismatch")
                check(done["fixture"]["capture_enabled"] == eligible, "fallback retained feature capture")
                check(done["fixture"]["external_bytes"] == 0, "draft resident budget leaked")
                check(inits == int(eligible and done["timings"]["predicted_n"] > 1), "draft load was eager or repeated")
                if not eligible:
                    check(done["fixture"]["verify_calls"] == 0 and done["fixture"]["feature_reads"] == 0, "fallback invoked speculative compute")
                    check(bool(done["timings"]["dspark_fallback_reason"]), "fallback reason missing")
                cases += 1
        finally:
            baseline.close(); fast.close()
    for backend, pool, threshold, reason in (("cpu",True,32,"backend"), ("cuda",False,32,"persistent_cpu_pool_required"), ("cuda",True,1,"expert_threshold")):
        worker = Worker(args.worker, True, backend=backend, pool=pool, threshold=threshold)
        try:
            _, events, inits = worker.run({"max_tokens": 8})
            done = events[-1]
            check(done["timings"]["dspark_fallback_reason"] == reason, "unsupported reason mismatch")
            check(inits == 0 and done["fixture"]["verify_calls"] == 0 and not done["fixture"]["capture_enabled"], "unsupported config attempted M3")
            cases += 1
        finally: worker.close()
    failure = Worker(args.worker, True, fail_once=True)
    try:
        _, events, _ = failure.run({"max_tokens": 8}, allow_error=True)
        check(events[-1]["event"] == "error" and "injected" in events[-1]["error"], "draft failure not reported")
        check(events[-1]["fixture"]["resident_drafts"] == 0, "partial draft allocation leaked")
        _, next_events, loads = failure.run({"max_tokens": 8})
        check(next_events[-1]["event"] == "done" and loads == 1, "next request after draft error failed")
        cases += 2
    finally: failure.close()
    print(json.dumps({"pass": True, "cases": cases, "checks": checks,
                      "scope": "actual worker + actual session helper with deterministic CPU mocks; no model/GPU numerical acceptance"}))

if __name__ == "__main__": main()
