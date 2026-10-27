#!/usr/bin/env python3
"""CPU native-worker regression for merged text and vision prompt caches.

Runs the compiled worker against make_fixture's tiny F32 model. The production
manifest reader requires 6144-wide media; one visual row is padded to that width,
and the tiny engine consumes its leading D=32 values. This deliberately tests
worker/cache protocol integration, not production vision geometry or quality.
The worker exposes sampled IDs but not raw logits, so continuation comparisons
below cover deterministic greedy IDs, not full-logit equality.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile

from make_fixture import D, make_fixture


SOURCE_PR = "6b95a9908d2c7b242817a447708aec38e937e2c0"
TEXT_IDS = [5, 7, 11, 17]
MIXED_IDS = [5, 7, 11, -1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_mixed(directory, sign):
    directory.mkdir()
    # Change values actually consumed by the tiny engine, not only padding.
    values = [sign * (0.15 + (i % 7) * 0.07) for i in range(D)]
    values += [0.0] * (6144 - D)
    (directory / "text_ids.i32").write_bytes(struct.pack("<4i", *MIXED_IDS))
    (directory / "visual_rows.i32").write_bytes(struct.pack("<4i", -1, -1, -1, 0))
    (directory / "positions.i32").write_bytes(struct.pack("<4i", 0, 1, 2, 3))
    (directory / "visual.f32").write_bytes(struct.pack("<6144f", *values))
    manifest = {
        "format": "m31-vision-batch-v1", "source_pr": SOURCE_PR,
        "encoded": True, "llm_inference": False, "n_embd": 6144,
        "rows": 4, "visual_rows": 1,
        "chunks": [
            {"index": 0, "type": "text", "row": 0, "position": 0,
             "rows": 3, "n_pos": 3, "tokens": TEXT_IDS[:3]},
            {"index": 1, "type": "image", "row": 3, "position": 3,
             "rows": 1, "n_pos": 1, "visual_row": 0,
             "decoder_positions": {"start": 3, "step": 1, "axes": 1}},
        ],
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def native_requests(worker, model, requests, *, cache=True, dspark=None):
    # Never inherit a card session or a production tuning/diagnostic environment.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("M31_", "GGML_"))}
    env.update(CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2",
               M31_PROMPT_CACHE_MIB="16" if cache else "0", M31_SHORT_PREFILL_MODE="off")
    argv = [str(worker), "--model", str(model), "--backend", "cpu", "--ctx", "32",
            "--chunk", "16", "--threads", "2", "--persistent-cpu-pool", "1",
            "--prefix-cache", "1" if cache else "0"]
    if dspark is not None:
        argv += ["--dspark-model", str(dspark)]
    completed = subprocess.run(argv, input="".join(json.dumps(req) + "\n" for req in requests),
                               text=True, capture_output=True, env=env, timeout=120)
    if completed.returncode:
        raise AssertionError(f"worker exited {completed.returncode}: {completed.stderr}")
    events = [json.loads(line) for line in completed.stdout.splitlines()]
    assert events and events.pop(0)["event"] == "ready", completed.stdout
    groups, group = [], []
    for event in events:
        assert event["event"] != "error", event
        group.append(event)
        if event["event"] == "done":
            groups.append(group)
            group = []
    assert not group and len(groups) == len(requests), events
    return groups


def prefill(group):
    return next(event for event in group if event["event"] == "prefill")


def tokens(group):
    return [event["token"] for event in group if event["event"] == "token"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    worker = args.worker.resolve()
    checks = 0

    def check(condition, explanation):
        nonlocal checks
        if not condition:
            raise AssertionError(explanation)
        checks += 1

    with tempfile.TemporaryDirectory(prefix="m31-vision-cache-merge-") as temporary:
        directory = Path(temporary)
        model = directory / "tiny.gguf"
        make_fixture(model)
        media = {"A": make_mixed(directory / "media-A", 1),
                 "B": make_mixed(directory / "media-B", -1)}
        common = {"max_tokens": 1, "temperature": 0, "top_p": 1,
                  "top_k": 0, "min_p": 0, "seed": 42, "ignore_eos": True,
                  "interactive": False, "ttft_trace": True}

        def request(kind, **options):
            result = {**common, "tokens": TEXT_IDS, **options}
            if kind in media:
                result["mixed_manifest"] = str(media[kind])
            return result

        # Adjacent A/B have identical token IDs, positions and row geometry.
        # Repeat A later verifies a genuine F32-keyed snapshot hit still works.
        cases = [
            ("text-cold", "text", {}), ("text-live-hit", "text", {}),
            ("mixed-A-cold", "A", {}), ("mixed-B-cold", "B", {}),
            ("text-after-mixed", "text", {}), ("text-live-again", "text", {}),
            ("mixed-A-snapshot-hit", "A", {}),
            ("mixed-B-snapshot-hit", "B", {}),
            ("mixed-A-explicit-false", "A", {"cache_prompt": False}),
            ("mixed-A-after-false", "A", {}),
            ("text-explicit-false", "text", {"cache_prompt": False}),
            ("text-after-false", "text", {}),
        ]
        groups = native_requests(worker, model, [request(kind, **opts) for _, kind, opts in cases])
        fresh_kinds = ["text", "A", "B"]
        controls = native_requests(worker, model,
                                   [request(kind, cache_prompt=False) for kind in fresh_kinds], cache=False)
        reference = dict(zip(fresh_kinds, map(tokens, controls)))
        records = {}
        for (name, kind, options), group in zip(cases, groups):
            event = prefill(group)
            check(tokens(group) == reference[kind], f"{name}: greedy IDs differ from fresh control")
            check(len(tokens(group)) == 1, f"{name}: expected one greedy token")
            check(group[-1]["kv_position"] == 4 and group[-1]["fed_generated_n"] == 0,
                  f"{name}: max_tokens=1 fed an un-emitted token")
            check(bool(event["actual_chunks"]) is (not event["cache_hit"]),
                  f"{name}: trace counted cached geometry as actual compute")
            check(event["prompt_eval_n"] == (0 if event["cache_hit"] else 4),
                  f"{name}: incorrect evaluated prompt length")
            if kind in media:
                # A mixed hit must come from the F32-hashed snapshot cache.
                check(event["cache_hit"] == event["prompt_cache"]["hit"],
                      f"{name}: mixed prompt reused live IDs-only KV")
            if options.get("cache_prompt") is False:
                check(not event["cache_hit"] and not event["prompt_cache"]["enabled"],
                      f"{name}: explicit false did not bypass both caches")
                check("input_sha256" not in event["prompt_cache"],
                      f"{name}: explicit false performed a snapshot lookup")
            records[name] = {"kind": kind, "prefill": event, "tokens": tokens(group)}

        def event(name):
            return records[name]["prefill"]

        check(not event("text-cold")["cache_hit"], "first text request unexpectedly hit")
        for name in ("text-live-hit", "text-live-again"):
            check(event(name)["cache_hit"] and not event(name)["prompt_cache"]["hit"],
                  f"{name}: live text cache API was lost")
        for name in ("mixed-A-cold", "mixed-B-cold"):
            check(not event(name)["cache_hit"], f"{name}: different media hit IDs-only cache")
            check(event(name)["prompt_cache"].get("stored") is True,
                  f"{name}: snapshot was not stored")
        key_a = event("mixed-A-cold")["prompt_cache"]["input_sha256"]
        key_b = event("mixed-B-cold")["prompt_cache"]["input_sha256"]
        check(key_a != key_b, "different F32 embeddings have the same snapshot key")
        for name, key in (("mixed-A-snapshot-hit", key_a), ("mixed-B-snapshot-hit", key_b),
                          ("mixed-A-after-false", key_a)):
            snapshot = event(name)["prompt_cache"]
            check(snapshot["hit"] and snapshot["input_sha256"] == key,
                  f"{name}: F32-keyed snapshot reuse failed")
        check(event("text-after-mixed")["prompt_cache"]["hit"],
              "mixed processing damaged the independent text snapshot")
        check(event("text-after-false")["prompt_cache"]["hit"],
              "explicit optout invalidated an independent snapshot")

        # CPU's unsupported-backend fallback must remain lazy even for media.
        # This does not exercise CUDA-eligible mixed_prompt DSpark fallback.
        fallback = native_requests(worker, model, [request("A"), request("B"), request("text")],
                                   dspark=directory / "MUST-NOT-OPEN.gguf")
        for kind, group in zip(("A", "B", "text"), fallback):
            timings = group[-1]["timings"]
            check(tokens(group) == reference[kind], f"{kind}: DSpark CPU fallback changed greedy IDs")
            check(timings["dspark_mode"] == "target_fallback" and
                  timings["dspark_fallback_reason"] == "backend", "DSpark CPU policy changed")
            check(not prefill(group)["cache_hit"], "draft-configured mixed fallback reused a cache")
        receipt = {
            "pass": True, "checks": checks, "requests": len(cases) + len(controls) + len(fallback),
            "worker": str(worker), "worker_sha256": sha256(worker),
            "test_sha256": sha256(__file__), "fixture_sha256": sha256(model),
            "media_f32_sha256": {kind: sha256(path.parent / "visual.f32") for kind, path in media.items()},
            "snapshot_keys": {"A": key_a, "B": key_b},
            "fresh_greedy_ids": reference, "cases": records,
            "scope": "Real compiled worker + CPU tiny F32 fixture; live text KV versus media-hashed snapshots, "
                     "same-ID/different-F32 media, explicit optout, fresh greedy IDs, lazy DSpark CPU fallback. "
                     "One 6144-wide manifest row padded around the tiny model's first 32 consumed values; "
                     "no full-logit, production-geometry, GPU, speed or CUDA DSpark assertion.",
        }
    if args.report:
        args.report.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "cases"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
