#!/usr/bin/env python3
"""Run image/video single-prefill parity and text code128 under an owned card job.

This executable does not reserve a GPU; call only from vision_card_launch.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from profile_plan import load_profile

ROOT = Path(__file__).resolve().parents[2]
MODEL = "/mnt/ai/models/MiniMax-M3.1-Flash-GGUF/UD-IQ3_S/MiniMax-M3.1-Flash-UD-IQ3_S-00001-of-00005.gguf"


def input_receipt(manifest):
    files = {}
    for path in (manifest, *(manifest.parent / name for name in
                   ("text_ids.i32", "visual_rows.i32", "positions.i32", "visual.f32"))):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 << 20), b""):
                digest.update(block)
        files[path.name] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    return {"manifest": str(manifest.resolve()), "files": files}


def mixed_inputs(summary_path=None):
    if summary_path is None:
        return {label: Path("/mnt/ai/logs/m31-port/vision-encode-v1") / (label + "-batch-batch/manifest.json")
                for label in ("image", "video")}
    summary = json.loads(summary_path.read_text())
    if summary.get("single_batch_bitwise_pass") is not True:
        raise ValueError("GPU encoder summary must pass single/batch parity")
    results = summary.get("results")
    if not isinstance(results, list) or len(results) != 2:
        raise ValueError("GPU encoder summary must contain image and video exactly once")
    paths = {}
    for case in results:
        label = case.get("case")
        if label not in ("image", "video") or label in paths or case.get("single_vs_batch", {}).get("pass") is not True:
            raise ValueError("invalid/duplicate GPU encoder case")
        ipc = Path(case["single_ipc"]).resolve(strict=True)
        manifest = ipc.parent.parent / "batch" / "manifest.json"
        metadata = json.loads(manifest.read_text())
        if metadata.get("backend") != "cuda":
            raise ValueError("GPU encoder mixed artifact must declare CUDA backend")
        paths[label] = manifest.resolve(strict=True)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--vision-oracle", type=Path, default=ROOT / "build-vision-oracle/m31_vision_oracle")
    parser.add_argument("--text-oracle", type=Path, default=ROOT / "build-vision-oracle/text_oracle/m31_oracle")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--head-tail", type=int, default=256)
    parser.add_argument("--encoder-gpu-summary", type=Path)
    parser.add_argument("--skip-text", action="store_true")
    parser.add_argument("--ctx", type=int, default=4096)
    args = parser.parse_args()
    if args.ctx <= 0 or args.head_tail <= 0:
        parser.error("context and head-tail must be positive")
    if os.environ.get("M31_CARD_SESSION") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        parser.error("requires owned aictl card0 worker")
    args.output.mkdir(parents=True, exist_ok=False)
    profile = load_profile(ROOT / "tools/m31/profiles/native-gpu.json")
    native_environment = os.environ.copy()
    for key in profile["unset_environment"]:
        native_environment.pop(key, None)
    native_environment.update(profile["environment"])
    results = []
    inputs = mixed_inputs(args.encoder_gpu_summary)
    receipts = {}

    def run(label, command, environment=None):
        started = time.monotonic()
        with (args.output / (label + ".log")).open("x") as output:
            proc = subprocess.run([str(value) for value in command], env=environment, stdout=output, stderr=subprocess.STDOUT)
        results.append({"name": label, "argv": [str(value) for value in command], "returncode": proc.returncode,
                        "wall_s": time.monotonic() - started})
        (args.output / "commands.json").write_text(json.dumps(results, indent=2) + "\n")
        if proc.returncode:
            raise RuntimeError(f"{label} failed rc={proc.returncode}; see {args.output / (label + '.log')}")

    for label in ("image", "video"):
        mixed = inputs[label]
        receipts[label] = input_receipt(mixed)
        row_count = json.loads(mixed.read_text())["rows"]
        shared = ["--model", MODEL, "--mixed-manifest", mixed, "--chunk", row_count,
                  "--ctx", args.ctx, "--threads", "80", "--rows", f"last:{args.head_tail}",
                  "--head-tail", args.head_tail, "--taps", "none"]
        native_dir, oracle_dir = args.output / (label + "-native"), args.output / (label + "-oracle")
        run(label + "-native", [args.native, *profile["engine_arguments"], *shared, "--out", native_dir], native_environment)
        run(label + "-oracle", [args.vision_oracle, "--backend", "cuda", "--oracle-op-offload", "1", *shared, "--out", oracle_dir])
        run(label + "-compare", [sys.executable, ROOT / "tools/m31/vision_compare_logits.py", oracle_dir / "manifest",
                                native_dir / "manifest", mixed, "--output", args.output / (label + "-compare.json")])
        if input_receipt(mixed) != receipts[label]:
            raise RuntimeError("mixed input changed during native/oracle comparison")
    if not args.skip_text:
        ids = Path("/mnt/ai/logs/m31-port/ids-code128-v1/code128.i32")
        shared = ["--model", MODEL, "--ids-file", ids, "--backend", "cuda", "--ctx", args.ctx, "--chunk", "128",
                  "--threads", "80", "--rows", "all", "--taps", "none"]
        run("code128-native", [args.native, *profile["engine_arguments"], *shared, "--out", args.output / "code128-native"], native_environment)
        run("code128-oracle", [args.text_oracle, *shared, "--oracle-op-offload", "1", "--out", args.output / "code128-oracle"])
        run("code128-compare", [sys.executable, ROOT / "tools/m31/compare_logits.py", args.output / "code128-oracle", args.output / "code128-native",
                                "--top1", ".99", "--mean-kl", ".001", "--max-kl", ".05", "--out", args.output / "code128-compare.json"])
    labels = ("image", "video") if args.skip_text else ("image", "video", "code128")
    summary = {"format": "m31-vision-numeric-suite-v1", "pass": True, "mixed_head_tail": args.head_tail,
               "context": args.ctx, "text_skipped": args.skip_text,
               "encoder_gpu_summary": str(args.encoder_gpu_summary.resolve()) if args.encoder_gpu_summary else None,
               "inputs_immutable_during_run": receipts,
               "checks": {label: json.loads((args.output / (label + "-compare.json")).read_text()) for label in labels}}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"pass": True, "output": str(args.output)}))


if __name__ == "__main__":
    main()
