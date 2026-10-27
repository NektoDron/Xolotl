#!/usr/bin/env python3
"""Check mixed-input provenance/one-prefill geometry, then compare PR/native logits."""
import argparse
import hashlib
import json
from pathlib import Path

from compare_logits import compare_case, execution_chunks, load_case
from vision_protocol import load_mixed_rows


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("mixed_manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top1", type=float, default=.99)
    parser.add_argument("--mean-kl", type=float, default=1e-3)
    args = parser.parse_args()
    reference, _ = load_case(args.reference)
    candidate, _ = load_case(args.candidate)
    mixed = load_mixed_rows(args.mixed_manifest, vocab_size=reference["vocab"], max_context=reference["context"])
    if reference["engine"] != "llama.cpp-PR31192" or not reference.get("oracle_op_offload") or reference["backend"] != "cuda":
        raise ValueError("vision acceptance requires PR CUDA GPU-op-offload oracle")
    if candidate["engine"] != "xolotl-m31-correctness" or candidate["backend"] != "cuda":
        raise ValueError("vision acceptance requires native CUDA candidate")
    for directory, manifest in ((args.reference, reference), (args.candidate, candidate)):
        if execution_chunks(manifest) != [(0, mixed.rows)]:
            raise ValueError("vision acceptance requires one actual full mixed prefill on both sides")
        if (directory / "tokens.i32").read_bytes() != (mixed.manifest.parent / "text_ids.i32").read_bytes():
            raise ValueError("logits token/visual placeholder stream differs from supplied mixed input")
    result = compare_case(args.reference, args.candidate)
    result.update({"format": "m31-vision-logits-validation-v1", "scope": "one-complete-mixed-prefill-all-logits" if result.get("full_rows") else "one-complete-mixed-prefill-bounded-head",
                   "pass": result["top1"] >= args.top1 and result["mean_kl"] <= args.mean_kl,
                   "bit_identical": sha256(args.reference / "logits.f32") == sha256(args.candidate / "logits.f32"),
                   "thresholds": {"top1": args.top1, "mean_kl": args.mean_kl},
                   "input": {"manifest": str(mixed.manifest), "rows": mixed.rows, "visual_rows": mixed.visual_count,
                             "sha256": {name: sha256(mixed.manifest.parent / name) for name in
                                        ("manifest.json", "text_ids.i32", "visual_rows.i32", "positions.i32", "visual.f32")}},
                   "limit": "Both executables must consume this immutable mixed artifact; hashes authenticate the comparison input, not an independent encoder."})
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("pass", "top1", "mean_kl", "max_abs", "bit_identical", "rows")}))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
