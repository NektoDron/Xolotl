#!/usr/bin/env python3
"""Find the first diverging layer on a rerun with matching chunk sizes and --taps layers/all."""
import argparse
import json
from pathlib import Path
import re

import numpy as np

STAGES = ["attn_norm", "Qcur_normed", "Kcur_normed", "Qcur", "Kcur", "Vcur", "msa_bs", "msa_topk",
          "msa_block_mask", "kqv_out", "ffn_inp", "ffn_norm", "ffn_moe_topk", "ffn_moe_gate", "ffn_moe_up",
          "ffn_moe_down_scaled", "ffn_shexp", "ffn_moe_out", "ffn_out", "l_out"]
DTYPE = {0: "<f4", 1: "<f2", 26: "<i4"}


def index(root):
    result = {}
    for line in (root / "taps.jsonl").read_text().splitlines():
        item = json.loads(line)
        key = (item["pos0"], item["name"])
        if key in result:
            raise ValueError(f"duplicate tap {key}")
        result[key] = item
    return result


def compare(reference, candidate, all_common=False, atol=1e-5, rtol=1e-4):
    for root in (reference, candidate):
        if not (root / "manifest.json").exists():
            raise ValueError(f"incomplete run {root}")
    if (reference / "tokens.i32").read_bytes() != (candidate / "tokens.i32").read_bytes():
        raise ValueError("token IDs differ")
    rm = json.loads((reference / "manifest.json").read_text())
    cm = json.loads((candidate / "manifest.json").read_text())
    if rm["chunk"] != cm["chunk"] or rm["model"] != cm["model"]:
        raise ValueError("tap localization needs identical chunks and model")
    a, b = index(reference), index(candidate)
    keys = a.keys() & b.keys()
    keys = [k for k in keys if re.fullmatch(r".+-\d+", k[1]) and
            (all_common or k[1].startswith("l_out-"))]
    if not keys:
        raise ValueError("no common layer taps")
    if not all_common:
        wanted = {k for k in a if k[1].startswith("l_out-")}
        if wanted != {k for k in b if k[1].startswith("l_out-")}:
            raise ValueError("layer output tap sets differ")

    def order(k):
        base, layer = k[1].rsplit("-", 1)
        return (k[0], int(layer), STAGES.index(base) if base in STAGES else 100, base)

    rows = []
    for key in sorted(keys, key=order):
        x, y = a[key], b[key]
        item = dict(pos0=key[0], name=key[1])
        if x["shape"] != y["shape"] or x["type"] != y["type"]:
            rows.append({**item, "pass": False, "reason": "shape/type differs"})
            continue
        size = int(np.prod(x["shape"]))
        arrays = []
        for root, t in ((reference, x), (candidate, y)):
            path = (root / t["file"]).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("tap path escapes run directory")
            dtype = np.dtype(DTYPE[t["type"]])
            if path.stat().st_size != size * dtype.itemsize:
                raise ValueError(f"truncated/extra tap data: {path}")
            arrays.append(np.memmap(path, dtype=dtype, mode="r").astype(np.float64))
        av, bv = arrays
        finite = np.isfinite(av) & np.isfinite(bv)
        matching_inf = np.isinf(av) & (av == bv)
        if not np.all(finite | matching_inf):
            rows.append({**item, "pass": False, "reason": "non-finite mismatch"})
            continue
        av, bv = av[finite], bv[finite]
        diff = av - bv
        max_abs = float(np.abs(diff).max(initial=0))
        rms = float(np.sqrt(np.mean(diff * diff))) if len(diff) else 0.
        ref_rms = float(np.sqrt(np.mean(av * av))) if len(av) else 0.
        ok = bool(np.all(diff == 0)) if x["type"] == 26 else bool(np.allclose(av, bv, atol=atol, rtol=rtol))
        rows.append({**item, "pass": ok, "max_abs": max_abs, "rms": rms,
                     "relative_rms": rms / max(ref_rms, 1e-30)})
    failed = [r for r in rows if not r["pass"]]
    return dict(first_divergence=failed[0] if failed else None, compared=len(rows),
                reference_only=len(a.keys() - b.keys()), candidate_only=len(b.keys() - a.keys()), rows=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--all-common", action="store_true")
    a = p.parse_args()
    try:
        result = compare(a.reference, a.candidate, a.all_common)
    except (ValueError, KeyError, OSError) as e:
        p.exit(2, f"comparison refused: {e}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return int(result["first_divergence"] is not None)


if __name__ == "__main__":
    raise SystemExit(main())
