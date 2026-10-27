"""CPU preflight for M31 mixed rows; no engine, encoder, or HTTP integration.

This checks the existing vision_batch.py artifact's row contract. It does not
authenticate the model/mmproj, re-run mtmd geometry, or claim inference parity.
The caller must keep the request-owned files immutable until consumption ends.
"""
from __future__ import annotations

from array import array
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

PR_SHA = "6b95a9908d2c7b242817a447708aec38e937e2c0"
EMBEDDING_WIDTH = 6144


@dataclass(frozen=True)
class MixedRows:
    manifest: Path
    token_ids: tuple[int, ...]
    visual_rows: tuple[int, ...]
    positions: tuple[int, ...]
    visual_path: Path
    visual_count: int

    @property
    def rows(self):
        return len(self.token_ids)

    @property
    def embedding_width(self):
        return EMBEDDING_WIDTH


def _integer(value, label, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label} must be an integer in {low}..{high}")
    return value


def _file(directory, name, size):
    path = (directory / name).resolve(strict=True)
    if path.parent != directory or not path.is_file() or path.stat().st_size != size:
        raise ValueError(f"{name}: expected a local file of exactly {size} bytes")
    return path


def _i32(path):
    values = array("i")
    if values.itemsize != 4:
        raise RuntimeError("this Python array implementation lacks 32-bit int")
    values.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        values.byteswap()
    return tuple(values)


def load_mixed_rows(manifest, *, vocab_size, max_context, check_finite=True):
    """Validate prepared text/visual row mapping, without running inference.

    The existing manifest has no model/mmproj identity receipt. Shape validation
    here must not be used as a substitute for that future engine-side check.
    """
    vocab_size = _integer(vocab_size, "vocab_size", 1, 2**31 - 1)
    max_context = _integer(max_context, "max_context", 1, 2**31 - 1)
    path = Path(manifest).resolve(strict=True)
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("mixed manifest exceeds 16 MiB")
    m = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(m, dict) or m.get("format") != "m31-vision-batch-v1":
        raise ValueError("expected m31-vision-batch-v1 manifest")
    if m.get("source_pr") != PR_SHA or m.get("n_embd") != EMBEDDING_WIDTH:
        raise ValueError("unsupported source PR or embedding width")
    if m.get("encoded") is not True or m.get("llm_inference") is not False:
        raise ValueError("mixed rows require encoded input without an inference claim")
    rows = _integer(m.get("rows"), "rows", 1, max_context)
    visual_count = _integer(m.get("visual_rows"), "visual_rows", 0, rows)
    directory = path.parent
    ids = _i32(_file(directory, "text_ids.i32", rows * 4))
    mapping = _i32(_file(directory, "visual_rows.i32", rows * 4))
    positions = _i32(_file(directory, "positions.i32", rows * 4))
    visual_path = _file(directory, "visual.f32", visual_count * EMBEDDING_WIDTH * 4)
    if positions != tuple(range(rows)):
        raise ValueError("M31 decoder positions must be contiguous linear rows")
    next_visual = 0
    for token, visual in zip(ids, mapping):
        if token == -1:
            if visual != next_visual or visual >= visual_count:
                raise ValueError("visual rows must be complete, ordered, and unique")
            next_visual += 1
        elif not 0 <= token < vocab_size or visual != -1:
            raise ValueError("text row needs an in-vocabulary ID and visual index -1")
    if next_visual != visual_count:
        raise ValueError("visual row count does not match mapping")

    chunks = m.get("chunks")
    if not isinstance(chunks, list) or not 1 <= len(chunks) <= rows:
        raise ValueError("invalid chunk list")
    first = first_visual = 0
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError("invalid chunk object")
        count = _integer(chunk.get("rows"), "chunk rows", 1, rows - first)
        for label in ("index", "row", "position", "n_pos"):
            _integer(chunk.get(label), f"chunk {label}", 0, rows)
        if (chunk.get("index") != index or chunk.get("row") != first or
                chunk.get("position") != first or chunk.get("n_pos") != count):
            raise ValueError("chunk row/position discontinuity")
        if chunk.get("type") == "text":
            tokens = chunk.get("tokens")
            if (not isinstance(tokens, list) or len(tokens) != count or
                    any(type(t) is not int or not 0 <= t < vocab_size for t in tokens) or
                    tuple(tokens) != ids[first:first + count]):
                raise ValueError("text chunk disagrees with binary token IDs")
        elif chunk.get("type") == "image":
            _integer(chunk.get("visual_row"), "chunk visual_row", 0, visual_count)
            decoder = chunk.get("decoder_positions")
            if not isinstance(decoder, dict) or any(type(v) is not int for v in decoder.values()):
                raise ValueError("invalid linear decoder position descriptor")
            if (chunk.get("visual_row") != first_visual or
                    decoder != {"start": first, "step": 1, "axes": 1} or
                    ids[first:first + count] != (-1,) * count):
                raise ValueError("image chunk disagrees with linear visual mapping")
            first_visual += count
        else:
            raise ValueError("only text/image mtmd chunks are supported")
        first += count
    if first != rows or first_visual != visual_count:
        raise ValueError("chunks do not cover all mixed rows")
    if check_finite:
        with visual_path.open("rb") as stream:
            while block := stream.read(256 * 1024):
                values = array("f")
                if values.itemsize != 4:
                    raise RuntimeError("this Python array implementation lacks float32")
                values.frombytes(block)
                if sys.byteorder != "little":
                    values.byteswap()
                if not all(math.isfinite(x) for x in values):
                    raise ValueError("visual embeddings contain non-finite values")
    return MixedRows(path, ids, mapping, positions, visual_path, visual_count)
