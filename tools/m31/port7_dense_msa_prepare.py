#!/usr/bin/env python3
"""Freeze real normalized hidden rows for the isolated MSA operator probe."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--capture', type=Path, default=Path('/mnt/ai/logs/m31-port/code128-native-v2/code128'))
    a = p.parse_args()
    a.out.mkdir()
    rows = [json.loads(x) for x in (a.capture / 'taps.jsonl').read_text().splitlines()]
    manifest = {'scope': 'Real captured attn_norm F32 rows. Index query/key projections, normalization and RoPE are reconstructed with n1 stock GPU ops. Long immutable key fixtures cycle the 128 derived F16 key rows; they are operator stress fixtures, not long-context model captures.',
                'capture_manifest_sha256': sha(a.capture / 'taps.jsonl'), 'layers': [], 'files': {}}
    for layer in (0, 3, 18, 32, 59):
        row = next(x for x in rows if x['name'] == f'attn_norm-{layer}' and x['pos0'] == 0)
        assert row['type'] == 0 and row['shape'] == [6144, 128, 1, 1]
        source = a.capture / row['file']
        name = f'layer{layer}-attn_norm.f32'
        assert source.stat().st_size == 6144 * 128 * 4
        shutil.copy2(source, a.out / name)
        manifest['layers'].append({'layer': layer, 'attn_norm': name, 'captured_tokens': 128})
        manifest['files'][name] = {'source': str(source), 'sha256': sha(source), 'bytes': source.stat().st_size}
    (a.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
