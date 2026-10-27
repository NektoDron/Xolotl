#!/usr/bin/env python3
"""Byte audit of frozen capture probe arms; runner owns ELF/ENV provenance."""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path


def compare_files(first, second, expected_bytes):
    if first.stat().st_size != expected_bytes or second.stat().st_size != expected_bytes:
        raise ValueError(f'wrong artifact length: {first} / {second}')
    hashes = [hashlib.sha256(), hashlib.sha256()]
    with first.open('rb') as a, second.open('rb') as b:
        while True:
            left, right = a.read(1 << 20), b.read(1 << 20)
            if left != right:
                raise ValueError(f'raw byte mismatch: {first} / {second}')
            if not left:
                break
            values = array.array('f')
            values.frombytes(left)
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f'nonfinite F32: {first}')
            hashes[0].update(left)
            hashes[1].update(right)
    return {'bytes': expected_bytes, 'sha256': hashes[0].hexdigest(), 'bytes_equal': True,
            'finite': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy', type=Path, required=True)
    parser.add_argument('--reuse', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    a, b = [json.loads((folder/'result.json').read_text()) for folder in (args.legacy, args.reuse)]
    if not a['pass'] or not b['pass'] or not b['reuse'] or (a['reuse'] and not (not a.get('view_mode',False) and b.get('view_mode',False))):
        raise ValueError('invalid probe arms')
    for key in ('feature_width', 'vocab', 'minimum', 'prompt_tokens'):
        if a[key] != b[key]:
            raise ValueError(f'arm geometry mismatch: {key}')
    if len(a['runs']) != len(b['runs']) or not a['runs']:
        raise ValueError('run count mismatch')
    records = []
    for index, (legacy, reuse) in enumerate(zip(a['runs'], b['runs'])):
        for key in ('position', 'saved_rows'):
            if legacy[key] != reuse[key]:
                raise ValueError(f'run {index} {key} differs')
        if legacy['position'] != a['prompt_tokens']:
            raise ValueError('incomplete prompt')
        geometry = lambda run: [(c['position'], c['n'], c['nk'], c['saved_rows']) for c in run['chunks']]
        if geometry(legacy) != geometry(reuse):
            raise ValueError(f'actual chunk/nk/saved row geometry differs at {index}')
        for run in (legacy, reuse):
            if sum(c['n'] for c in run['chunks']) != a['prompt_tokens']:
                raise ValueError('incomplete chunk trace')
            if sum(c['saved_rows'] for c in run['chunks']) != run['saved_rows']:
                raise ValueError('saved row sum mismatch')
        if legacy['saved_rows'] <= 0:
            raise ValueError('no compared features')
        record = {'run': index, 'chunks': geometry(legacy),
                  'features': compare_files(args.legacy/f'features-{index}.f32', args.reuse/f'features-{index}.f32', legacy['saved_rows']*a['feature_width']*4),
                  'head': compare_files(args.legacy/f'head-{index}.f32', args.reuse/f'head-{index}.f32', a['vocab']*4)}
        if b.get('view_mode',False):
            expected=legacy['saved_rows']*a['feature_width']*4
            record['whole_window']=compare_files(args.reuse/f'features-{index}.f32',args.reuse/f'window-{index}.f32',expected)
            record['lease_after_workspace_release']=compare_files(args.reuse/f'window-{index}.f32',args.reuse/f'window-after-release-{index}.f32',expected)
            lifetime=reuse.get('view_lifetime')
            if not lifetime or not all(lifetime.values()):raise ValueError('missing model lease lifecycle proof')
            record['view_lifetime']=lifetime
        if index:
            record['reuse_repeat_features'] = compare_files(args.reuse/'features-0.f32', args.reuse/f'features-{index}.f32', legacy['saved_rows']*a['feature_width']*4)
            record['reuse_repeat_head'] = compare_files(args.reuse/'head-0.f32', args.reuse/f'head-{index}.f32', a['vocab']*4)
        records.append(record)
    result = {'pass': True, 'scope': 'streamed raw feature/head byte comparison and finite audit; ELF/ENV binding belongs to immutable runner receipt',
              'legacy': str(args.legacy), 'reuse': str(args.reuse), 'runs': records}
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'pass': True, 'runs': len(records), 'out': str(args.out)}))


if __name__ == '__main__':
    main()
