#!/usr/bin/env python3
"""Freeze real captured model activations for stock row-batching components."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--captures', type=Path, default=Path('/mnt/ai/logs/m31-port/code128-native-v2/code128'))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(exist_ok=False)
    taps = {v['name']: v for v in map(json.loads, (a.captures/'taps.jsonl').read_text().splitlines())}
    files = {}
    rows = []
    first, count = 17, 4
    for layer in (0, 3, 18, 32, 59):
        fields = {'attn_norm':f'attn_norm-{layer}', 'ffn_input':f'ffn_inp-{layer}', 'ffn_norm':f'ffn_norm-{layer}'}
        if layer:
            fields['attn_hidden'] = f'l_out-{layer-1}'
        if layer >= 3:
            fields.update(gate=f'ffn_moe_gate-{layer}', up=f'ffn_moe_up-{layer}')
        row = {'layer':layer, 'first_source_token':first, 'tokens':count}
        for key, name in fields.items():
            tap = taps[name]
            if tap['type'] != 0:
                raise ValueError('component requires captured F32 inputs')
            shape = tap['shape']
            width = shape[0]*(shape[1] if key in ('gate','up') else 1)
            source = a.captures/tap['file']
            with source.open('rb') as f:
                f.seek(first*width*4)
                data = f.read(count*width*4)
            if len(data) != count*width*4:
                raise ValueError('short activation capture')
            target = f'layer{layer}-{key}.f32'
            (a.out/target).write_bytes(data)
            row[key] = target
            files[target] = {'source':str(source), 'source_shape':shape, 'offset_bytes':first*width*4,
                             'bytes':len(data), 'sha256':hashlib.sha256(data).hexdigest()}
        rows.append(row)
    result = {'scope':'Real captured F32 model activations; numerical component replay, not a new model oracle. Raw Q/K, index Q/K and dense GU/router are derived with n1 GPU projections of captured normalized hidden. Stress absolute RoPE positions are intentionally independent of original capture context.',
              'captures':str(a.captures), 'capture_manifest_sha256':hashlib.sha256((a.captures/'manifest.json').read_bytes()).hexdigest(),
              'layers':rows, 'files':files, 'position_starts':[first,254,32766,65534]}
    (a.out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'layers':len(rows),'files':len(files),'bytes':sum(x['bytes'] for x in files.values())},indent=2))


if __name__=='__main__':
    main()
