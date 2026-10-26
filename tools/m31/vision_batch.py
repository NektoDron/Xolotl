#!/usr/bin/env python3
"""Validate mtmd IPC and assemble one mixed text/vision LLM row stream.

This prepares inputs only: it does not execute Xolotl or llama.cpp inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np

PR_SHA = '6b95a9908d2c7b242817a447708aec38e937e2c0'


class Reader:
    def __init__(self, data):
        self.data, self.offset = data, 0

    def read(self, fmt):
        size = struct.calcsize('<' + fmt)
        if self.offset + size > len(self.data):
            raise ValueError('truncated mtmd metadata')
        values = struct.unpack_from('<' + fmt, self.data, self.offset)
        self.offset += size
        return values[0] if len(values) == 1 else values

    def string(self):
        size = self.read('Q')
        if size > len(self.data) - self.offset:
            raise ValueError('invalid metadata string length')
        data = self.data[self.offset:self.offset+size]
        self.offset += size
        return data.decode('utf-8')


def image_geometry(data):
    """Decode only audited PR serialization v3 (mtmd.cpp), never pixel payloads."""
    r = Reader(data)
    if r.read('Q') != 3 or r.read('I') != 1 or r.read('Q') != 0 or r.read('B') != 1:
        raise ValueError('expected v3 image chunk metadata without text')
    nx, ny, pos, index, merge = r.read('5I')
    media_id = r.string()
    if r.read('B'):
        raise ValueError('audio metadata in image chunk')
    count = r.read('Q')
    if count < 1 or count > len(data) // 34:
        raise ValueError('invalid image entry count')
    entries = []
    for _ in range(count):
        viewsep, newline = r.read('2B')
        lead, suffix, gx, gy, ox, oy, width, height = r.read('8i')
        if width <= 0 or height <= 0 or width % 28 or height % 28:
            raise ValueError('MiniMax resized image must have dimensions divisible by28')
        entries.append({'resized_width': width, 'resized_height': height,
                        'patch_grid': [width//14, height//14],
                        'merged_grid': [width//28, height//28]})
    if r.read('B') or r.offset != len(data) or pos != 0 or merge != 1:
        raise ValueError('unsupported mtmd image position/temporal layout')
    return {'decoder_grid': [nx, ny], 'frame_index': index, 'temporal_merge': merge,
            'id': media_id, 'entries': entries}


def load(path):
    path = Path(path)
    result = json.loads(path.read_text())
    if result.get('format') != 'm31-vision-ipc-v1' or result.get('source_pr') != PR_SHA:
        raise ValueError('unexpected IPC format/PR SHA')
    if result.get('n_embd') != 6144 or result.get('llm_inference') is not False:
        raise ValueError('unexpected embedding dimension or LLM inference claim')
    row = visual = 0
    for i, chunk in enumerate(result['chunks']):
        n = chunk['rows']
        if chunk['index'] != i or chunk['row'] != row or chunk['position'] != row or chunk['n_pos'] != n or n <= 0:
            raise ValueError('invalid row/position continuity')
        if chunk['type'] == 'text':
            if len(chunk['tokens']) != n or any(type(t) is not int or t < 0 for t in chunk['tokens']):
                raise ValueError('invalid text token span')
        elif chunk['type'] == 'image':
            if chunk['visual_row'] != visual:
                raise ValueError('visual rows reordered or overlapping')
            geometry = image_geometry((path.parent/chunk['metadata']).read_bytes())
            expected = sum(e['merged_grid'][0]*e['merged_grid'][1] for e in geometry['entries'])
            if expected != n or geometry['id'] != chunk['id']:
                raise ValueError('serialized image grid/ID disagrees with row count')
            chunk['geometry'] = geometry
            visual += n
        else:
            raise ValueError('unknown chunk type')
        row += n
    if result['rows'] != row or result['visual_rows'] != visual:
        raise ValueError('invalid manifest totals')
    return result


def embedding(path, chunk, dim):
    values = np.fromfile(Path(path).parent/chunk['embeddings'], dtype='<f4')
    if values.size != chunk['rows']*dim or not np.isfinite(values).all():
        raise ValueError('invalid embedding shape/values')
    return values.reshape(chunk['rows'], dim)


def assemble(path, out):
    m = load(path)
    out = Path(out)
    out.mkdir(exist_ok=False)
    tokens = np.full(m['rows'], -1, dtype='<i4')
    visual_index = np.full(m['rows'], -1, dtype='<i4')
    with (out/'visual.f32').open('wb') as f:
        for chunk in m['chunks']:
            start, n = chunk['row'], chunk['rows']
            if chunk['type'] == 'text':
                tokens[start:start+n] = chunk['tokens']
            else:
                visual_index[start:start+n] = np.arange(chunk['visual_row'], chunk['visual_row']+n, dtype='<i4')
                if m['encoded']:
                    embedding(path, chunk, m['n_embd']).tofile(f)
    tokens.tofile(out/'text_ids.i32')
    visual_index.tofile(out/'visual_rows.i32')
    np.arange(m['rows'], dtype='<i4').tofile(out/'positions.i32')
    m.update(format='m31-vision-batch-v1', source_manifest=str(Path(path).resolve()),
             row_contract='text_ids>=0 selects token embedding; text_ids=-1 selects visual_rows in visual.f32',
             llm_batches=1, native_engine_validated=False)
    (out/'manifest.json').write_text(json.dumps(m, ensure_ascii=False, indent=2)+'\n')
    return {'rows': m['rows'], 'visual_rows': m['visual_rows'], 'image_chunks': sum(c['type']=='image' for c in m['chunks']),
            'encoded': m['encoded'], 'native_engine_validated': False}


def compare(first, second, atol, rtol):
    a, b = load(first), load(second)
    if not a['encoded'] or not b['encoded'] or a['rows'] != b['rows'] or len(a['chunks']) != len(b['chunks']):
        raise ValueError('comparison needs encoded matching row streams')
    rows = []
    for x,y in zip(a['chunks'], b['chunks']):
        for key in ('type','rows','row','position'):
            if x[key] != y[key]:
                raise ValueError('chunk layouts differ')
        if x['type'] == 'text':
            if x['tokens'] != y['tokens']:
                raise ValueError('text token IDs differ')
            continue
        if x['id'] != y['id'] or x['geometry'] != y['geometry']:
            raise ValueError('image sources/preprocess grids differ')
        av,bv = embedding(first,x,a['n_embd']),embedding(second,y,b['n_embd'])
        delta=av.astype(np.float64)-bv
        rows.append({'chunk':x['index'],'rows':x['rows'], 'max_abs':float(np.max(np.abs(delta))),
                     'relative_l2':float(np.linalg.norm(delta)/max(np.linalg.norm(av.astype(np.float64)),1e-30)),
                     'bit_identical':av.tobytes()==bv.tobytes(), 'pass':bool(np.allclose(av,bv,atol=atol,rtol=rtol))})
    return {'pass':all(row['pass'] for row in rows),'atol':atol,'rtol':rtol,'chunks':rows}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    modes=p.add_subparsers(dest='mode',required=True)
    a=modes.add_parser('assemble');a.add_argument('manifest',type=Path);a.add_argument('out',type=Path)
    c=modes.add_parser('compare');c.add_argument('first',type=Path);c.add_argument('second',type=Path)
    c.add_argument('--atol',type=float,default=1e-6);c.add_argument('--rtol',type=float,default=1e-6)
    args=p.parse_args()
    result=assemble(args.manifest,args.out) if args.mode=='assemble' else compare(args.first,args.second,args.atol,args.rtol)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    if result.get('pass') is False:
        raise SystemExit(1)


if __name__=='__main__':
    main()
