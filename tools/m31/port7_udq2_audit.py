#!/usr/bin/env python3
"""Read GGUF headers only, including incomplete/sparse local download files.

No numpy/GPU/model load and no writes to model files. Tensor payload untouched.
"""
import ast
import collections
import datetime
import hashlib
import json
import math
import pathlib
import re
import struct

ROOT = pathlib.Path('/mnt/ai/Xolotl-m31')
MODEL = pathlib.Path('/mnt/ai/models/MiniMax-M3.1-Flash-GGUF/UD-Q2_K_XL')
CONSTANTS = ROOT / 'build-m31-cpu/_deps/strata_llamacpp-src/gguf-py/gguf/constants.py'


def quant_constants():
    tree = ast.parse(CONSTANTS.read_text())
    names, sizes = {}, {}
    def integer(x):
        if isinstance(x, ast.Constant): return x.value
        if isinstance(x, ast.Name) and x.id == 'QK_K': return 256
        if isinstance(x, ast.BinOp):
            a, b = integer(x.left), integer(x.right)
            if isinstance(x.op, ast.Add): return a + b
            if isinstance(x.op, ast.FloorDiv): return a // b
            if isinstance(x.op, ast.Mult): return a * b
        raise ValueError(ast.dump(x))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'GGMLQuantizationType':
            for entry in node.body:
                if isinstance(entry, ast.Assign): names[entry.targets[0].id] = integer(entry.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == 'GGML_QUANT_SIZES':
            for key, value in zip(node.value.keys, node.value.values):
                sizes[names[key.attr]] = (key.attr, *(integer(v) for v in value.elts))
    return sizes


class Header:
    formats = {0:'B', 1:'b', 2:'H', 3:'h', 4:'I', 5:'i', 6:'f', 7:'?', 10:'Q', 11:'q', 12:'d'}
    def __init__(self, f): self.f = f
    def n(self, fmt):
        size = struct.calcsize('<' + fmt)
        data = self.f.read(size)
        if len(data) != size: raise EOFError('incomplete GGUF header')
        return struct.unpack('<' + fmt, data)[0]
    def string(self, keep=True):
        n = self.n('Q')
        if n > 64 * 1024 * 1024: raise ValueError('invalid GGUF string length')
        if keep:
            data = self.f.read(n)
            if len(data) != n: raise EOFError('incomplete GGUF string')
            return data.decode('utf-8')
        self.f.seek(n, 1)
    def value(self, kind, keep=True):
        if kind in self.formats: return self.n(self.formats[kind])
        if kind == 8: return self.string(keep)
        if kind == 9:
            subtype, count = self.n('I'), self.n('Q')
            retain = keep and count <= 256
            values = []
            if not retain and subtype in self.formats:
                self.f.seek(count * struct.calcsize('<' + self.formats[subtype]), 1)
            else:
                for _ in range(count):
                    v = self.value(subtype, retain)
                    if retain: values.append(v)
            return values if retain else {'array_type':subtype, 'count':count, 'values_omitted':True}
        raise ValueError(f'unsupported GGUF metadata type{kind}')


def read_header(path, quants):
    before = path.stat()
    with path.open('rb') as f:
        h = Header(f)
        if f.read(4) != b'GGUF': raise ValueError('GGUF magic')
        version, count, kv_count = h.n('I'), h.n('Q'), h.n('Q')
        if version != 3: raise ValueError('GGUFv3 required')
        metadata = {}
        for _ in range(kv_count):
            key = h.string()
            metadata[key] = h.value(h.n('I'))
        tensors = []
        for _ in range(count):
            name, dims = h.string(), h.n('I')
            shape = [h.n('Q') for _ in range(dims)]
            kind, offset = h.n('I'), h.n('Q')
            typename, block, size = quants[kind]
            if shape[0] % block: raise ValueError('nondivisible quant row')
            size_bytes = math.prod(shape) // block * size
            match = re.match(r'blk\.(\d+)\.(.*)', name)
            tensors.append({'name':name, 'shape':shape, 'type_id':kind, 'type':typename,
                            'block_elements':block, 'block_bytes':size, 'bytes':size_bytes,
                            'offset':offset, 'layer':int(match[1]) if match else None,
                            'role':match[2] if match else name,
                            'routed_expert':bool(re.search(r'ffn_(gate|up|down)_exps\.weight$', name))})
        header_bytes = f.tell()
        alignment = metadata.get('general.alignment', 32)
        data_start = (header_bytes + alignment - 1) // alignment * alignment
        expected_extent = data_start + max(t['offset'] + t['bytes'] for t in tensors) if tensors else header_bytes
        f.seek(0)
        header_sha = hashlib.sha256(f.read(header_bytes)).hexdigest()
    after = path.stat()
    groups = collections.defaultdict(lambda:{'tensors':0, 'bytes':0, 'layers':set(), 'roles':set()})
    for t in tensors:
        key = ('routed' if t['routed_expert'] else 'other') + '/' + t['type']
        g = groups[key]; g['tensors'] += 1; g['bytes'] += t['bytes']
        if t['layer'] is not None: g['layers'].add(t['layer'])
        g['roles'].add(t['role'])
    for g in groups.values(): g['layers'] = sorted(g['layers']); g['roles'] = sorted(g['roles'])
    return {'path':str(path), 'file_size_before':before.st_size, 'file_size_after':after.st_size,
            'allocated_bytes':after.st_blocks * 512, 'aria2_sidecar_present':path.with_suffix(path.suffix+'.aria2').exists(),
            'version':version, 'tensor_count':count, 'metadata':metadata, 'header_bytes':header_bytes,
            'header_sha256':header_sha, 'data_start':data_start, 'expected_extent_from_header':expected_extent,
            'logical_size_covers_expected_extent':after.st_size >= expected_extent,
            'payload_verified':False, 'groups':dict(groups), 'tensors':tensors}


if __name__ == '__main__':
    quants = quant_constants()
    result = {'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope':'local read-only GGUFv3 headers; no tensor payload read or availability/quality proof',
              'model_directory':str(MODEL), 'quant_constants_path':str(CONSTANTS),
              'quant_constants_sha256':hashlib.sha256(CONSTANTS.read_bytes()).hexdigest(),
              'shards':[read_header(p, quants) for p in sorted(MODEL.glob('*.gguf'))]}
    out = pathlib.Path('/mnt/ai/logs/m31-port/port7-ud-q2-header-audit-v1.json')
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for shard in result['shards']:
        brief = {k:v for k,v in shard['metadata'].items() if k.startswith('split.') or k in ['general.architecture','general.file_type']}
        print(pathlib.Path(shard['path']).name, shard['tensor_count'], brief)
        print(json.dumps(shard['groups'], ensure_ascii=False, indent=2))
    print(out)
