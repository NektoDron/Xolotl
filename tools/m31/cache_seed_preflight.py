#!/usr/bin/env python3
"""Verify small rank files and current GGUF headers/stats without scanning weights.

This inherits previously verified model content SHA256 values. Header/stat checks
are not a new full-content verification and cannot detect payload-only changes
that preserve file sizes and timestamps.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat as stat_module
import struct

MAX_JSON_BYTES = 32 << 20


def read_bounded(path, maximum, *, exact=None, label='input'):
    """Check before open/read; a concurrent growth can consume at most one extra byte."""
    path = Path(path)
    def check_size(info):
        if not stat_module.S_ISREG(info.st_mode):
            raise ValueError(f'{label} must be a regular file')
        if info.st_size > maximum or (exact is not None and info.st_size != exact):
            raise ValueError(f'{label} byte size {info.st_size} exceeds bound or differs from expected size')
    before = path.stat()
    check_size(before)
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        check_size(opened)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns):
            raise ValueError(f'{label} changed before bounded read')
        data = stream.read(opened.st_size + 1)
        after = os.fstat(stream.fileno())
        if len(data) != opened.st_size or (after.st_size, after.st_mtime_ns) != (opened.st_size, opened.st_mtime_ns):
            raise ValueError(f'{label} changed during bounded read')
    return data


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def parse_json(data):
    return json.loads(data.decode('utf-8'), object_pairs_hook=unique_object)


def read_json(path):
    return parse_json(read_bounded(path, MAX_JSON_BYTES, label='JSON input'))


def positive(value, name, maximum=16777216, zero=False):
    if type(value) is not int or not (0 if zero else 1) <= value <= maximum:
        raise ValueError(f'invalid {name}')
    return value


def hex_sha(value):
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError('invalid SHA256')
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError('invalid SHA256') from exc
    if len(raw) != 32 or value != value.lower():
        raise ValueError('invalid canonical SHA256')
    return raw


class Header:
    """Bounded little-endian GGUFv3 header cursor; skip tokenizer array payloads."""
    def __init__(self, file, size):
        self.file, self.size = file, size

    def read(self, n):
        if n < 0 or n > self.size - self.file.tell():
            raise ValueError('truncated GGUF header')
        data = self.file.read(n)
        if len(data) != n:
            raise ValueError('truncated GGUF header')
        return data

    def number(self, fmt):
        return struct.unpack('<' + fmt, self.read(struct.calcsize('<' + fmt)))[0]

    def skip(self, n):
        if n < 0 or n > self.size - self.file.tell():
            raise ValueError('truncated GGUF header')
        self.file.seek(n, 1)

    def string(self, capture=True):
        length = self.number('Q')
        if length > 32 << 20:
            raise ValueError('unbounded GGUF string')
        if capture:
            return self.read(length).decode('utf-8')
        self.skip(length)

    def value(self, kind, capture=False, depth=0):
        sizes = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
                 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
        if kind in sizes:
            value = self.number(sizes[kind])
            return value if capture else None
        if kind == 8:
            return self.string(capture)
        if kind != 9 or depth > 1:
            raise ValueError('unsupported GGUF metadata kind')
        item_kind, count = self.number('I'), self.number('Q')
        if count > 16777216:
            raise ValueError('unbounded GGUF array')
        if item_kind in sizes:
            self.skip(count * struct.calcsize('<' + sizes[item_kind]))
        else:
            for _ in range(count):
                self.value(item_kind, False, depth + 1)
        return None


def tensor_header(path):
    path = Path(path)
    stat = path.stat()
    metadata, tensors = {}, []
    with path.open('rb') as f:
        h = Header(f, stat.st_size)
        if h.read(4) != b'GGUF' or h.number('I') != 3:
            raise ValueError('expected little-endian GGUFv3')
        nt, nm = h.number('Q'), h.number('Q')
        if nt > 1000000 or nm > 1000000:
            raise ValueError('unbounded GGUF directory')
        keys = set()
        for _ in range(nm):
            key, kind = h.string(), h.number('I')
            if key in keys:
                raise ValueError('duplicate GGUF metadata key')
            keys.add(key)
            capture = key.startswith('minimax-m3.') or key in ('general.architecture', 'general.alignment')
            value = h.value(kind, capture)
            if capture:
                metadata[key] = value
        names = set()
        for _ in range(nt):
            name, dims = h.string(), h.number('I')
            if name in names or not 1 <= dims <= 4:
                raise ValueError('duplicate tensor or invalid dimensions')
            names.add(name)
            shape = [positive(h.number('Q'), 'tensor dimension', maximum=1 << 40) for _ in range(dims)]
            kind, offset = h.number('I'), h.number('Q')
            tensors.append({'name': name, 'shape': shape, 'type_id': kind, 'offset': offset,
                            'shard': path.name})
        alignment = metadata.get('general.alignment', 32)
        positive(alignment, 'GGUF alignment', maximum=1 << 20)
        data_start = (f.tell() + alignment - 1) // alignment * alignment
        if nt == 0 and data_start > stat.st_size and f.tell() == stat.st_size:
            data_start = stat.st_size  # Metadata-only split shard need not have tensor padding.
        if data_start > 32 << 20 or data_start > stat.st_size:
            raise ValueError('unbounded/truncated GGUF header region')
        f.seek(0)
        header_sha = sha(h.read(data_start))
    return metadata, tensors, data_start, header_sha


def decode_rank(data, identity, layers, experts, leading):
    count = (layers - leading) * experts
    if len(data) != 56 + count * 4:
        raise ValueError('binary rank truncated or has trailing bytes')
    magic, version, got_layers, got_experts, got_count = struct.unpack('<8sIIII', data[:24])
    if (magic, version, got_layers, got_experts, got_count) != (b'M31RANK\0', 1, layers, experts, count):
        raise ValueError('binary rank header mismatch')
    if data[24:56] != hex_sha(identity):
        raise ValueError('binary rank identity mismatch')
    ids = list(struct.unpack('<' + 'I' * count, data[56:]))
    if set(ids) != set(range(leading * experts, layers * experts)):
        raise ValueError('binary rank is not a complete unique routed-ID permutation')
    return ids


def seed_budget(slots, experts, layers, leading, used, requested=None):
    for name, value in [('slots', slots), ('experts', experts), ('layers', layers), ('used', used)]:
        positive(value, name)
    if type(leading) is not int or not 0 <= leading < layers or used > experts or slots < experts or slots > (layers - leading) * experts:
        raise ValueError('seed cache must reserve one complete layer working set')
    maximum = slots - experts
    recommended = min(maximum, max(0, slots - (layers - leading) * used))
    hot = recommended if requested is None else positive(requested, 'hot count', zero=True)
    if hot > maximum:
        raise ValueError('hot count leaves fewer than experts stream slots')
    return {'slots': slots, 'max_hot': maximum, 'recommended_hot': recommended,
            'requested_hot': hot, 'stream_slots': slots - hot,
            'full_decode_working_set': (layers - leading) * used,
            'recommendation_scope': 'short calibration replay; production optimum not established'}


def verify(rank_path, binary_path, tensors_path, integrity_path, slots, requested_hot=None):
    rank_path, binary_path = Path(rank_path), Path(binary_path)
    rank_data = read_bounded(rank_path, MAX_JSON_BYTES, label='rank JSON')
    rank = parse_json(rank_data)
    if rank.get('format') != 'm31-expert-ranking-v1' or rank.get('ranking_only') is not True:
        raise ValueError('unsupported ranking format')
    identity = rank['identity']
    if sha(json.dumps(identity, sort_keys=True).encode()) != rank['identity_sha256']:
        raise ValueError('JSON identity checksum mismatch')
    g = identity['geometry']
    layers, experts = positive(g['layers'], 'layers', 1024), positive(g['experts'], 'experts', 4096)
    leading = positive(g['dense_lead'], 'dense_lead', layers - 1, zero=True)
    used = positive(g['used'], 'used', experts)
    expected_rank_bytes = 56 + (layers - leading) * experts * 4
    data = read_bounded(binary_path, expected_rank_bytes, exact=expected_rank_bytes, label='binary rank')
    ids = decode_rank(data, rank['identity_sha256'], layers, experts, leading)
    records = rank['records']
    if len(records) != len(ids):
        raise ValueError('JSON ranking count mismatch')
    for row in records:
        positive(row['layer'], 'rank layer', layers - 1, zero=True)
        positive(row['expert'], 'rank expert', experts - 1, zero=True)
        positive(row['count'], 'route count', maximum=1 << 63, zero=True)
        if type(row['weight_sum']) not in (int, float) or not math.isfinite(row['weight_sum']) or row['weight_sum'] < 0:
            raise ValueError('invalid ranking weight')
    if ids != [row['layer'] * experts + row['expert'] for row in records]:
        raise ValueError('binary rank order differs from JSON')
    if records != sorted(records, key=lambda row: (-row['count'], -row['weight_sum'], row['layer'], row['expert'])):
        raise ValueError('JSON rank order inconsistent with scores')
    if sum(row['count'] for row in records) != rank['training_route_count']:
        raise ValueError('training route count mismatch')
    tensors_path = Path(tensors_path)
    tensors_data = read_bounded(tensors_path, MAX_JSON_BYTES, label='tensor JSON')
    if sha(tensors_data) != identity['tensors_sha256']:
        raise ValueError('reference tensor directory checksum mismatch')
    expected = parse_json(tensors_data)
    expected_by_shard = {}
    expected_names = set()
    for row in expected:
        if row['name'] in expected_names:
            raise ValueError('duplicate reference tensor')
        expected_names.add(row['name'])
        key = Path(row['shard']).name
        expected_by_shard.setdefault(key, {})[row['name']] = row
    integrity = read_json(integrity_path)
    content_shas = {Path(row['file']).name: (row['actual'], row['revision']) for row in integrity
                    if row['match'] is True and row['actual'] == row['expected']}
    if len(content_shas) != len(integrity):
        raise ValueError('reference integrity report contains failed/duplicate rows')
    checked, actual_tensors, metadata = [], [], {}
    shard_names = set()
    for shard in identity['shards']:
        path = Path(shard['path'])
        if str(path.resolve()) != shard['path'] or path.name in shard_names:
            raise ValueError('noncanonical/duplicate model shard path')
        shard_names.add(path.name)
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (shard['bytes'], shard['mtime_ns']):
            raise ValueError('current model size/mtime differs from ranking identity')
        if content_shas.get(path.name) != (shard['reference_sha256'], shard['revision']):
            raise ValueError('model inherited content integrity mismatch')
        hex_sha(shard['reference_sha256'])
        meta, tensors, data_start, header_sha = tensor_header(path)
        if data_start == stat.st_size and header_sha != shard['reference_sha256']:
            raise ValueError('metadata-only shard content SHA256 differs from reference')
        metadata.update(meta)
        ref = expected_by_shard.get(path.name, {})
        if {t['name'] for t in tensors} != set(ref):
            raise ValueError('current tensor directory names differ')
        for tensor in tensors:
            row = ref[tensor['name']]
            if any(tensor[k] != row[k] for k in ('name', 'shape', 'type_id', 'offset')):
                raise ValueError('current tensor descriptor differs')
            if tensor['offset'] + row['bytes'] > stat.st_size - data_start:
                raise ValueError('tensor payload exceeds current shard bounds')
            actual_tensors.append(tensor)
        checked.append(dict(shard, header_sha256=header_sha, data_start=data_start))
    if not set(expected_by_shard) <= shard_names or shard_names != set(content_shas):
        raise ValueError('model shard sets differ')
    if Path(g['model']).resolve() != Path(identity['shards'][0]['path']):
        raise ValueError('model entry shard differs from ranking identity')
    if metadata.get('general.architecture') != 'minimax-m3':
        raise ValueError('current model architecture differs')
    keys = {'layers': 'block_count', 'embd': 'embedding_length', 'experts': 'expert_count',
            'used': 'expert_used_count', 'dense_lead': 'leading_dense_block_count',
            'expert_ff': 'expert_feed_forward_length', 'heads': 'attention.head_count',
            'kv_heads': 'attention.head_count_kv', 'head_dim': 'attention.key_length',
            'index_dim': 'attention.indexer.key_length', 'index_heads': 'attention.indexer.head_count',
            'block': 'attention.indexer.block_size', 'top_blocks': 'attention.indexer.top_k'}
    for name, key in keys.items():
        if metadata.get('minimax-m3.' + key) != g[name]:
            raise ValueError(f'current model geometry differs: {name}')
    tensor_digest = sha(json.dumps(sorted(actual_tensors, key=lambda t: t['name']), sort_keys=True).encode())
    return {'format': 'm31-seed-preflight-v1', 'verified': True,
            'rank_json': str(rank_path.resolve()), 'rank_json_sha256': sha(rank_data),
            'rank_binary': str(binary_path.resolve()), 'rank_binary_sha256': sha(data),
            'identity_sha256': rank['identity_sha256'], 'layers': layers, 'experts': experts,
            'dense_lead': leading, 'count': len(ids), 'budget': seed_budget(slots, experts, layers, leading, used, requested_hot),
            'tensor_header_sha256': tensor_digest, 'checked_shards': checked,
            'model_content_verification': 'inherited reference SHA256; current headers/sizes/mtime checked only',
            'limitation': 'Payload-only edits preserving header/size/mtime require a fresh full-content hash to detect.',
            'integration': 'Not installed. Caller must pass expected identity and binary SHA256 from this trusted receipt to the C++ reader.'}


def seal_receipt(receipt):
    """Portable small binary receipt for dependency-free engine validation."""
    if receipt.get('format') != 'm31-seed-preflight-v1' or receipt.get('verified') is not True:
        raise ValueError('only a verified preflight receipt can be sealed')
    def string(value):
        raw = value.encode('utf-8')
        if not raw or len(raw) > 16384 or b'\0' in raw:
            raise ValueError('invalid seal path')
        return struct.pack('<I', len(raw)) + raw
    shards = receipt['checked_shards']
    data = struct.pack('<8sIIIIQQI', b'M31SEED\0', 1, receipt['layers'], receipt['experts'],
                       receipt['dense_lead'], receipt['budget']['slots'], receipt['budget']['requested_hot'], len(shards))
    data += hex_sha(receipt['identity_sha256']) + hex_sha(receipt['rank_binary_sha256']) + string(receipt['rank_binary'])
    for shard in shards:
        data += string(shard['path']) + struct.pack('<QQQ', shard['bytes'], shard['mtime_ns'], shard['data_start'])
        data += hex_sha(shard['header_sha256'])
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('rank', 'binary', 'tensors', 'integrity', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--slots', type=int, required=True)
    p.add_argument('--hot', type=int)
    p.add_argument('--seal', type=Path, help='also write a NEW compact receipt for engine init')
    a = p.parse_args()
    try:
        receipt = verify(a.rank, a.binary, a.tensors, a.integrity, a.slots, a.hot)
        with a.output.open('x') as f:
            json.dump(receipt, f, indent=2, allow_nan=False)
            f.write('\n')
        if a.seal:
            sealed = seal_receipt(receipt)
            with a.seal.open('xb') as f:
                f.write(sealed)
            print(json.dumps({'receipt_seal': str(a.seal.resolve()), 'receipt_seal_sha256': sha(sealed)}))
        print(json.dumps({'verified': True, 'budget': receipt['budget'],
                          'tensor_header_sha256': receipt['tensor_header_sha256'],
                          'model_content_verification': receipt['model_content_verification']}, indent=2))
    except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
        p.exit(2, f'cache_seed_preflight: {exc}\n')


if __name__ == '__main__':
    main()
