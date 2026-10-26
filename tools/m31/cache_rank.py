#!/usr/bin/env python3
"""Rank unchanged M31 routes and replay held-out traffic; no GPU/model execution."""
import argparse
from collections import Counter, OrderedDict, defaultdict
import hashlib
import json
import math
from pathlib import Path
import struct

GIB = 1 << 30
BLOCK_BYTES = {18: 98, 23: 136, 40: 36}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_trace(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if not rows or rows[0].get('format') != 'm31-routing-trace-v1':
        raise ValueError('missing routing header')
    header = rows[0]
    cases = {}
    grouped = defaultdict(dict)
    for row in rows[1:]:
        if row['kind'] == 'case':
            if row['case'] in cases:
                raise ValueError('duplicate case')
            cases[row['case']] = row
        elif row['kind'] == 'routes':
            name, layer, n, pos = row['case'], row['layer'], row['n'], row['position']
            if name not in cases or not header['dense_lead'] <= layer < header['layers']:
                raise ValueError('invalid case/layer')
            if n <= 0 or pos < 0 or pos + n > cases[name]['tokens']:
                raise ValueError('invalid observed span')
            if len(row['ids']) != n * header['used'] or len(row['weights']) != len(row['ids']):
                raise ValueError('route dimensions differ')
            for token in range(n):
                a, b = token * header['used'], (token + 1) * header['used']
                ids, weights = row['ids'][a:b], row['weights'][a:b]
                if len(set(ids)) != len(ids) or any(not 0 <= x < header['experts'] for x in ids):
                    raise ValueError('invalid/duplicate top-k ID')
                if any(not math.isfinite(x) or x < 0 for x in weights):
                    raise ValueError('invalid route weight')
                key = (pos + token, layer)
                if key in grouped[name]:
                    raise ValueError('duplicate observed token/layer')
                grouped[name][key] = (ids, weights)
    for name, case in cases.items():
        expected = case['tokens'] * (header['layers'] - header['dense_lead'])
        if len(grouped[name]) != expected:
            raise ValueError(f'incomplete route trace: {name}')
    return header, cases, grouped


def simulate(requests, capacity, hot=()):
    hot = set(hot)
    if len(hot) >= capacity:
        raise ValueError('hot seeds leave no stream slots')
    stream = OrderedDict()
    stats = Counter()
    for keys in requests:
        protected = set(keys)
        if len(protected - hot) > capacity - len(hot):
            raise ValueError('stream cache cannot fit current batch')
        for key in keys:
            stats['requests'] += 1
            if key in hot:
                stats['hits'] += 1
                stats['hot_hits'] += 1
            elif key in stream:
                stats['hits'] += 1
                stream.move_to_end(key)
            else:
                stats['misses'] += 1
                if len(stream) == capacity - len(hot):
                    victim = next((k for k in stream if k not in protected), None)
                    if victim is None:
                        raise ValueError('all stream slots protected')
                    del stream[victim]
                    stats['evictions'] += 1
                stream[key] = None
    counts = {key: stats[key] for key in ('requests', 'hits', 'hot_hits', 'misses', 'evictions')}
    return dict(counts, capacity=capacity, hot_seeds=len(hot),
                seed_upload_slots=len(hot), hit_rate=stats['hits'] / max(1, stats['requests']))


def requests_for(header, grouped, names, mode, chunk):
    for name in names:
        tokens = 1 + max(p for p, _ in grouped[name])
        if mode == 'decode':
            for position in range(tokens):
                for layer in range(header['dense_lead'], header['layers']):
                    yield [(layer, e) for e in grouped[name][position, layer][0]]
        else:
            for position in range(0, tokens, chunk):
                for layer in range(header['dense_lead'], header['layers']):
                    ids = {e for p in range(position, min(tokens, position + chunk))
                           for e in grouped[name][p, layer][0]}
                    yield [(layer, e) for e in sorted(ids)]


def rank(args):
    header, cases, grouped = load_trace(args.trace)
    provenance = json.loads(Path(args.inputs).read_text())
    input_by_name = {x['case']: x for x in provenance['inputs']}
    if set(cases) != set(input_by_name):
        raise ValueError('trace/input manifest case sets differ')
    for name, case in cases.items():
        item = input_by_name[name]
        if digest(case['input']) != item['sha256'] or case['tokens'] != item['tokens']:
            raise ValueError('calibration input fingerprint mismatch')
    train = sorted(n for n in cases if input_by_name[n]['split'] == 'train')
    held = sorted(n for n in cases if input_by_name[n]['split'] == 'held')
    if not train or not held:
        raise ValueError('separate train/held cases are required')
    counts, weights = Counter(), Counter()
    for name in train:
        for (_, layer), (ids, route_weights) in grouped[name].items():
            for e, weight in zip(ids, route_weights):
                counts[layer, e] += 1
                weights[layer, e] += weight
    order = sorted(((l, e) for l in range(header['dense_lead'], header['layers'])
                    for e in range(header['experts'])),
                   key=lambda k: (-counts[k], -weights[k], k))
    records = [{'layer': l, 'expert': e, 'count': counts[l, e],
                'weight_sum': weights[l, e]} for l, e in order]
    simulations = []
    for capacity in args.slots:
        for mode in ('decode', 'prefill'):
            requests = list(requests_for(header, grouped, held, mode, args.chunk))
            reserve = header['used'] if mode == 'decode' else header['experts']
            hot_count = max(0, capacity - reserve)
            token_working_set = (header['layers'] - header['dense_lead']) * header['used']
            whole_token_hot_count = max(0, capacity - token_working_set) if mode == 'decode' else hot_count
            lru = simulate(requests, capacity)
            seeded = simulate(requests, capacity, order[:hot_count])
            sweeps = [simulate(requests, capacity, order[:min(hot_count, count)])
                      for count in sorted(set([0, capacity // 8, capacity // 4, capacity // 2,
                                               whole_token_hot_count, hot_count]))]
            simulations.append({'mode': mode, 'lru': lru, 'hot_plus_stream': seeded,
                                'whole_token_hot_plus_stream': simulate(requests, capacity, order[:whole_token_hot_count]),
                                'seed_count_sweep': sweeps})
    integrity_path = Path(args.integrity)
    integrity = json.loads(integrity_path.read_text())
    if not integrity or any(not row['match'] or row['actual'] != row['expected'] for row in integrity):
        raise ValueError('reference GGUF integrity not verified')
    model = Path(header['model'])
    shards = []
    for row in integrity:
        path = model.parent / Path(row['file']).name
        stat = path.stat()
        shards.append({'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                       'reference_sha256': row['actual'], 'revision': row['revision']})
    fingerprint = {'geometry': header, 'tensors_sha256': digest(args.tensors), 'shards': shards}
    result = {'format': 'm31-expert-ranking-v1', 'ranking_only': True,
              'identity': fingerprint,
              'identity_sha256': hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest(),
              'trace': {'path': str(args.trace), 'sha256': digest(args.trace)},
              'inputs': provenance['inputs'], 'train_cases': train, 'held_cases': held,
              'training_route_count': sum(counts.values()), 'records': records,
              'simulations': simulations,
              'limits': ['No router changes, pruning or requantization.',
                         'Decode replay reorders batch-captured routes token-major; actual decode routing can differ.',
                         '8 short prompts do not establish production hit rate; long-context and real decode remain required.',
                         'Model content SHA values are inherited from the reference integrity report, not recomputed here.',
                         'Runtime seed/cache integration is not implemented by this offline report.']}
    output = Path(args.output)
    with output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    binary = output.with_suffix('.bin')
    # Header magic/version + complete routed-layer uint32 permutation. Dense IDs omitted.
    with binary.open('xb') as f:
        f.write(struct.pack('<8sIIII', b'M31RANK\0', 1, header['layers'], header['experts'], len(order)))
        f.write(bytes.fromhex(result['identity_sha256']))
        f.write(struct.pack('<' + 'I' * len(order), *(l * header['experts'] + e for l, e in order)))
    print(json.dumps({'records': len(order), 'train_tokens': sum(cases[n]['tokens'] for n in train),
                      'held_tokens': sum(cases[n]['tokens'] for n in held), 'simulations': simulations}, indent=2))


def budget(args):
    # Header/tensor-directory payload lengths only; no large weight read or GPU.
    tensors = json.loads(Path(args.tensors).read_text())
    by_name = {t['name']: t for t in tensors}
    align = lambda n, a: (n + a - 1) // a * a
    multiple, maximum = args.alignment, 0
    layers = []
    for layer in range(3, 60):
        span = raw = 0
        for role in ('gate', 'up', 'down'):
            tensor = by_name[f'blk.{layer}.ffn_{role}_exps.weight']
            multiple = math.lcm(multiple, BLOCK_BYTES[tensor['type_id']])
            size = tensor['bytes'] // 128
            raw += size
            span = align(span, args.alignment) + align(size, args.alignment)
        maximum = max(maximum, span)
        layers.append({'layer': layer, 'raw_expert_bytes': raw, 'aligned_roles_span': span})
    stride = align(maximum, multiple)
    gpu_weights = sum(t['bytes'] for t in tensors if '_exps.' not in t['name'] and t['name'] != 'token_embd.weight')
    rows = []
    for context in (4096, 32768, 65536):
        n = min(context, args.chunk)
        score = context * 4 * n * 4
        shared = context * n * 4 + context * 4 + n * 4 + context // 128 * n * 4
        # A reservation estimate, not a proven allocator peak: 2 FP32 scores plus
        # one F16 attention mask and F32 converted index cache. Alias reuse varies.
        msa_envelope = 2 * score + score // 2 + context * 128 * 4 + shared
        for kv in ('f16-native', 'q8-main-f16-index-hypothetical', 'q8-reference-main-and-index-kv'):
            if kv == 'f16-native':
                kv_bytes = context * 60 * (2 * 4 * 128 + 128) * 2
            elif kv.startswith('q8-main'):
                kv_bytes = context * 60 * (2 * 4 * 128 * 34 // 32 + 128 * 2)
            else:
                kv_bytes = context * 60 * (2 * 4 * 128 + 2 * 128) * 34 // 32
            remaining = args.vram_mib * (1 << 20) - gpu_weights - kv_bytes - msa_envelope - int(args.extra_scratch_gib * GIB)
            slots = max(0, remaining // stride)
            rows.append({'context': context, 'kv': kv, 'chunk': n, 'gpu_weight_bytes': gpu_weights,
                         'kv_bytes': kv_bytes, 'msa_reservation_envelope': msa_envelope,
                         'extra_scratch_reserve': int(args.extra_scratch_gib * GIB),
                         'expert_budget_bytes': max(0, remaining), 'fixed_slots': slots,
                         'complete_128expert_layers': slots // 128})
    result = {'format': 'm31-vram-budget-v1', 'vram_mib': args.vram_mib, 'slot_stride': stride,
              'stride_multiple': multiple, 'largest_aligned_role_span': maximum, 'layers': layers, 'rows': rows,
              'limits': ['Slot stride matches current fixed-slot code for this UD-IQ3_S directory and alignment.',
                         'MSA reservation is an explicit envelope, not a measured gallocr peak.',
                         'Other graph scratch, CUDA context and transfer workspace use configurable extra reserve.',
                         'Native q8 main cache is not implemented; its row is a planning scenario.',
                         'Reference indexer allocates both K and dummy V; native keeps only F16 index K.',
                         'Ref at64Kq8/ubatch2048 measured3 resident expert layers and6801.21MiB CUDA compute buffer.']}
    with Path(args.output).open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'slot_stride': stride, 'rows': rows}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('rank')
    for key in ('trace', 'inputs', 'integrity', 'tensors', 'output'):
        r.add_argument('--' + key, type=Path, required=True)
    r.add_argument('--slots', type=int, nargs='+', default=[128, 248, 384, 512])
    r.add_argument('--chunk', type=int, default=128)
    b = sub.add_parser('budget')
    b.add_argument('--tensors', type=Path, required=True)
    b.add_argument('--output', type=Path, required=True)
    b.add_argument('--alignment', type=int, default=128)
    b.add_argument('--vram-mib', type=int, default=32110)
    b.add_argument('--chunk', type=int, default=2048)
    b.add_argument('--extra-scratch-gib', type=float, default=2)
    a = p.parse_args()
    (rank if a.command == 'rank' else budget)(a)


if __name__ == '__main__':
    main()
