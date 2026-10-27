#!/usr/bin/env python3
"""Prepare long-position GPU-oracle checks; replay follows actual native chunks.

Never reserves a GPU. Use prepare, then native job, replay, oracle job, compare.
Every prepared directory is new and contains frozen executables and comparators.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

ROOT = Path('/mnt/ai/Xolotl-m31')
MODEL = '/mnt/ai/models/MiniMax-M3.1-Flash-GGUF/UD-IQ3_S/MiniMax-M3.1-Flash-UD-IQ3_S-00001-of-00005.gguf'


def fingerprint(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def freeze(source, target, executable=False):
    before = fingerprint(source)
    with source.open('rb') as inp, target.open('xb') as out:
        shutil.copyfileobj(inp, out, 1 << 20)
    target.chmod(0o555 if executable else 0o444)
    if fingerprint(target)['sha256'] != before['sha256']:
        raise ValueError('source changed during freeze: ' + str(source))
    return before


def prepare(args):
    from port6_bench_receipt import verify_receipt
    base = json.loads(args.baseline.read_text())
    env = dict(base['environment'])
    for assignment in args.env:
        key, sep, value = assignment.partition('=')
        if not sep or key in ('CUDA_VISIBLE_DEVICES', 'M31_CARD_SESSION'):
            raise ValueError('invalid environment override')
        env[key] = value
    if env.get('GGML_CUDA_DISABLE_FUSION') != '1':
        raise ValueError('long acceptance requires no fusion')
    args.directory.mkdir(parents=True, exist_ok=False)
    directory = args.directory.resolve()
    engine = directory / 'm31_generate-frozen'
    original = freeze(args.engine, engine, True)
    receipt = verify_receipt(args.build_receipt, original['sha256'], 'm31_generate')
    oracle = directory / 'm31_oracle-frozen'
    oracle_original = freeze(args.oracle, oracle, True)
    oracle_receipt = json.loads(args.oracle_receipt.read_text())
    if oracle_receipt.get('elf', {}).get('m31_oracle') != oracle_original['sha256']:
        raise ValueError('oracle receipt ELF mismatch')
    shutil.copyfile(args.oracle_receipt, directory / 'oracle-build-receipt.json')
    helpers = directory / 'comparator'
    helpers.mkdir()
    for name in ('compare_logits.py', 'port6_validate.py', 'port6_validate_long.py'):
        freeze(ROOT / 'tools/m31' / name, helpers / name)
    # Vocabulary-only helper explicitly hides CUDA before initializing llama.
    corpus = Path('/mnt/ai/logs/q38bench-corpus.txt')
    content = corpus.read_text()[:1200000]
    with (directory / 'tokenizer.log').open('x') as log:
        proc = subprocess.Popen([str(args.tokenizer), MODEL], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=log, text=True,
                                env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
        try:
            ready = json.loads(proc.stdout.readline())
            if ready.get('event') != 'ready':
                raise ValueError('tokenizer did not become ready')
            proc.stdin.write(json.dumps({'op': 'tokenize', 'content': content,
                                         'add_special': True, 'parse_special': False}) + '\n')
            proc.stdin.flush()
            ids = json.loads(proc.stdout.readline())['tokens']
        finally:
            proc.stdin.close()
            proc.wait(timeout=30)
    if proc.returncode or len(ids) < 32768 + 7919:
        raise ValueError('invalid or short corpus tokenization')
    ids_dir = directory / 'ids'
    ids_dir.mkdir()
    commands = []
    cases = []
    unset = ['CUDA_DEVICE_MAX_CONNECTIONS', 'CUDA_DEVICE_MAX_COPY_CONNECTIONS']
    case_spec = ((10272, 0),) if args.direct10k else ((16384, 0), (32768, 7919))
    for length, offset in case_spec:
        case = f'corpus_{length}'
        selected = ids[offset:offset + length]
        path = ids_dir / (case + '.i32')
        path.write_bytes(struct.pack('<' + 'i' * length, *selected))
        native = [str(engine), '-m', MODEL, '--backend', 'cuda', '--ctx', '40000',
                  '--chunk', '32768', '--threads', '80', '--arena-gib', '170',
                  '--gpu-experts-gib', '7', '--gpu-experts-min-tokens', '32',
                  '--reuse-allocations', '1', '--persistent-cpu-pool', '1',
                  '--cpu-residual', '1', '--cpu-decode-cache', '1',
                  '--dense-decode-cache', '1', '--profile', '1', '--taps', 'none',
                  '--decode-tail', '32', '--head-tail', '256', '--rows', 'last:256',
                  '--cases', str(ids_dir), '--out', str(directory / 'native')]
        if not commands:
            commands.append(['/usr/bin/env', *[x for name in unset for x in ('-u', name)],
                             *[k + '=' + v for k, v in sorted(env.items())], *native])
        cases.append({'name': case, 'length': length, 'offset': offset,
                      'ids': fingerprint(path), 'native_output': str(directory / 'native')})
    write(directory / 'native-job.json', {'preset': 'm31-xolotl-check', 'measurement': False,
                                         'commands': commands})
    write(directory / 'manifest.json', {'format': 'm31-port6-long-plan-v1',
          'engine': fingerprint(engine), 'build_receipt': receipt, 'environment': env,
          'unset_inherited_environment': unset, 'context': 40000, 'chunk_limit': 32768,
          'threads': 80, 'gpu_experts_gib': 7, 'seed_hot': 0, 'head_tail': 256,
          'decode_tail': 32, 'selected_rows': 'last:256', 'cases': cases,
          'corpus': fingerprint(corpus), 'tokenizer': fingerprint(args.tokenizer),
          'oracle': fingerprint(oracle), 'oracle_build_receipt': fingerprint(args.oracle_receipt),
          'oracle_contract': 'GPU-offload llama.cpp PR31192; F16 KV; no fusion',
          'thresholds': {'top1_min': .99, 'mean_kl_max': .001, 'max_kl_max': .05},
          'scope': ('direct first10240-token batch plus32decode' if args.direct10k else
                    'selected long positions; supplements mandatory complete gate20 and quality'),
          'limits': ['Oracle is prepared only after native actual chunks exist.',
                     '256 final positions per case cover 224 prefill and 32 teacher-forced decode rows.']})
    print(directory / 'native-job.json')


def replay(args):
    directory = args.directory.resolve()
    manifest = json.loads((directory / 'manifest.json').read_text())
    commands = []
    plans = []
    for case in manifest['cases']:
        native = Path(case['native_output']) / case['name']
        meta = json.loads((native / 'manifest.json').read_text())
        chunks = meta['actual_chunks']
        if meta['tokens'] != case['length'] or meta['positions'] != list(range(case['length'] - 256, case['length'])):
            raise ValueError('native long selected rows mismatch')
        expected = 0
        for item in chunks:
            if item['position'] != expected or not 0 < item['tokens'] <= 32768:
                raise ValueError('invalid actual chunk coverage')
            expected += item['tokens']
        if expected != case['length']:
            raise ValueError('actual chunk coverage incomplete')
        path = directory / (case['name'] + '-replay.tsv')
        with path.open('x') as out:
            for item in chunks:
                out.write(f"{item['position']}\t{item['tokens']}\n")
        output = directory / f"oracle-{case['length']}"
        commands.append(['/usr/bin/env', '-u', 'CUDA_DEVICE_MAX_CONNECTIONS',
                         '-u', 'CUDA_DEVICE_MAX_COPY_CONNECTIONS',
                         'GGML_CUDA_DISABLE_FUSION=1', 'M31_ORACLE_CHUNK_PLAN=' + str(path),
                         str(directory / 'm31_oracle-frozen'), '-m', MODEL,
                         '--backend', 'cuda', '--ctx', '40000', '--chunk', '32768',
                         '--threads', '80', '--decode-tail', '32', '--head-tail', '256',
                         '--rows', 'last:256', '--oracle-op-offload', '1',
                         '--ids-file', case['ids']['path'], '--out', str(output)])
        plans.append({'case': case['name'], 'actual_chunks': chunks, 'replay': fingerprint(path)})
    commands.append([sys.executable, str(directory / 'comparator/port6_validate_long.py'),
                     'compare', '--directory', str(directory)])
    write(directory / 'oracle-job.json', {'preset': 'm31-xolotl-check', 'measurement': False,
                                         'commands': commands})
    write(directory / 'replay-receipt.json', {'format': 'm31-port6-long-replay-v1', 'plans': plans})
    print(directory / 'oracle-job.json')


def compare(args):
    import compare_logits
    from port6_validate import THRESHOLDS
    directory = args.directory.resolve()
    manifest = json.loads((directory / 'manifest.json').read_text())
    cases = []
    for case in manifest['cases']:
        reference = directory / f"oracle-{case['length']}" / case['name']
        candidate = Path(case['native_output']) / case['name']
        a, _ = compare_logits.load_case(reference)
        b, _ = compare_logits.load_case(candidate)
        equivalence = manifest.get('capacity_equivalence', {})
        bounded_capacity = equivalence.get('authorized') is True
        expected_context = case['length'] if bounded_capacity else 40000
        if (a['engine'] != 'llama.cpp-PR31192' or b['engine'] != 'xolotl-m31-correctness'
                or not a.get('oracle_op_offload') or a['context'] != expected_context or b['context'] != 40000
                or a.get('cuda_fusion_disabled') is not True or b.get('cuda_fusion_disabled') is not True):
            raise ValueError('invalid long oracle contract')
        if compare_logits.execution_chunks(a) != compare_logits.execution_chunks(b):
            raise ValueError('long actual chunk geometry mismatch')
        value = compare_logits.compare_case(reference, candidate)
        if bounded_capacity:
            trace_path = reference / 'capacity-equivalence.jsonl'
            trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
            if len(trace) != len(a['actual_chunks']):
                raise ValueError('capacity-equivalence trace coverage differs from actual chunks')
            for item, chunk in zip(trace, a['actual_chunks']):
                end = chunk['position'] + chunk['tokens']
                if (item['position'] != chunk['position'] or item['tokens'] != chunk['tokens']
                        or item['seq_min'] != 0 or item['seq_max'] != end - 1
                        or item['active_keys_padded_from_verified_positions'] != ((end + 255) // 256) * 256
                        or item['physical_context_requested'] != case['length']
                        or item['physical_context_actual'] != ((case['length'] + 255) // 256) * 256
                        or item['rope_ext_factor'] != 0 or item['rope_orig_context'] != 1048576
                        or item['rope_frequency_scale'] != 1):
                    raise ValueError('capacity equivalence runtime trace violates contiguous KV / trained RoPE')
            value['capacity_equivalence'] = {'authorized': True, 'native_context': b['context'],
                 'oracle_context_header': a['context'], 'oracle_context_padded': trace[0]['physical_context_actual'],
                 'runtime_trace': fingerprint(trace_path), 'runtime_chunks': trace,
                 'key_padding_basis': 'Audited PR get_n_kv: PAD(used_max_p1,256), clipped by capacity; runtime KV min/max verify contiguous0..end-1 and no wrap/shift.',
                 'rope': {'scaling': 'linear', 'frequency_scale': 1, 'ext_factor': 0, 'original_training_context': 1048576}}
        value['actual_chunks'] = a['actual_chunks']
        value['pass'] = value['top1'] >= THRESHOLDS['top1'] and value['mean_kl'] <= THRESHOLDS['mean_kl'] and value['max_kl'] <= THRESHOLDS['max_kl']
        cases.append(value)
    direct = [c['tokens'] for c in cases] == [10272]
    result = {'format': 'm31-port6-direct-batch-gate-v1' if direct else 'm31-port6-long-gate-v1', 'pass': all(c['pass'] for c in cases),
              'scope': ('GPU-offload nofusion direct10240 batch selected224prefill+32decode' if direct else
                        'GPU-offload nofusion selected256 positions at16K and32K; supplements full20'),
              'engine': manifest['engine'], 'build_receipt': manifest['build_receipt'],
              'environment': manifest['environment'],
              'unset_inherited_environment': manifest['unset_inherited_environment'],
              'cases': cases, 'thresholds': THRESHOLDS}
    write(directory / 'compare.json', result)
    print(json.dumps(result, indent=2))
    return 0 if result['pass'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    for name in ('directory', 'engine', 'build-receipt', 'oracle', 'oracle-receipt', 'baseline'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, default=ROOT / 'build-server-eval/m31_tokenizer')
    p.add_argument('--env', action='append', default=[])
    p.add_argument('--direct10k', action='store_true', help='direct10240-token batch with32decode instead of16K/32K cases')
    p.set_defaults(run=prepare)
    for command, fn in (('replay', replay), ('compare', compare)):
        p = sub.add_parser(command)
        p.add_argument('--directory', type=Path, required=True)
        p.set_defaults(run=fn)
    args = parser.parse_args()
    return args.run(args)


if __name__ == '__main__':
    raise SystemExit(main())
