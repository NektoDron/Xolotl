#!/usr/bin/env python3
"""Prepare/replay/audit selected UD-Q2 stock-path numeric diagnostics; never launch GPU."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil
import sys

ROOT = Path('/mnt/ai/Xolotl-m31')
LOG = Path('/mnt/ai/logs/m31-port')
MODEL = Path('/mnt/ai/models/MiniMax-M3.1-Flash-GGUF/UD-Q2_K_XL/MiniMax-M3.1-Flash-UD-Q2_K_XL-00001-of-00004.gguf')


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def fp(path):
    path = path.resolve(strict=True)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            h.update(part)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())


def freeze(source, target, executable=False):
    before = fp(source)
    with source.open('rb') as src, target.open('xb') as dst:
        shutil.copyfileobj(src, dst, 1 << 20)
    target.chmod(0o555 if executable else 0o444)
    result = fp(target)
    if result['sha256'] != before['sha256']:
        raise ValueError('source changed while freezing: ' + str(source))
    return result


def verified_receipt(path, binary, name):
    receipt = json.loads(path.read_text())
    if receipt.get('elf', {}).get(name) != binary['sha256'] or not receipt.get('source'):
        raise ValueError('build receipt does not identify ' + name)
    return fp(path)


def prepare(args):
    directory = args.directory.resolve()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{1,120}', directory.name):
        raise ValueError('directory basename must be a unique safe arm name')
    directory.mkdir(parents=True, exist_ok=False)
    native = freeze(args.engine, directory / 'm31_generate-frozen', True)
    oracle = freeze(args.oracle, directory / 'm31_oracle-frozen', True)
    native_receipt = verified_receipt(args.build_receipt, native, 'm31_generate')
    oracle_receipt = verified_receipt(args.oracle_receipt, oracle, 'm31_oracle')
    freeze(args.build_receipt, directory / 'native-build-receipt.json')
    freeze(args.oracle_receipt, directory / 'oracle-build-receipt.json')
    profile = json.loads(args.profile.read_text())
    if profile.get('inherits'):
        raise ValueError('profile must be resolved')
    env = profile['environment'].copy()
    env.update(M31_CPU_DIRECT='0', M31_CPU_FUSED=str(args.cpu_stock_experts), M31_CPU_STOCK_EXPERTS=str(args.cpu_stock_experts))
    disabled = ('M31_GPU_CPU_QUANTS', 'M31_GPU_EXACT', 'M31_GPU_FAST_FFN',
                'M31_GPU_CACHED_DECODE', 'M31_GPU_CACHED_DECODE_EXACT',
                'M31_GPU_CACHED_DECODE_NATIVE', 'M31_GPU_DECODE_SEED_HOT',
                'M31_GPU_DECODE_FUSION', 'M31_GPU_RMS_FUSION',
                'M31_DECODE_FUSION_UNFENCED')
    if any(env.get(k, '0') != '0' for k in disabled) or env.get('GGML_CUDA_DISABLE_FUSION') != '1':
        raise ValueError('stock no-fusion/no-cache profile required')
    if any(v != '0' for k, v in env.items() if k.startswith('M31_DSPARK_')):
        raise ValueError('DSpark must be disabled for this diagnostic')
    unset = sorted(set(profile.get('unset_environment', [])) | set(profile.get('unset_inherited_environment', [])) |
                   {'CUDA_DEVICE_MAX_CONNECTIONS', 'CUDA_DEVICE_MAX_COPY_CONNECTIONS', 'M31_ORACLE_CHUNK_PLAN', 'M31_ORACLE_CAPACITY_EQUIVALENCE'})
    env_argv = ['/usr/bin/env', *[v for key in unset for v in ('-u', key)],
                *[k + '=' + str(v) for k, v in sorted(env.items())]]
    write(directory / 'resolved-profile.json', dict(environment=env, unset_environment=unset,
          source_profile=fp(args.profile), policy=('opt-in stock-dot IQ2 CPU wrappers' if args.cpu_stock_experts else 'stock CPU RoutedGraph fallback') + '; native GPU prefill; no IQ3 rank/cached/exact/DSpark'))
    freeze(ROOT / 'tools/m31/q2_compat_prepare.py', directory / 'q2_compat_prepare.py')
    freeze(ROOT / 'tools/m31/compare_logits.py', directory / 'compare_logits.py')
    specs = [
        ('code128', LOG / 'ids-v1/code_02_128.i32', 128, 4096, 2048, 32, 'all', 0),
        ('code3000', LOG / 'ids-v1/code_18_3000.i32', 3000, 4096, 8192, 0, 'last:32', 32),
        ('long16384', LOG / 'port6r4-final-v8-acceptance-v1/long/ids/corpus_16384.i32', 16384, 40000, 8192, 32, 'last:32', 32),
        ('long32768', LOG / 'port6r4-final-v8-acceptance-v1/long/ids/corpus_32768.i32', 32768, 40000, 8192, 32, 'last:32', 32),
    ]
    cases = []
    for name, source, count, context, chunk, tail, rows, head in specs:
        case_dir = directory / name
        case_dir.mkdir()
        if source.stat().st_size != count * 4:
            raise ValueError('unexpected ID extent ' + str(source))
        ids = freeze(source, case_dir / (source.stem + '.i32'))
        common = ['-m', str(MODEL), '--backend', 'cuda', '--ctx', str(context), '--chunk', str(chunk),
                  '--threads', '80', '--rows', rows, '--taps', 'none', '--ids-file', ids['path']]
        if tail:
            common += ['--decode-tail', str(tail)]
        if head:
            common += ['--head-tail', str(head)]
        native_command = [*env_argv, native['path'], *common,
             '--reuse-allocations', '1', '--persistent-cpu-pool', '1', '--cpu-residual', '1',
             '--cpu-decode-cache', '1', '--dense-decode-cache', '1', '--profile', '1',
             '--arena-gib', '140', '--gpu-experts-gib', '7', '--gpu-experts-min-tokens', '32',
             '--out', str(case_dir / 'native')]
        job = case_dir / (directory.name + '-' + name + '-native.json')
        write(job, dict(preset='m31-xolotl-check', measurement=False,
              command_timeout_seconds=1800, commands=[native_command]))
        case = dict(name=name, stem=source.stem, ids=ids, tokens=count, context=context,
                    chunk=chunk, decode_tail=tail, rows=rows, head_tail=head, native_job=str(job),
                    oracle_common=common)
        write(case_dir / 'plan.json', case)
        cases.append(case)
    shards = []
    for path in sorted(MODEL.parent.glob('*.gguf')):
        stat = path.stat()
        shards.append(dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
    if len(shards) != 4:
        raise ValueError('requires exactly four Q2 shards')
    write(directory / 'manifest.json', dict(format='m31-q2-stock-compat-v1', acceptance_eligible=False,
          model=str(MODEL), model_revision='b17b7f54', model_payload_checksums='Verified externally by user; not rehashed by this preparation.',
          shards=shards, native=native, oracle=oracle, native_receipt=native_receipt,
          oracle_receipt=oracle_receipt, profile=fp(directory / 'resolved-profile.json'), cases=cases,
          fusion_policy='CUDA fusion disabled in both; all native per-backend fusion overrides0.',
          oracle_contract='PR31192 GPU op-offload; experts stored CPU, stock CUDA MMQ/MMVQ as selected by oracle scheduler.',
          geometry_policy='Run native first; oracle replays actual contiguous native chunk boundaries. Same requested physical context; no silent capacity fallback.',
          limitations=['Four selected teacher-forced cases only; not gate20, greedy rollout, GSM/tools or performance acceptance.',
                       'Correctness dumps use F16 KV; reference speed server uses Q8 KV and different ubatch.']))
    print(json.dumps(dict(directory=str(directory), native_jobs=[c['native_job'] for c in cases]), indent=2))


def load_case(args):
    root = args.directory.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    case = next(c for c in manifest['cases'] if c['name'] == args.case)
    return root, manifest, case, root / args.case


def completed(job_path):
    job = json.loads(job_path.read_text())
    result = json.loads(job_path.with_suffix('.result.json').read_text())
    if result.get('returncode') != 0 or len(result['commands']) != len(job['commands']):
        raise ValueError('job did not complete successfully: ' + str(job_path))
    for plan, observed in zip(job['commands'], result['commands']):
        if observed['argv'] != plan or observed['returncode'] != 0:
            raise ValueError('completed argv differs from plan')
    return result


def replay(args):
    root, manifest, case, directory = load_case(args)
    result = completed(Path(case['native_job']))
    header = json.loads((directory / 'native' / case['stem'] / 'manifest.json').read_text())
    chunks = header['actual_chunks']
    if not chunks or sum(c['tokens'] for c in chunks) != case['tokens']:
        raise ValueError('actual native chunks missing/incomplete')
    cursor = 0
    for c in chunks:
        if c['position'] != cursor or not 0 < c['tokens'] <= case['chunk']:
            raise ValueError('actual chunks not contiguous or above declared cap')
        cursor += c['tokens']
    tsv = directory / 'oracle-chunk-plan.tsv'
    with tsv.open('x') as stream:
        for c in chunks:
            stream.write(f"{c['position']}\t{c['tokens']}\n")
    job = directory / (root.name + '-' + case['name'] + '-oracle.json')
    write(job, dict(preset='m31-xolotl-check', measurement=False, command_timeout_seconds=1800,
          commands=[['/usr/bin/env', '-u', 'CUDA_DEVICE_MAX_CONNECTIONS', '-u', 'CUDA_DEVICE_MAX_COPY_CONNECTIONS',
                     '-u', 'M31_ORACLE_CAPACITY_EQUIVALENCE', 'GGML_CUDA_DISABLE_FUSION=1',
                     'M31_ORACLE_CHUNK_PLAN=' + str(tsv), manifest['oracle']['path'], *case['oracle_common'],
                     '--oracle-op-offload', '1', '--out', str(directory / 'oracle')]]))
    write(directory / 'replay-receipt.json', dict(native_result=fp(Path(case['native_job']).with_suffix('.result.json')),
          native_log=fp(Path(result['commands'][0]['log'])), native_header=fp(directory / 'native' / case['stem'] / 'manifest.json'),
          actual_chunks=chunks, chunk_plan=fp(tsv), job=str(job)))
    print(job)


def compare(args):
    from compare_logits import compare_case, execution_chunks
    root, manifest, case, directory = load_case(args)
    completed(Path(case['native_job']))
    replay_receipt = json.loads((directory / 'replay-receipt.json').read_text())
    completed(Path(replay_receipt['job']))
    for role in ('native', 'oracle'):
        descriptor = manifest[role]
        if fp(Path(descriptor['path'])) != descriptor:
            raise ValueError(role + ' ELF changed')
    native = directory / 'native' / case['stem']
    oracle = directory / 'oracle' / case['stem']
    a = json.loads((oracle / 'manifest.json').read_text())
    b = json.loads((native / 'manifest.json').read_text())
    for h in (a, b):
        if h['model'] != manifest['model'] or h['context'] != case['context'] or not h['cuda_fusion_disabled']:
            raise ValueError('model/context/fusion mismatch')
    if not a['oracle_op_offload'] or a['engine'] != 'llama.cpp-PR31192' or b['engine'] != 'xolotl-m31-correctness':
        raise ValueError('wrong engine/placement contract')
    if execution_chunks(a) != execution_chunks(b) or a['head_tail'] != b['head_tail']:
        raise ValueError('different actual execution/head geometry')
    if b.get('m31_decode_fusion_override_requested') or b.get('m31_decode_fusion_unfenced_requested'):
        raise ValueError('native has fusion override')
    result = compare_case(oracle, native)
    result.update(format='m31-q2-stock-compat-result-v1', acceptance_eligible=False,
          thresholds=dict(top1=.99, mean_kl=.001, max_kl=.05),
          native_header=fp(native / 'manifest.json'), oracle_header=fp(oracle / 'manifest.json'),
          actual_chunks=execution_chunks(a), context=case['context'], ids=case['ids'])
    result['pass'] = result['top1'] >= .99 and result['mean_kl'] <= .001 and result['max_kl'] <= .05
    write(directory / 'compare.json', result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result['pass'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action', required=True)
    p = subs.add_parser('prepare')
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--engine', type=Path, default=ROOT / 'build-port7-canonical/v2/bin-cuda/m31_generate')
    p.add_argument('--build-receipt', type=Path, default=LOG / 'port7-canonical-preflight-v3/canonical-cuda-compat-receipt.json')
    p.add_argument('--oracle', type=Path, default=ROOT / 'build-port6-bench-oracle/replay-v3/build/m31_oracle')
    p.add_argument('--oracle-receipt', type=Path, default=ROOT / 'build-port6-bench-oracle/replay-v3/build-receipt.json')
    p.add_argument('--profile', type=Path, default=LOG / 'port7-canonical-preflight-v3/baseline-resolved.json')
    p.add_argument('--cpu-stock-experts', type=int, choices=(0, 1), default=0, help='Explicit IQ2 CPU wrapper opt-in; requires a newly validated build.')
    for action in ('replay', 'compare'):
        p = subs.add_parser(action)
        p.add_argument('--directory', type=Path, required=True)
        p.add_argument('--case', choices=('code128', 'code3000', 'long16384', 'long32768'), required=True)
    args = parser.parse_args()
    return globals()[args.action](args)


if __name__ == '__main__':
    raise SystemExit(main())
