#!/usr/bin/env python3
"""Freeze an engine and write a NEW speed-only aictl session; never execute it."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import bench_m31
from server_make_session import MODEL, ROOT


def fingerprint(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1 << 20), b''):
            digest.update(chunk)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, default=ROOT / 'build-server-eval/m31_tokenizer')
    parser.add_argument('--ctx', type=int, default=40000)
    parser.add_argument('--chunk', type=int, default=2048)
    parser.add_argument('--threads', type=int, default=64)
    parser.add_argument('--port', type=int, default=8180)
    parser.add_argument('--depths', default='warmup,4K,16K,32K,4K,16K,32K')
    parser.add_argument('--arena-gib', type=float, default=0)
    parser.add_argument('--gpu-experts-gib', type=float, default=0)
    parser.add_argument('--gpu-experts-min-tokens', type=int, default=32)
    parser.add_argument('--env', action='append', default=[], metavar='NAME=VALUE')
    parser.add_argument('--skip-freeze', action='store_true', help='use an already frozen engine')
    parser.add_argument('--build-receipt', type=Path, help='receipt tying the prebuilt ELF to its build/source')
    parser.add_argument('--max-chunk-limit', type=int, default=2048, help='raise only after the engine accepts larger chunks')
    parser.add_argument('--seed-hot', type=int, default=20, help='ranked protected slots when GPU experts are enabled; 0 disables')
    args = parser.parse_args()
    depths = args.depths.split(',')
    sizes = {'warmup': 1024, '4K': 4096, '16K': 16384, '32K': 32768}
    if not depths or any(depth not in sizes for depth in depths):
        parser.error('depths must be warmup/4K/16K/32K')
    # om_bench adds chat/template/instruction tokens after selecting its nominal slice.
    if args.ctx < max(sizes[depth] for depth in depths) + 2048:
        parser.error('context needs nominal depth plus 2048 tokens for template/generation')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,150}', args.arm):
        parser.error('invalid arm')
    if (not 1 <= args.chunk <= args.max_chunk_limit or args.threads < 1 or not 1 <= args.port <= 65535 or args.port == 8179
            or args.arena_gib < 0 or args.gpu_experts_gib < 0 or args.seed_hot < 0):
        parser.error('invalid chunk/threads/port')
    environment = {'GGML_CUDA_DISABLE_FUSION': '1', 'M31_CPU_SHARED_OVERLAP': '1',
                   'M31_REUSE_HOST_READS': '1', 'M31_CPU_SHARED_OVERLAP_TRACE': '0'}
    if args.gpu_experts_gib:
        environment.update(M31_GPU_PREFILL_ASYNC='1', M31_GPU_PREFILL_COMPACT='1',
                           M31_GPU_HOST_THREADS='32', M31_GPU_EXPERT_SORT_ROUTES='1', M31_GPU_CPU_QUANTS='1',
                           M31_GPU_CPU_QUANTS_BATCHED='1', M31_GPU_REUSE_BUFFERS='1', M31_GPU_PARALLEL_REDUCE='1',
                           M31_GPU_PREFETCH_NEXT='0', M31_GPU_CPU_QUANTS_TIMING='0', M31_GPU_EXPERT_PROFILE='0',
                           M31_GPU_CACHED_DECODE='0')
    for item in args.env:
        key, separator, value = item.partition('=')
        if not separator or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key):
            parser.error('--env requires NAME=VALUE')
        if key in ('CUDA_VISIBLE_DEVICES', 'M31_CARD_SESSION'):
            parser.error('card ownership/device environment belongs to card_session')
        environment[key] = value
    original = fingerprint(args.engine)
    tokenizer = fingerprint(args.tokenizer)
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    frozen = args.engine.resolve()
    if not args.skip_freeze:
        frozen = directory / 'm31_serve-frozen'
        with args.engine.open('rb') as source, frozen.open('xb') as destination:
            shutil.copyfileobj(source, destination, 1 << 20)
        frozen.chmod(0o555)
        if fingerprint(frozen)['sha256'] != original['sha256']:
            parser.error('engine changed while freezing; use a completed build')
    # The runner and adapter must remain tied to this session while other agents
    # keep editing their sources. Their code has no local import dependencies.
    frozen_helpers = {}
    for helper in ('serve_m31.py', 'server_session_job.py'):
        source = ROOT / 'tools/m31' / helper
        target = directory / helper
        with source.open('rb') as stream, target.open('xb') as output:
            shutil.copyfileobj(stream, output)
        target.chmod(0o444)
        frozen_helpers[helper] = target
    server = [sys.executable, str(frozen_helpers['serve_m31.py']), '--engine', str(frozen),
              '--tokenizer', tokenizer['path'], '--model', MODEL, '--backend', 'cuda', '--alias', args.arm,
              '--port', str(args.port), '--ctx', str(args.ctx), '--chunk', str(args.chunk), '--threads', str(args.threads),
              '--arena-gib', str(args.arena_gib), '--gpu-experts-gib', str(args.gpu_experts_gib),
              '--gpu-experts-min-tokens', str(args.gpu_experts_min_tokens), '--reuse-allocations',
              '--persistent-cpu-pool', '--cpu-residual', '--cpu-decode-cache', '--dense-decode-cache', '--profile']
    speed = {'name': 'speed', 'env': {'OM_OFF': '{"reasoning_effort":"low"}'},
             'argv': [sys.executable, '/home/nekto/om_bench.py', 'speed', args.arm, str(args.port),
                      str(directory / 'speed.tsv'), args.depths], 'timeout': 14400}
    session_path = directory / (args.arm + '-http-session.json')
    job_path = directory / (args.arm + '-card-job.json')
    session = {'server': server, 'health': f'http://127.0.0.1:{args.port}/health', 'clients': [speed]}
    execute = [sys.executable, str(frozen_helpers['server_session_job.py']), str(session_path)]
    seed = None
    if args.gpu_experts_gib and args.seed_hot:
        slots = int(args.gpu_experts_gib * (1 << 30)) // 25909632
        if not 0 < args.seed_hot < slots:
            parser.error('seed-hot must leave at least one streaming slot')
        seed = {'rank': '/mnt/ai/logs/m31-port/cache-ranking-v3.json',
                'binary': '/mnt/ai/logs/m31-port/cache-ranking-v3.bin',
                'tensors': '/mnt/ai/logs/m31-port/refs/gguf-tensors.json',
                'integrity': '/mnt/ai/logs/m31-ref/integrity-ud-original.json'}
        seed_command = [sys.executable, str(ROOT / 'tools/m31/cache_seed_launch.py')]
        for key, path in seed.items():
            Path(path).resolve(strict=True)
            seed_command += ['--' + key, path]
        execute = seed_command + ['--slots', str(slots), '--hot', str(args.seed_hot), '--output',
                                  str(directory / 'cache-seed-preflight.json'), '--'] + execute
        seed.update(slots=slots, hot=args.seed_hot)
    command = ['/usr/bin/env'] + [key + '=' + value for key, value in sorted(environment.items())] + execute
    job = {'preset': 'm31-xolotl', 'measurement': True, 'commands': [command]}
    backup = '/etc/aictl/config.json.bak-m31-port-' + job_path.stem
    source = {'scope': 'Current workspace at preparation; a prebuilt ELF may come from an earlier source. Use its build receipt for attribution.'}
    for key, argv in [('head', ['git', '-C', str(ROOT), 'rev-parse', 'HEAD']),
                      ('status', ['git', '-C', str(ROOT), 'status', '--short']),
                      ('diff', ['git', '-C', str(ROOT), 'diff', '--', 'tools/m31'])]:
        source[key] = subprocess.check_output(argv, text=True)
    (directory / 'source-diff.patch').write_text(source.pop('diff'))
    source['files'] = [fingerprint(path) for path in sorted((ROOT / 'tools/m31').glob('*'))
                       if path.suffix in ('.cpp', '.cu', '.hpp', '.py', '.sh')]
    cache = args.engine.resolve().parent / 'CMakeCache.txt'
    if cache.is_file():
        shutil.copyfile(cache, directory / 'CMakeCache.txt')
        source['cmake_cache'] = fingerprint(directory / 'CMakeCache.txt')
    manifest = {'format': 'm31-port5-speed-session-v1', 'source': source,
                'original_engine': original, 'engine': fingerprint(frozen), 'tokenizer': tokenizer,
                'environment': environment, 'context': args.ctx, 'chunk': args.chunk, 'threads': args.threads,
                'kv': 'F16 main/index', 'arena_gib': args.arena_gib, 'gpu_experts_gib': args.gpu_experts_gib,
                'gpu_experts_min_tokens': args.gpu_experts_min_tokens,
                'reference': bench_m31.REFERENCE, 'client': fingerprint('/home/nekto/om_bench.py'),
                'corpus': fingerprint('/mnt/ai/logs/q38bench-corpus.txt'),
                'adapter': fingerprint(frozen_helpers['serve_m31.py']),
                'session_runner': fingerprint(frozen_helpers['server_session_job.py']), 'seed': seed,
                'depths': depths, 'complete_two_repetitions': all(depths.count(depth) >= 2 for depth in ('4K', '16K', '32K')),
                'launch_argv': [sys.executable, str(ROOT / 'tools/m31/card_launch.py'), str(job_path), '--unit', args.arm],
                'rollback_argv': ['sudo', 'python3', str(ROOT / 'tools/m31/card_preset.py'), 'rollback', backup, 'm31-xolotl'],
                'limits': ['Prepared only. Coordinator declares an isolated measurement window before launch.',
                           'Use the existing correctness gates for this exact engine before measuring.',
                           'F16 KV differs from reference Q8_0; rank/cache cannot change routing.',
                           'HTTP health is readiness only. No quality result is inferred from this speed session.']}
    if args.build_receipt:
        manifest['build_receipt'] = fingerprint(args.build_receipt)
    for path, value in [(session_path, session), (job_path, job), (directory / 'manifest.json', manifest)]:
        with path.open('x') as destination:
            json.dump(value, destination, ensure_ascii=False, indent=2)
            destination.write('\n')
    print(json.dumps({'job': str(job_path), 'launch_argv': manifest['launch_argv'], 'engine_sha256': manifest['engine']['sha256']}, indent=2))


if __name__ == '__main__':
    main()
