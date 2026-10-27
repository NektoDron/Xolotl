#!/usr/bin/env python3
"""Same-day raw-ID long-prefill comparison using one native process per ELF.

Run only inside the owned card launcher. This client neither takes a card nor
changes presets. It reuses serve_m31.RPC and never starts HTTP/tokenizer helpers.
"""
from __future__ import annotations

import argparse
import contextlib
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import time

from serve_m31 import RPC
import serve_m31


def fingerprint(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def load_ids(path):
    payload = Path(path).read_bytes()
    if not payload or len(payload) % 4 or len(payload) > 4*1048576:
        raise ValueError('invalid i32 input')
    return list(struct.unpack('<'+'i'*(len(payload)//4), payload))


def raw_request(rpc, ids, label):
    body = {'tokens': ids, 'max_tokens': 1, 'temperature': 0, 'top_p': 1,
            'top_k': 0, 'min_p': 0, 'seed': 0, 'ignore_eos': True,
            'interactive': False, 'cache_prompt': False, 'ttft_trace': True}
    started = time.perf_counter()
    prefill, token, done, first_token_ms = None, None, None, None
    with rpc.lock:
        rpc.write(body)
        while done is None:
            frame = rpc.read()
            event = frame.get('event')
            if event == 'error' or 'error' in frame:
                raise RuntimeError(f'{label}: {frame}')
            if event == 'prefill' and prefill is None:
                prefill = frame
            elif event == 'token' and token is None:
                token = frame
                first_token_ms = 1000*(time.perf_counter()-started)
            elif event == 'done':
                done = frame
            else:
                raise RuntimeError(f'{label}: unexpected/duplicate native frame {frame}')
    wall_ms = 1000*(time.perf_counter()-started)
    if prefill is None or token is None or token.get('index') != 0:
        raise RuntimeError(f'{label}: missing prefill/first token')
    timings = done['timings']
    chunks = timings['actual_chunks']
    position = 0
    for chunk in chunks:
        if chunk['position'] != position or chunk['rows'] <= 0:
            raise RuntimeError(f'{label}: invalid chunk sequence')
        position += chunk['rows']
    if position != len(ids) or timings['prompt_n'] != len(ids) or timings['predicted_n'] != 1:
        raise RuntimeError(f'{label}: did not execute entire fresh prefill')
    cache = timings.get('prompt_cache', prefill.get('prompt_cache', {}))
    if cache.get('hit') or cache.get('enabled'):
        raise RuntimeError(f'{label}: forbidden prompt cache reuse')
    stages = timings['ttft_stages_ms']
    if 'reset_ms' not in stages:
        raise RuntimeError(f'{label}: reset was not traced')
    forward_ms = stages['prefill_forward_ms']
    prompt_ms = timings['prompt_ms']
    if any(not math.isfinite(value) or value <= 0 for value in (wall_ms, forward_ms, prompt_ms)):
        raise RuntimeError(f'{label}: invalid timings')
    return {'label': label, 'pass': True, 'tokens': len(ids),
            'request_parameters': {k: v for k, v in body.items() if k != 'tokens'},
            'request_token_sha256': hashlib.sha256(struct.pack('<'+'i'*len(ids), *ids)).hexdigest(),
            'actual_chunks': chunks, 'forward_ms': forward_ms,
            'forward_tps': 1000*len(ids)/forward_ms, 'head_ms': stages.get('prefill_head_ms'),
            'native_prompt_ms': prompt_ms, 'native_prompt_tps': timings['prompt_per_second'],
            'wall_ms': wall_ms, 'wall_tps': 1000*len(ids)/wall_ms,
            'first_token_wall_ms': first_token_ms, 'sampled_token': token['token'],
            'prompt_cache_proof': {'environment_budget_mib': 0, 'request_cache_prompt': False,
                                   'reported': cache, 'all_rows_fresh': True},
            'native_timings': timings}


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        if not row['warmup']:
            groups[(row['arm'], row['tokens'])].append(row)
    result = []
    for (arm, tokens), samples in sorted(groups.items()):
        value = {'arm': arm, 'tokens': tokens, 'samples': len(samples),
                 'actual_chunks': [r['actual_chunks'] for r in samples]}
        for key in ('forward_ms', 'forward_tps', 'native_prompt_ms', 'native_prompt_tps',
                    'wall_ms', 'wall_tps', 'first_token_wall_ms'):
            value[key] = statistics.median(r[key] for r in samples)
        result.append(value)
    return result


def close_gracefully(rpc, timeout=300):
    """Let native arena/CUDA destruction finish before RPC's short fallback."""
    started = time.perf_counter()
    try:
        if rpc.process.poll() is None:
            with contextlib.suppress(BrokenPipeError, OSError, ValueError):
                rpc.process.stdin.close()
            try:
                rpc.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f'native graceful shutdown exceeded {timeout}s') from error
    finally:
        rpc.close()
    return 1000*(time.perf_counter()-started)


def environment_argv(profile, overrides):
    if overrides.get('M31_PROMPT_CACHE_MIB', '0') != '0':
        raise ValueError('long benchmark requires M31_PROMPT_CACHE_MIB=0')
    unset = set(profile.get('unset_environment', []))
    unset.update(key for key in os.environ if key.startswith('M31_SHORT_'))
    # Exercise safe GPU quant/OAI defaults even if the caller inherited flags.
    unset.update(('M31_SHORT_PREFILL_GPU_QUANT', 'M31_SHORT_PREFILL_GPU_OAI',
                  'M31_SHORT_PREFILL_TASK_PROFILE'))
    argv = ['/usr/bin/env']
    for key in sorted(unset):
        argv += ['-u', key]
    env = {**profile['environment'], 'M31_GPU_PREFETCH_HISTORY': '0',
           'M31_PROMPT_CACHE_MIB': '0', **overrides}
    argv += [f'{key}={value}' for key, value in env.items()]
    return argv, env


def run(plan, output):
    output.mkdir(exist_ok=False)
    profile = plan['profile']
    arguments = profile['engine_arguments']
    if len(arguments) % 2:
        raise ValueError('native profile arguments must be flag/value pairs')
    flags = dict(zip(arguments[::2], arguments[1::2]))
    required = {'--ctx': 40000, '--chunk': 32768, '--threads': 80,
                '--arena-gib': 170, '--gpu-experts-gib': 7}
    if flags.get('--backend') != 'cuda' or any(float(flags.get(k, 'nan')) != v for k, v in required.items()):
        raise ValueError('profile differs from required native configuration')
    prewarm = load_ids(plan['prewarm_ids'])
    corpus = load_ids(plan['corpus_ids'])
    lengths = plan.get('lengths', [4096, 16384, 32768])
    if len(prewarm) != 515 or max(lengths) > len(corpus):
        raise ValueError('invalid prewarm515/corpus length')
    result = {'format': 'm31-shortprefill-long-rpc-v1', 'pass': False,
              'configuration': {'context': 40000, 'chunk': 32768, 'threads': 80,
                                'arena_gib': 170, 'gpu_experts_gib': 7},
              'client': fingerprint(__file__), 'rpc_adapter': fingerprint(serve_m31.__file__),
              'profile': profile, 'inputs': {'prewarm': fingerprint(plan['prewarm_ids']),
                                           'corpus': fingerprint(plan['corpus_ids'])},
              'engine_initializations': 0, 'arms': [], 'prewarms': [], 'runs': []}
    if plan.get('client_sha256') not in (None, result['client']['sha256']):
        raise ValueError('client changed since preparation')
    if plan.get('rpc_adapter_sha256') not in (None, result['rpc_adapter']['sha256']):
        raise ValueError('RPC adapter changed since preparation')
    def save():
        result['aggregate'] = aggregate(result['runs'])
        (output/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    save()
    try:
        for arm in plan['arms']:
            identity = fingerprint(arm['binary'])
            receipt = json.loads(Path(arm['build_receipt']).read_text())
            if receipt['files']['m31_serve']['sha256'] != identity['sha256']:
                raise ValueError('frozen ELF differs from build receipt')
            launch, env = environment_argv(profile, arm['environment'])
            launch += [arm['binary'], '--model', plan['model'], *profile['engine_arguments']]
            record = {'name': arm['name'], 'binary': identity,
                      'build_receipt': fingerprint(arm['build_receipt']),
                      'source_identity_sha256': hashlib.sha256(json.dumps(receipt['source'], sort_keys=True).encode()).hexdigest(),
                      'source_head': receipt.get('head'), 'source_files': receipt['source'],
                      'environment': env, 'argv': launch}
            result['arms'].append(record)
            print('M31 LONG RPC init arm='+arm['name'], file=sys.stderr, flush=True)
            started = time.perf_counter()
            rpc = RPC(launch, read_timeout=plan.get('timeout', 900), startup_timeout=900)
            result['engine_initializations'] += 1
            record['startup_ms'] = 1000*(time.perf_counter()-started)
            record['ready'] = rpc.ready
            try:
                if rpc.ready['context'] != 40000:
                    raise RuntimeError('unexpected context')
                if len(result['arms']) > 1 and rpc.ready['vocab'] != result['arms'][0]['ready']['vocab']:
                    raise RuntimeError('binary vocabulary mismatch')
                row = raw_request(rpc, prewarm, arm['name']+'-prewarm515')
                row['arm'] = arm['name']
                result['prewarms'].append(row)
                save()
                for iteration in range(-plan.get('warmups', 1), plan.get('repeats', 2)):
                    for length in lengths:
                        label = f'{arm["name"]}-{length}-{iteration}'
                        print('M31 LONG RPC sample '+label, file=sys.stderr, flush=True)
                        row = raw_request(rpc, corpus[:length], label)
                        row.update({'arm': arm['name'], 'iteration': iteration, 'warmup': iteration < 0})
                        result['runs'].append(row)
                        with (output/'runs.jsonl').open('a') as stream:
                            stream.write(json.dumps(row)+'\n')
                        save()
            finally:
                try:
                    record['shutdown_ms'] = close_gracefully(rpc, plan.get('shutdown_timeout', 300))
                finally:
                    record['returncode'] = rpc.process.returncode
            if record['returncode'] != 0:
                raise RuntimeError('native process did not exit cleanly')
        result['pass'] = True
        save()
        return result
    except BaseException as error:
        result['error'] = f'{type(error).__name__}: {error}'
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('M31_CARD_SESSION') != '1' or os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        parser.error('requires an aictl-owned GPU0 session')
    plan = json.loads(args.plan.read_text())
    if plan.get('format') != 'm31-shortprefill-long-rpc-plan-v1':
        parser.error('unknown plan format')
    value = run(plan, args.out)
    print(json.dumps({'pass': value['pass'], 'engines': value['engine_initializations'],
                      'runs': len(value['runs']), 'aggregate': value['aggregate']}))


if __name__ == '__main__':
    main()
