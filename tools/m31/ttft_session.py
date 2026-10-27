#!/usr/bin/env python3
"""Same-request cold/warm TTFT probes. Run only in the owned GPU0 card job.

The encoder embedding cache is disabled for this diagnostic session. The
full-prompt cache is retained, then archived between cold/warm-worker probes;
the latter therefore runs the same media bytes through an already loaded
encoder. Every inference still resets the native LLM as the server normally does.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from profile_plan import load_profile
from vision_http_session import fingerprint, standalone_arguments
from vision_validation import MODEL, MMPROJ, REFERENCE, image, text

ROOT = Path(__file__).resolve().parents[2]


class Vocab:
    def __init__(self, binary, model, log):
        self.log = log.open('wb')
        self.process = subprocess.Popen([str(binary), str(model)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=self.log, text=True,
                                        env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
        ready = json.loads(self.process.stdout.readline())
        if ready.get('event') != 'ready':
            self.close()
            raise RuntimeError('vocabulary helper did not become ready')

    def call(self, request):
        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()
        result = json.loads(self.process.stdout.readline())
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()


def exact_text_chat(vocab, corpus_ids, target):
    """Natural user text, exact total rendered token count, same low template."""
    seen = {}

    def candidate(length):
        if length not in seen:
            payload = vocab.call({'op': 'detokenize', 'tokens': corpus_ids[:length]})['content']
            messages = [{'role': 'user', 'content': 'Read the following excerpt, then reply only OK.\n' + payload + '\nReply only OK.'}]
            rendered = vocab.call({'op': 'chat', 'messages': messages, 'tools': [],
                                   'tool_choice': 'auto', 'parallel_tool_calls': True,
                                   'enable_thinking': True,
                                   'chat_template_kwargs': {'reasoning_effort': 'low'}})
            seen[length] = (messages, rendered['tokens'])
        return seen[length]

    low, high = 0, min(len(corpus_ids), target + 32)
    if len(candidate(low)[1]) > target or len(candidate(high)[1]) < target:
        raise ValueError('target does not fit template/corpus')
    while low <= high:
        middle = (low + high) // 2
        messages, ids = candidate(middle)
        if len(ids) == target:
            return messages, ids
        if len(ids) < target:
            low = middle + 1
        else:
            high = middle - 1
    # BPE boundary changes can skip a count during the binary search.
    for length in range(max(0, high - 16), min(len(corpus_ids), low + 16) + 1):
        messages, ids = candidate(length)
        if len(ids) == target:
            return messages, ids
    raise ValueError(f'could not render exactly {target} text tokens')


def changed_question_same_tokens(vocab, original, variants):
    """Choose an unseen prompt without changing the ordinary text token count."""
    def count(value):
        return len(vocab.call({'op': 'tokenize', 'content': value,
                               'add_special': False, 'parse_special': True})['tokens'])
    expected = count(original)
    for variant in variants:
        if variant != original and count(variant) == expected:
            return variant
    raise ValueError('no length-matched question variation found')


def request_options(max_tokens=1200, allow_prompt_cache=False):
    return {'model': 'm31-ttft', 'stream': True, 'temperature': 0, 'seed': 0,
            'reasoning_effort': 'low', 'cache_prompt': bool(allow_prompt_cache),
            'stream_options': {'include_usage': True}, 'max_tokens': max_tokens}


def cache_optout_body(body):
    return {**body, 'cache_prompt': False}


def assert_optout(result, previous_exact, expected_rows):
    """Prove an explicitly forbidden reuse after a populated exact cache hit."""
    reference_info = previous_exact.get('native_timings', {}).get('prompt_cache', {})
    if reference_info.get('hit') is not True:
        raise RuntimeError('optout control must follow an observed populated exact KV hit')
    native = result.get('native_timings', {})
    info = native.get('prompt_cache', {})
    chunks = native.get('actual_chunks', [])
    checks = {'no_cache_hit': not bool(info.get('hit', False)),
              'one_actual_prefill': len(chunks) == 1 and chunks[0].get('position') == 0 and chunks[0].get('rows') == expected_rows,
              'same_prompt_payload': result['request_prompt_sha256'] == previous_exact['request_prompt_sha256'],
              'same_content': result['content'] == previous_exact['content'],
              'same_reasoning': result['reasoning_content'] == previous_exact['reasoning_content'],
              'same_finish': result.get('finish_reason') == previous_exact.get('finish_reason'),
              'same_generated_count': native.get('predicted_n') == previous_exact.get('native_timings', {}).get('predicted_n')}
    result['prompt_cache_optout_checks'] = checks
    if not all(checks.values()):
        raise RuntimeError('KV optout/greedy equivalence failed: '+str(checks))


def request(port, path, body, request_id, timeout):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
    started = time.perf_counter()
    encoded = json.dumps(body).encode()
    serialized = time.perf_counter()
    result = {'id': request_id, 'endpoint': path, 'request_bytes': len(encoded),
              'client_json_ms': (serialized - started) * 1000,
              'ttft_s': None, 'visible_ttft_s': None, 'reasoning_ttft_s': None,
              'content': '', 'reasoning_content': '', 'events': []}
    try:
        connection.request('POST', path, encoded, {'Content-Type': 'application/json',
                                                 'X-M31-TTFT-ID': request_id})
        response = connection.getresponse()
        result['headers_s'] = time.perf_counter() - started
        if response.status != 200:
            raise RuntimeError(f'HTTP {response.status}: {response.read().decode(errors="replace")}')
        complete = False
        while line := response.readline():
            if not line.startswith(b'data: '):
                continue
            payload = line[6:].strip()
            if payload == b'[DONE]':
                complete = True
                break
            event = json.loads(payload)
            result['events'].append(event)
            if 'error' in event:
                raise RuntimeError(str(event['error']))
            choice = event.get('choices', [{}])[0]
            delta = choice.get('delta', {})
            content = delta.get('content', '') or choice.get('text', '') or ''
            reasoning = delta.get('reasoning_content', '') or ''
            elapsed = time.perf_counter() - started
            if (content or reasoning) and result['ttft_s'] is None:
                result['ttft_s'] = elapsed
            if content and result['visible_ttft_s'] is None:
                result['visible_ttft_s'] = elapsed
            if reasoning and result['reasoning_ttft_s'] is None:
                result['reasoning_ttft_s'] = elapsed
            result['content'] += content
            result['reasoning_content'] += reasoning
            if event.get('timings') is not None:
                result['native_timings'] = event['timings']
            if event.get('usage') is not None:
                result['usage'] = event['usage']
            if choice.get('finish_reason'):
                result['finish_reason'] = choice['finish_reason']
        if not complete or result['ttft_s'] is None:
            raise RuntimeError('missing DONE/generated delta')
        result['wall_s'] = time.perf_counter() - started
        if result['visible_ttft_s'] is not None:
            result['reasoning_to_content_s'] = result['visible_ttft_s'] - result['ttft_s']
        return result
    finally:
        connection.close()


def trace_for(path, request_id, wait=2):
    deadline = time.monotonic() + wait
    while True:
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if value.get('id') == request_id:
                    return value
        if time.monotonic() >= deadline:
            return None
        time.sleep(.02)


def archive_batches(cache_dir, archive_dir, label, expected_rows):
    """Move only this completed diagnostic's full-batch entry, never sources."""
    archive_dir.mkdir(exist_ok=True)
    moved = []
    for manifest in cache_dir.glob('*/batch/manifest.json'):
        metadata = json.loads(manifest.read_text())
        if metadata['rows'] != expected_rows:
            continue
        entry = manifest.parent.parent
        if len(entry.name) != 64 or any(c not in '0123456789abcdef' for c in entry.name):
            raise RuntimeError('unexpected cache entry name')
        destination = archive_dir / (label + '-' + entry.name)
        entry.rename(destination)
        moved.append({'original': str(entry), 'archive': str(destination),
                      'rows': metadata['rows'], 'backend': metadata['backend'],
                      'source_manifest_is_historical': True})
    if len(moved) != 1:
        raise RuntimeError(f'expected one cache batch for {expected_rows}, got {len(moved)}')
    return moved


def breakdown(result):
    """Disjoint milestone partition; nested timers are never added twice."""
    trace = result['trace']
    points = trace['points_ms']
    names = [('http_receive_json_ms', None, 'http_body_parsed'),
             ('server_prepare_queue_ms', 'http_body_parsed', 'prepared'),
             ('prepare_to_native_rpc_ms', 'prepared', 'native_rpc_start'),
             ('native_rpc_to_first_token_ms', 'native_rpc_start', 'first_native_token'),
             ('token_parse_to_first_delta_ms', 'first_native_token', 'first_generated_delta')]
    partition = {name: points[end] - (points[start] if start else 0)
                 for name, start, end in names}
    partition['client_network_and_start_offset_ms'] = result['ttft_s']*1000 - points['first_generated_delta']
    native = result.get('native_timings', {}).get('ttft_stages_ms', {})
    cache = result.get('native_timings', {}).get('prompt_cache') or trace.get('details', {}).get('native_prefill', {}).get('prompt_cache', {})
    leaf_names = ('json_parse_ms', 'mixed_manifest_read_validate_ms', 'request_setup_ms',
                  'reset_ms', 'prefill_plan_ms', 'prefill_forward_ms', 'prefill_head_ms',
                  'first_sampling_including_grammar_ms')
    leaves = {key: native[key] for key in leaf_names if key in native}
    cache_leaves = {key: cache[key] for key in ('geometry_probe_ms', 'key_ms', 'lookup_ms',
                    'restore_ms', 'snapshot_store_ms') if key in cache}
    leaves.update({'prompt_cache.'+key: value for key, value in cache_leaves.items()})
    native_first = native.get('first_token_sent_from_request_ms')
    return {'critical_path_ms': partition,
            'critical_path_sum_ms': sum(partition.values()),
            'client_ttft_ms': result['ttft_s']*1000,
            'partition_residual_ms': result['ttft_s']*1000 - sum(partition.values()),
            'native_before_first_token_ms': leaves,
            'native_prompt_cache_ms': cache_leaves,
            'native_before_first_token_leaf_sum_ms': sum(leaves.values()),
            'native_unattributed_before_first_token_ms': None if native_first is None else native_first - sum(leaves.values()),
            'native_transport_ms': None if native_first is None else partition['native_rpc_to_first_token_ms'] - native_first,
            'after_first_token_ms': {key: native[key] for key in ('first_decode_forward_ms', 'first_decode_head_ms') if key in native},
            'reasoning_until_content_ms': None if points.get('first_content_delta') is None else points['first_content_delta']-points['first_generated_delta'],
            'notes': 'Milestone partition is disjoint; nested media/encoder timers explain parents and are not summed with them. First decode forward follows the first native token; if the parser withholds that token, decode can fall within token_parse_to_first_delta_ms and must not be added again.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--engine', type=Path, default=ROOT/'build-m31-cuda/m31_serve')
    parser.add_argument('--adapter', type=Path, default=ROOT/'build-vision/m31_vision_adapter')
    parser.add_argument('--tokenizer', type=Path, default=Path('/mnt/ai/logs/m31-vision/final-build/m31_tokenizer'))
    parser.add_argument('--model', type=Path, default=Path(MODEL))
    parser.add_argument('--mmproj', type=Path, default=Path(MMPROJ))
    parser.add_argument('--profile', type=Path, default=ROOT/'tools/m31/profiles/native-gpu.json')
    parser.add_argument('--corpus', type=Path, default=Path('/mnt/ai/logs/q38bench-corpus.txt'))
    parser.add_argument('--port', type=int, default=8193)
    parser.add_argument('--context', type=int, default=40000)
    parser.add_argument('--backend', choices=['cpu', 'cuda', 'auto'], default='auto')
    parser.add_argument('--max-tokens', type=int, default=1200)
    parser.add_argument('--embedding-cache-mb', type=int, default=0,
                        help='0 isolates real warm-worker encoding; 256 matches production memory cache')
    parser.add_argument('--prewarm-encoder', action='store_true',
                        help='start encoder after native model load, before HTTP readiness')
    parser.add_argument('--allow-prompt-cache', action='store_true',
                        help='send cache_prompt=true only in the explicit whole-prompt KV arm')
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--repeat', type=int, default=2)
    parser.add_argument('--first-media', choices=['image', 'video'], default='image')
    parser.add_argument('--env', action='append', default=[], help='explicit candidate ENV NAME=VALUE')
    parser.add_argument('--raw-probe', action='store_true')
    args = parser.parse_args()
    if os.environ.get('M31_CARD_SESSION') != '1' or os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        parser.error('requires an aictl-owned GPU0 session and exclusive speed lock')
    if not 1 <= args.repeat <= 10:
        parser.error('repeat must be 1..10')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with socket.socket() as check:
        check.bind(('127.0.0.1', args.port))
    profile = load_profile(args.profile)
    env = os.environ.copy()
    for key in profile['unset_environment']:
        env.pop(key, None)
    env.update(profile['environment'])
    env.setdefault('M31_GPU_PREFETCH_HISTORY', '0')
    env.setdefault('M31_PROMPT_CACHE_MIB', '0')
    for item in args.env:
        key, separator, value = item.partition('=')
        if not separator or key in ('CUDA_VISIBLE_DEVICES', 'M31_CARD_SESSION'):
            parser.error('invalid environment override')
        env[key] = value
    if args.allow_prompt_cache and int(env['M31_PROMPT_CACHE_MIB']) <= 0:
        parser.error('--allow-prompt-cache requires positive M31_PROMPT_CACHE_MIB')
    trace = out/'trace.jsonl'
    env['M31_TTFT_TRACE'] = str(trace)
    config = {'adapter': str(args.adapter.resolve()), 'model': str(args.model.resolve()),
              'mmproj': str(args.mmproj.resolve()), 'cache_dir': str(out/'cache'),
              'backend': args.backend, 'threads': 16, 'cpu_workers': 6,
              'embedding_cache_mb': args.embedding_cache_mb,
              'limits': {'allow_local': True, 'allow_http': False, 'video_fps': 1}}
    if args.prewarm_encoder:
        config['prewarm'] = True
    config_path = out/'vision-config.json'
    config_path.write_text(json.dumps(config, indent=2)+'\n')
    vocab = Vocab(args.tokenizer, args.model, out/'vocab.log')
    text_cases = {}
    image_prompt = 'Describe the visible shape, its color and its position. Be concise.'
    video_prompt = 'Describe the sequence of changes in this video in chronological order: the shape, color and position in each stage. Be concise.'
    try:
        corpus_ids = vocab.call({'op': 'tokenize', 'content': args.corpus.read_text()[:200000],
                                 'add_special': False, 'parse_special': False})['tokens']
        for count in (515, 3863):
            messages, ids = exact_text_chat(vocab, corpus_ids, count)
            text_cases[count] = {'messages': messages, 'tokens': ids}
        changed_image_prompt = changed_question_same_tokens(vocab, image_prompt,
            [image_prompt.replace('visible', word) for word in ('shown', 'pictured', 'displayed')])
        changed_video_prompt = changed_question_same_tokens(vocab, video_prompt,
            [video_prompt.replace('changes', word) for word in ('events', 'shifts', 'stages')])
    finally:
        vocab.close()
    (out/'text-inputs.json').write_text(json.dumps(text_cases, ensure_ascii=False, indent=2)+'\n')
    common = request_options(args.max_tokens, args.allow_prompt_cache)
    png = {**common, 'messages': [{'role': 'user', 'content': [
        text(image_prompt), image(REFERENCE/'0.png')]}]}
    video = {**common, 'video_fps': 1, 'messages': [{'role': 'user', 'content': [
        text(video_prompt),
        {'type': 'video_url', 'video_url': {'url': 'file://'+str(REFERENCE/'shapes-12s.mp4')}}]}]}
    changed_png = json.loads(json.dumps(png))
    changed_png['messages'][0]['content'][0] = text(changed_image_prompt)
    changed_video = json.loads(json.dumps(video))
    changed_video['messages'][0]['content'][0] = text(changed_video_prompt)
    cases = []
    for index in range(args.repeat):
        cases.append((f'text515-{index}', '/v1/chat/completions', {**common, 'messages': text_cases[515]['messages']}, 515, False))
    media_order = [('image', png, changed_png, 515), ('video', video, changed_video, 3863)]
    if args.first_media == 'video':
        media_order.reverse()
    for index, (label, body, changed_body, count) in enumerate(media_order):
        initial_state = 'configured-prewarm' if index == 0 and args.prewarm_encoder else 'cold' if index == 0 else 'warm'
        cases.extend([(f'{label}-worker-{initial_state}-media-cold', '/v1/chat/completions', body, count, True),
                      (f'{label}-worker-warm-same-media', '/v1/chat/completions', body, count, False),
                      (f'{label}-worker-warm-newprompt', '/v1/chat/completions', changed_body, count, False)])
        for repeat in range(args.repeat):
            cases.append((f'{label}-fullcache-{repeat}', '/v1/chat/completions', body, count, False))
        if args.allow_prompt_cache:
            cases.append((f'{label}-prompt-cache-optout', '/v1/chat/completions', cache_optout_body(body), count, False))
    for index in range(args.repeat):
        cases.append((f'text3863-{index}', '/v1/chat/completions', {**common, 'messages': text_cases[3863]['messages']}, 3863, False))
    if args.raw_probe:
        cases.append(('text515-raw-same-ids', '/v1/completions',
                      {**common, 'prompt': text_cases[515]['tokens'], 'max_tokens': 16, 'ignore_eos': True}, 515, False))
    argv = [sys.executable, str(ROOT/'tools/m31/serve_m31.py'), '--engine', str(args.engine.resolve()),
            '--tokenizer', str(args.tokenizer.resolve()), '--model', str(args.model.resolve()),
            '--port', str(args.port), '--alias', 'm31-ttft', '--vision-config', str(config_path),
            *standalone_arguments(profile['engine_arguments'], args.context, None)]
    receipt = {'format': 'm31-ttft-session-v1', 'server_argv': argv, 'environment': {
        key: value for key, value in env.items() if key.startswith(('M31_', 'GGML_'))},
        'files': [fingerprint(p) for p in (args.engine, args.adapter, args.tokenizer, args.profile, Path(__file__))],
        'encoder_embedding_cache_mb': args.embedding_cache_mb,
        'encoder_prewarm_requested': args.prewarm_encoder,
        'prompt_cache_requested': env.get('M31_PROMPT_CACHE_MIB', '0'),
        'body_cache_prompt_default': args.allow_prompt_cache,
        'prompt_cache_optout_probes': args.allow_prompt_cache,
        'prefetch_history_requested': env.get('M31_GPU_PREFETCH_HISTORY', '0'),
        'llm_prefix_reuse_default': 'off; exact prompt cache is separately controlled by explicit ENV',
        'trace_file': str(trace), 'cases': [], 'pass': False}
    (out/'plan.json').write_text(json.dumps({'cases': [dict(name=n, path=p, expected_rows=c, body=b) for n,p,b,c,_ in cases]}, ensure_ascii=False, indent=2)+'\n')
    process = None
    def stop(signum, frame):
        raise InterruptedError(f'TTFT session interrupted ({signum})')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        with (out/'server.log').open('x') as log:
            process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True)
            start = time.monotonic()
            while True:
                if process.poll() is not None:
                    raise RuntimeError('server exited before readiness')
                if time.monotonic()-start > args.timeout:
                    raise TimeoutError('server readiness timeout')
                connection = http.client.HTTPConnection('127.0.0.1', args.port, timeout=2)
                try:
                    connection.request('GET', '/health')
                    response = connection.getresponse()
                    response.read()
                    if response.status == 200:
                        break
                except OSError:
                    pass
                finally:
                    connection.close()
                time.sleep(.2)
            receipt['server_startup_s'] = time.monotonic()-start
            for number, (name, path, body, count, archive) in enumerate(cases):
                request_id = f'{out.name[:32]}-{number:02d}-{name}'
                result = {'name': name, 'expected_rows': count,
                          'request_body_sha256': hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
                          'request_prompt_sha256': hashlib.sha256(json.dumps({key: value for key, value in body.items() if key != 'cache_prompt'}, sort_keys=True).encode()).hexdigest(),
                          'body_cache_prompt': body['cache_prompt'],
                          'kv_expectation': 'miss: explicit cache_prompt=false' if not body['cache_prompt'] else
                                            'miss: unseen prompt' if name.endswith('newprompt') else
                                            'possible exact hit: explicitly enabled'}
                try:
                    result.update(request(args.port, path, body, request_id, args.timeout))
                    actual = result.get('native_timings', {}).get('prompt_n')
                    if actual != count:
                        raise RuntimeError(f'prompt_n={actual}, expected {count}')
                    result['trace'] = trace_for(trace, request_id)
                    if result['trace'] is None:
                        raise RuntimeError('missing per-request TTFT trace')
                    result['observed_encoder_worker_started'] = result['trace'].get('details', {}).get('encoder', {}).get('worker_started')
                    result['breakdown'] = breakdown(result)
                    if not body['cache_prompt'] and result.get('native_timings', {}).get('prompt_cache', {}).get('hit'):
                        raise RuntimeError('cache_prompt=false was ignored by native engine')
                    if name.endswith('newprompt') and result.get('native_timings', {}).get('prompt_cache', {}).get('hit'):
                        raise RuntimeError('unseen prompt unexpectedly reused a KV checkpoint')
                    if name.endswith('prompt-cache-optout'):
                        assert_optout(result, receipt['cases'][-1], count)
                    if archive:
                        result['archived_batches'] = archive_batches(out/'cache', out/'cache-archive', name, count)
                    result['pass'] = True
                except Exception as exc:
                    result.update({'pass': False, 'error': f'{type(exc).__name__}: {exc}'})
                receipt['cases'].append(result)
                (out/'results.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
                print(json.dumps({key: result.get(key) for key in ('name', 'pass', 'ttft_s', 'visible_ttft_s', 'error')}), flush=True)
                if not result['pass']:
                    raise RuntimeError(f'case failed: {name}')
            receipt['pass'] = all(c['pass'] for c in receipt['cases'])
    except Exception as exc:
        receipt['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        receipt['server_returncode'] = None if process is None else process.returncode
        (out/'results.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    return 0 if receipt['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
