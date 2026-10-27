#!/usr/bin/env python3
"""Prepare frozen Q2 HTTP speed/quality jobs; card_launch remains explicit."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from profile_plan import load_profile

ROOT = Path(__file__).resolve().parents[2]
MODEL = Path('/mnt/ai/models/MiniMax-M3.1-Flash-GGUF/UD-Q2_K_XL/MiniMax-M3.1-Flash-UD-Q2_K_XL-00001-of-00004.gguf')
BOOLS = {'--reuse-allocations', '--persistent-cpu-pool', '--cpu-residual',
         '--cpu-decode-cache', '--dense-decode-cache', '--profile'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', required=True, type=Path)
    p.add_argument('--canonical', required=True, type=Path)
    p.add_argument('--profile', type=Path, default=ROOT / 'tools/m31/profiles/q2-native.json')
    p.add_argument('--mode', required=True, choices=('speed', 'quality'))
    p.add_argument('--port', type=int, default=8180)
    p.add_argument('--arena-gib', type=float, default=140.0)
    args = p.parse_args()
    canonical = args.canonical.resolve(strict=True)
    receipt = json.loads((canonical / 'build-receipt.json').read_text())
    if not receipt.get('compile_complete') or not receipt.get('source_stable'):
        p.error('complete, source-stable canonical build required')
    if not MODEL.is_file():
        p.error('Q2 first shard missing')
    profile = load_profile(args.profile)
    profile.pop('inherits', None)
    profile['engine_arguments'] = list(profile['engine_arguments'])
    ai = profile['engine_arguments'].index('--arena-gib')
    profile['engine_arguments'][ai + 1] = str(args.arena_gib)
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'eval').mkdir()
    frozen = {}
    copies = {
        canonical / 'bin-cuda/m31_serve': directory / 'm31_serve-frozen',
        canonical / 'runtime-helpers/m31_tokenizer': directory / 'm31_tokenizer-frozen',
        canonical / 'source-tree/tools/m31/serve_m31.py': directory / 'serve_m31.py',
        canonical / 'source-tree/tools/m31/server_session_job.py': directory / 'server_session_job.py',
    }
    for source, target in copies.items():
        shutil.copy2(source, target)
        target.chmod(0o555 if '-frozen' in target.name else 0o444)
        if sha(source) != sha(target):
            raise RuntimeError('snapshot mismatch: ' + str(source))
        frozen[str(target)] = {'source': str(source), 'sha256': sha(target)}
    runtime = directory / 'runtime-profile.json'
    runtime.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + '\n')
    server = ['/usr/bin/python3', str(directory / 'serve_m31.py'),
              '--engine', str(directory / 'm31_serve-frozen'),
              '--tokenizer', str(directory / 'm31_tokenizer-frozen'),
              '--model', str(MODEL), '--alias', directory.name, '--port', str(args.port)]
    engine_args = profile['engine_arguments']
    for i in range(0, len(engine_args), 2):
        key, value = engine_args[i:i + 2]
        if key in BOOLS:
            if value == '1':
                server.append(key)
            elif value != '0':
                raise ValueError('boolean engine argument: ' + key)
        else:
            server.extend((key, value))
    if args.mode == 'speed':
        clients = [{'argv': ['/usr/bin/python3', '/home/nekto/om_bench.py', 'speed',
                             directory.name, str(args.port), str(directory / 'speed.tsv'),
                             'warmup,4K,16K,32K'],
                    'env': {'OM_OFF': '{"reasoning_effort":"low"}'}, 'timeout': 14400}]
    else:
        clients = [{'argv': ['/usr/bin/python3', '/home/nekto/eval_duty_1023.py',
                             directory.name, str(args.port), str(directory / 'eval'), '100'],
                    'env': {'EVAL_TEMP': '1.0', 'EVAL_TOPP': '0.95'},
                    'unset_env': ['EVAL_TOPK', 'EVAL_MINP', 'EVAL_MAX', 'OM_OFF'],
                    'timeout': 21600}]
    session = directory / (directory.name + '-http-session.json')
    session.write_text(json.dumps({'server': server, 'health': f'http://127.0.0.1:{args.port}/health',
                                   'ready_timeout': 600, 'clients': clients}, indent=2) + '\n')
    command = ['/usr/bin/env']
    for key in profile['unset_environment']:
        command.extend(('-u', key))
    command.extend(key + '=' + value for key, value in sorted(profile['environment'].items()))
    command.extend(('/usr/bin/python3', str(directory / 'server_session_job.py'), str(session)))
    job = directory / (directory.name + '-card-job.json')
    job.write_text(json.dumps({'preset': 'm31-xolotl-check', 'measurement': args.mode == 'speed',
                               'command_timeout_seconds': 24000, 'commands': [command]}, indent=2) + '\n')
    external = [Path('/home/nekto/om_bench.py'), Path('/home/nekto/eval_duty_1023.py'),
                Path('/mnt/ai/eval/gsm8k_test.jsonl'), Path('/mnt/ai/logs/q38bench-corpus.txt'),
                args.profile.resolve(), canonical / 'build-receipt.json']
    manifest = {'format': 'm31-q2-http-job-v1', 'mode': args.mode, 'canonical': str(canonical),
                'model': str(MODEL), 'model_revision': 'b17b7f5459c0b8f127ea12ee774409b282521d6a',
                'model_payload_checksum': 'User verified; not repeated by this preparation script.',
                'profile': str(runtime), 'profile_sha256': sha(runtime), 'frozen': frozen,
                'external_sha256': {str(path): sha(path) for path in external},
                'launch_argv': ['/usr/bin/python3', str(ROOT / 'tools/m31/card_launch.py'), str(job),
                                '--unit', 'codex-m31-' + directory.name],
                'acceptance': 'PENDING: no IQ3 or reference quality acceptance inherited.',
                'locks': 'speed exclusive; quality shared; all GPU0 through aictl/card lock'}
    (directory / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
