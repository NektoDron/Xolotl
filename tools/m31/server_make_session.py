#!/usr/bin/env python3
"""Write NEW reviewable session/job files; never start a server or change aictl."""
import argparse
import json
from pathlib import Path
import sys

import bench_m31

ROOT = Path(__file__).resolve().parents[2]
MODEL = '/mnt/ai/models/MiniMax-M3.1-Flash-GGUF/UD-IQ3_S/MiniMax-M3.1-Flash-UD-IQ3_S-00001-of-00005.gguf'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--arm', required=True)
    p.add_argument('--ctx', type=int, required=True)
    p.add_argument('--chunk', type=int, default=128)
    p.add_argument('--threads', type=int, default=64)
    p.add_argument('--port', type=int, default=8180)
    p.add_argument('--mode', choices=['smoke', 'toolsmoke', 'tools23', 'speed4k', 'full'], default='smoke')
    p.add_argument('--engine', default=str(ROOT / 'build-m31-cuda/m31_serve'))
    p.add_argument('--tokenizer', default=str(ROOT / 'build-server-eval/m31_tokenizer'))
    p.add_argument('--cpu-decode-cache', action='store_true')
    p.add_argument('--dense-decode-cache', action='store_true')
    p.add_argument('--arena-gib', type=float, default=0)
    p.add_argument('--gpu-experts-gib', type=float, default=0)
    a = p.parse_args()
    if a.port == 8179:
        p.error('8179 is owned by card_worker health; use a separate HTTP port')
    if a.ctx < (40000 if a.mode == 'full' else 8192) or not 1 <= a.chunk <= 2048 or a.threads < 1:
        p.error('insufficient context or invalid chunk/threads; full suite needs >=40000 context')
    a.directory = a.directory.resolve()
    # Existing benchmark scripts append or overwrite their files. Require a new directory.
    a.directory.mkdir(parents=True, exist_ok=False)
    plan = bench_m31.plan(argparse.Namespace(arm=a.arm, port=a.port, directory=str(a.directory), home='/home/nekto'))
    server = [sys.executable, str(ROOT / 'tools/m31/serve_m31.py'), '--engine', a.engine,
              '--tokenizer', a.tokenizer, '--model', MODEL, '--backend', 'cuda', '--alias', a.arm,
              '--port', str(a.port), '--ctx', str(a.ctx), '--chunk', str(a.chunk), '--threads', str(a.threads),
              '--arena-gib', str(a.arena_gib), '--gpu-experts-gib', str(a.gpu_experts_gib),
              '--reuse-allocations', '--persistent-cpu-pool', '--cpu-residual']
    if a.cpu_decode_cache:
        server.append('--cpu-decode-cache')
    if a.dense_decode_cache:
        server.append('--dense-decode-cache')
    clients = [{'argv': [sys.executable, str(ROOT / 'tools/m31/server_smoke.py'),
                         '--port', str(a.port), '--out', str(a.directory / 'http-smoke.json')]}]
    if a.mode in ('toolsmoke', 'tools23'):
        clients = [{'argv': [sys.executable, str(ROOT / 'tools/m31/server_tool_smoke.py'),
                             '--port', str(a.port), '--out', str(a.directory / 'http-tool-smoke.json')], 'timeout': 3600}]
    if a.mode == 'tools23':
        clients.append({'argv': [sys.executable, str(ROOT / 'tools/m31/server_tools23.py'),
                                '--port', str(a.port), '--name', a.arm, '--directory', str(a.directory)],
                        'env': {'EVAL_TEMP': '1.0', 'EVAL_TOPP': '0.95', 'EVAL_MAX': '6000'},
                        'unset_env': ['EVAL_TOPK', 'EVAL_MINP', 'OM_OFF'], 'timeout': 14400})
    if a.mode in ('speed4k', 'full'):
        speed = plan['commands'][1]
        if a.mode == 'speed4k':
            speed['argv'][-1] = 'warmup,4K,4K'
        clients.append({**speed, 'timeout': 14400})
    if a.mode == 'full':
        clients.append({**plan['commands'][2], 'unset_env': ['OM_OFF'], 'timeout': 86400})
    session_path = a.directory / (a.arm + '-http-session.json')
    job_path = a.directory / (a.arm + '-card-job.json')
    session = {'server': server, 'health': f'http://127.0.0.1:{a.port}/health', 'clients': clients}
    job = {'preset': 'm31-xolotl', 'measurement': a.mode in ('speed4k', 'full'), 'commands': [[sys.executable, str(ROOT / 'tools/m31/server_session_job.py'), str(session_path)]]}
    backup = Path('/etc/aictl') / ('config.json.bak-m31-port-' + job_path.stem)
    manifest = {'plan': plan, 'aictl_preset': 'm31-xolotl',
                'launch_argv': [sys.executable, str(ROOT / 'tools/m31/card_launch.py'), str(job_path), '--unit', a.arm],
                'rollback_argv': ['sudo', 'python3', str(ROOT / 'tools/m31/card_preset.py'), 'rollback', str(backup), 'm31-xolotl'],
                'limits': ['Preparation only; no GPU or config changes.',
                           'Coordinator must authorize order of quality/card jobs and the exclusive measurement window for speed.',
                           'card_launch freezes card_session to a per-job read-only snapshot with SHA256; card and build locks precede preset backup/start/cleanup.',
                           'Speed/full use exclusive .build.lock; smoke/toolsmoke/tools23 explicitly set measurement=false for shared .build.lock.',
                           'Current KV is F16; full 32K suite requires a separate successful VRAM/context gate.']}
    for name, data in [(session_path, session), (job_path, job), (a.directory / 'manifest.json', manifest)]:
        with name.open('x') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write('\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
