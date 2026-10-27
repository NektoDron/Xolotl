#!/usr/bin/env python3
"""Prepare immutable Q2_K_XL reference HTTP jobs; never reserve a GPU here."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('/mnt/ai/llama.cpp-m31pr-1026')
MODEL = Path('/mnt/ai/models/MiniMax-M3.1-Flash-GGUF/UD-Q2_K_XL/MiniMax-M3.1-Flash-UD-Q2_K_XL-00001-of-00004.gguf')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--mode', choices=('speed', 'quality'), required=True)
    parser.add_argument('--port', type=int, default=8183)
    parser.add_argument('--layers', default='10,11,12')
    args = parser.parse_args()
    layers = [int(value) for value in args.layers.split(',')]
    if not layers or len(layers) != len(set(layers)) or any(value < 3 or value >= 60 for value in layers):
        parser.error('invalid full expert layers')
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    arm = 'm31-q2-ref-e' + str(len(layers))
    source_binary = SOURCE / 'build/bin/llama-server'
    binary = directory / 'llama-server-frozen'
    shutil.copy2(source_binary, binary)
    binary.chmod(0o555)
    runner = directory / 'server_session_job.py'
    shutil.copy2(ROOT / 'tools/m31/server_session_job.py', runner)
    runner.chmod(0o444)
    (directory / 'eval').mkdir()
    # This is the exact resolved stock m31 command from start_llm.sh, with only
    # model/alias/port/log paths changed. Avoid start_llm's unrelated-port pgrep.
    override = 'blk\\.(' + '|'.join(map(str, layers)) + ')\\.ffn_.*_exps\\.=CUDA0,ffn_.*_exps\\.=CPU'
    unset = ('GGML_CUDA_DISABLE_FUSION', 'GGML_CUDA_DISABLE_GRAPHS',
             'GGML_CUDA_FORCE_MMQ', 'GGML_CUDA_FORCE_CUBLAS',
             'GGML_OP_OFFLOAD_MIN_BATCH', 'CUDA_DEVICE_MAX_CONNECTIONS',
             'CUDA_DEVICE_MAX_COPY_CONNECTIONS', 'OMP_WAIT_POLICY', 'GOMP_SPINCOUNT')
    server = ['/usr/bin/env']
    for key in unset:
        server.extend(('-u', key))
    server.extend(('CUDA_VISIBLE_DEVICES=0', str(binary), '--model', str(MODEL),
                   '--alias', arm, '--host', '127.0.0.1', '--port', str(args.port),
                   '--ctx-size', '65536', '--parallel', '1', '--threads', '64',
                   '--threads-batch', '96', '--batch-size', '2048', '--ubatch-size', '2048',
                   '--n-gpu-layers', '999', '--split-mode', 'layer', '--tensor-split', '1,1',
                   '--load-mode', 'none', '--fit', 'off', '--log-verbosity', '4',
                   '-ot', override, '--flash-attn', 'on', '--cache-type-k', 'q8_0',
                   '--cache-type-v', 'q8_0', '--temp', '1.0', '--top-p', '0.95',
                   '--cont-batching', '--jinja', '--metrics', '--slots', '--log-timestamps', '--log-prefix'))
    client_unset = ['EVAL_TOPK', 'EVAL_MINP', 'EVAL_MAX', 'OM_OFF']
    if args.mode == 'speed':
        clients = [{'argv': ['/usr/bin/python3', '/home/nekto/om_bench.py', 'speed', arm,
                             str(args.port), str(directory / 'speed.tsv'), 'warmup,4K,16K,32K'],
                    'env': {'OM_OFF': '{"reasoning_effort":"low"}'}, 'timeout': 14400}]
    else:
        clients = [{'argv': ['/usr/bin/python3', '/home/nekto/eval_duty_1023.py', arm,
                             str(args.port), str(directory / 'eval'), '100'],
                    'env': {'EVAL_TEMP': '1.0', 'EVAL_TOPP': '0.95'},
                    'unset_env': client_unset, 'timeout': 21600}]
    session = directory / (directory.name + '-http-session.json')
    session.write_text(json.dumps({'server': server, 'health': f'http://127.0.0.1:{args.port}/health',
                                   'ready_timeout': 600, 'clients': clients}, indent=2) + '\n')
    job = directory / (directory.name + '-card-job.json')
    job.write_text(json.dumps({'preset': 'm31-xolotl-check', 'measurement': args.mode == 'speed',
                               'command_timeout_seconds': 24000,
                               'commands': [['/usr/bin/python3', str(runner), str(session)]]}, indent=2) + '\n')
    tracked = [source_binary, binary, runner, ROOT / 'tools/m31/q2_ref_prepare.py',
               Path('/home/nekto/om_bench.py'), Path('/home/nekto/eval_duty_1023.py'),
               Path('/mnt/ai/logs/q38bench-corpus.txt'), Path('/mnt/ai/eval/gsm8k_test.jsonl')]
    manifest = {'mode': args.mode, 'model': str(MODEL), 'model_revision': 'b17b7f5459c0b8f127ea12ee774409b282521d6a',
                'llama_sha': subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
                'source_status': subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'], text=True),
                'reference_baseline': '/mnt/ai/logs/m31-ref/REPORT.md',
                'expert_layers': layers, 'server_argv': server, 'files': {str(p): digest(p) for p in tracked},
                'launch_argv': ['/usr/bin/python3', str(ROOT / 'tools/m31/card_launch.py'), str(job),
                                '--unit', 'codex-m31-' + directory.name],
                'limits': ['Preparation only. Launch only after coordinator GPU-window approval.',
                           'Speed owns exclusive build lock; quality owns shared build lock.',
                           'Quality uses original eval client with temperature=1/top_p=.95/max_tokens=6000.',
                           'No model payload checksum repeated; user verified four revision-pinned shards.']}
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
