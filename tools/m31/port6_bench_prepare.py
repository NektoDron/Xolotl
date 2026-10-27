#!/usr/bin/env python3
"""Prepare a frozen port6 speed/quality HTTP session; never execute or reserve GPU.

All port5 preparation safeguards are reused. Pass through its --engine,
--directory, --arm, --ctx, --chunk, --env and receipt options unchanged.
"""
import argparse
import json
from pathlib import Path
import subprocess
import shutil
import sys

from port5_bench_prepare import fingerprint
from port6_bench_receipt import verify_receipt
from server_make_session import ROOT

LOG = Path('/mnt/ai/logs/m31-port')


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--quality', action='store_true', help='append original GSM8K100+tools23 after speed')
    parser.add_argument('--quality-only', action='store_true', help='only original GSM8K100+tools23, no speed')
    parser.add_argument('--helpers-from', type=Path, help='copy known stable serve_m31.py/session runner from an immutable earlier session')
    options, passthrough = parser.parse_known_args()
    if not passthrough or '--help' in passthrough:
        print(__doc__)
        print('Extra options: --quality or --quality-only. Remaining options:')
        return subprocess.call([sys.executable, str(ROOT / 'tools/m31/port5_bench_prepare.py'), '--help'])
    # Adopt measured port5 performance settings, then apply caller overrides.
    environment = json.loads((LOG / 'port5-gpu-v4-full-summary-v1.json').read_text())['environment']
    environment.update(M31_GPU_EXACT='0', M31_GPU_FAST_FFN='0', M31_GPU_CPU_QUANTS='0',
                       M31_GPU_FAST_FFN_DOTS='cublas', M31_GPU_FAST_FFN_LIBM_EXPF='0', M31_GPU_FAST_FFN_EXP_DOUBLE='0')
    defaults = ['--ctx', '40000', '--chunk', '8192', '--threads', '80', '--arena-gib', '170',
                '--gpu-experts-gib', '7', '--max-chunk-limit', '32768', '--seed-hot', '0']
    for key, value in sorted(environment.items()):
        defaults += ['--env', key + '=' + value]
    # No shell interpolation. This writes a new directory and never launches it.
    prepared = subprocess.run([sys.executable, str(ROOT / 'tools/m31/port5_bench_prepare.py'),
                               *defaults, *passthrough], check=True, capture_output=True, text=True)
    result = json.loads(prepared.stdout)
    job_path = Path(result['job'])
    directory = job_path.parent
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if options.helpers_from:
        helper_sources = {}
        for name, key in [('serve_m31.py', 'adapter'), ('server_session_job.py', 'session_runner')]:
            source = options.helpers_from.resolve() / name
            target = directory / name
            target.chmod(0o644)
            with source.open('rb') as stream, target.open('wb') as destination:
                shutil.copyfileobj(stream, destination)
            target.chmod(0o444)
            manifest[key] = fingerprint(target)
            helper_sources[name] = fingerprint(source)
        manifest['immutable_helper_sources'] = helper_sources
    if 'build_receipt' in manifest:
        manifest['build_receipt'] = verify_receipt(Path(manifest['build_receipt']['path']), manifest['engine']['sha256'], 'm31_serve')
    job = json.loads(job_path.read_text())
    unset_inherited = ['CUDA_DEVICE_MAX_CONNECTIONS', 'CUDA_DEVICE_MAX_COPY_CONNECTIONS']
    job['commands'][0][1:1] = [item for name in unset_inherited for item in ('-u', name)]
    manifest['unset_inherited_environment'] = unset_inherited
    manifest.update(format='m31-port6-http-plan-v1', numeric_contract={'top1_min': .99,
                    'mean_kl_max': .001, 'max_kl_max': .05, 'bitwise_required': False,
                    'oracle': 'GPU-offload llama.cpp PR31192', 'cpu_strict': 'opt-in verification only'},
                    goals={'pp32k_min': 1000, 'tg4k_min': 25.0, 'tg16k_min': 25.0,
                           'tg32k_min': 25.0, 'gsm8k100_min': 95, 'tools23_min': 19})
    manifest['goals_provenance'] = 'Human coordinator steering for resumed port6: generation >=25 t/s at every depth; PP32K >=1000 t/s. Prior port5 medians remain comparison data.'
    session_path = directory / (job_path.name[:-len('-card-job.json')] + '-http-session.json')
    session = json.loads(session_path.read_text())
    if options.quality or options.quality_only:
        arm = session['clients'][0]['argv'][3]
        port = session['clients'][0]['argv'][4]
        quality = {'name': 'quality', 'argv': [sys.executable, '/home/nekto/eval_duty_1023.py',
                    arm, port, str(directory), '100'],
                   'env': {'EVAL_TEMP': '1.0', 'EVAL_TOPP': '0.95', 'EVAL_MAX': '6000'},
                   'unset_env': ['EVAL_TOPK', 'EVAL_MINP', 'OM_OFF'], 'timeout': 86400}
        session['clients'] = [quality] if options.quality_only else [*session['clients'], quality]
        manifest['quality'] = {'gsm8k_cases': 100, 'tools_cases': 23, 'gsm8k_min': 95,
                               'tools_min': 19, 'client': fingerprint('/home/nekto/eval_duty_1023.py'),
                               'dataset': fingerprint('/mnt/ai/eval/gsm8k_test.jsonl'),
                               'sampling': quality['env'], 'thinking': 'model default adaptive',
                               'tools_executed': False}
    if options.quality_only:
        job['measurement'] = False
        manifest['depths'] = []
        manifest['complete_two_repetitions'] = False
    job_path.write_text(json.dumps(job, indent=2) + '\n')
    session_path.write_text(json.dumps(session, indent=2) + '\n')
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
