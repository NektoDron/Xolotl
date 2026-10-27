#!/usr/bin/env python3
"""Prepare a new port7 HTTP plan from the accepted v8 profile; never launch it."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from port5_bench_prepare import fingerprint
from server_make_session import ROOT

V8 = Path('/mnt/ai/logs/m31-port/port6r4-final-v8-acceptance-v1/speed')


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--tiny-decode', action='store_true')
    parser.add_argument('--profile', type=Path, default=ROOT / 'tools/m31/profiles/native-gpu.json')
    options, passthrough = parser.parse_known_args()
    if not passthrough or '--help' in passthrough:
        return subprocess.call([sys.executable, str(ROOT / 'tools/m31/port6_bench_prepare.py'), '--help'])
    profile = json.loads(options.profile.read_text())
    env_arguments = [item for key, value in sorted(profile['environment'].items())
                     for item in ('--env', key + '=' + value)]
    command = [sys.executable, str(ROOT / 'tools/m31/port6_bench_prepare.py'),
               '--helpers-from', str(V8), '--chunk', '32768', *env_arguments, *passthrough]
    result = json.loads(subprocess.check_output(command, text=True))
    job_path = Path(result['job'])
    directory = job_path.parent
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    job = json.loads(job_path.read_text())
    # Explicitly isolate this plan from inherited debug/seed controls as well.
    unset = sorted(set(manifest['unset_inherited_environment']) | set(profile['unset_environment']))
    env_command = job['commands'][0]
    original_unset = set(manifest['unset_inherited_environment'])
    env_command[1:1] = [item for key in unset if key not in original_unset for item in ('-u', key)]
    manifest.update(format='m31-port7-http-plan-v1', unset_inherited_environment=unset,
                    runtime_profile=fingerprint(options.profile),
                    v8_comparison={'source': str(V8 / 'report.json'),
                      'pp_median': {'4K': 1390.85, '16K': 1535.0, '32K': 1246.65},
                      'tg_median': {'4K': 27.445, '16K': 27.015, '32K': 25.305}},
                    goals={'tg_each_depth_min': 35.0, 'tg_stretch': 40.0,
                           'prefill_regression': False, 'gsm8k100_min': 95, 'tools23_min': 19})
    manifest['goals_provenance'] = 'Explicit port7 human steering: generation35–40+t/s without losing acceptedv8 prefill or quality.'
    session_path = next(directory.glob('*-http-session.json'))
    session = json.loads(session_path.read_text())
    if options.tiny_decode:
        client = directory / 'port7_decode_client.py'
        shutil.copyfile(ROOT / 'tools/m31/port7_decode_client.py', client)
        client.chmod(0o444)
        port = session['health'].split(':')[-1].split('/')[0]
        session['clients'].append({'name': 'tinydecode', 'argv': [sys.executable, str(client),
             '--port', port, '--out', str(directory / 'tinydecode.jsonl')], 'timeout': 7200})
        manifest['tiny_decode_client'] = fingerprint(client)
    for path, value in ((job_path, job), (session_path, session), (manifest_path, manifest)):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
