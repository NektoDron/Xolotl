#!/usr/bin/env python3
"""Continue prepared long verify jobs after a completed, guarded causal pilot.

Runs outside the card session; every GPU command is submitted by card_launch.py.
The prepared jobs and their original output directories remain authoritative.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/mnt/ai/Xolotl-m31')


def run(argv):
    subprocess.run([str(x) for x in argv], check=True)


def await_result(job, timeout=3600):
    result = job.with_suffix('.result.json')
    deadline = time.monotonic() + timeout
    while not result.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('no atomic result for ' + str(job))
        time.sleep(2)
    value = json.loads(result.read_text())
    if value.get('returncode') != 0:
        raise ValueError('prerequisite/job failed: ' + str(result))
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--causal-job', type=Path, required=True)
    parser.add_argument('--directory', type=Path, action='append', required=True)
    args = parser.parse_args()
    await_result(args.causal_job)
    causal = args.causal_job.parent
    manifest = json.loads((causal / 'manifest.json').read_text())
    for binding in manifest['bindings']:
        if binding['mode'] != 'spec':
            continue
        ci, ri = binding['command'], binding['reference_command']
        reference = manifest['bindings'][ri]
        run(['flock', '-s', '-w', '86400', '/mnt/ai/logs/m31-port/.build.lock',
             sys.executable, Path(__file__).with_name('port7_sweep_compare.py'),
             '--reference-job', args.causal_job, '--reference-command', ri,
             '--reference-log', causal / (args.causal_job.stem + f'-{ri:02}.log'),
             '--reference-ids', Path(reference['output']) / 'corpus_3324.generated.i32',
             '--candidate-job', args.causal_job, '--candidate-command', ci,
             '--candidate-log', causal / (args.causal_job.stem + f'-{ci:02}.log'),
             '--summary', Path(binding['output']) / 'summary.jsonl', '--expected-arms', '1',
             '--out', causal / f'strict-causal-arm{ci}-audit-v1.json'])
    for directory in args.directory:
        plan = json.loads((directory / 'manifest.json').read_text())
        for role in ('native', 'oracle'):
            if role == 'oracle':
                run([sys.executable, Path(__file__).with_name('port7_verify_gate.py'),
                     'replay', '--directory', directory])
            job = directory / (plan['arm'] + '-' + role + '-card-job.json')
            if job.with_suffix('.result.json').exists():
                raise ValueError('refusing to relaunch completed job ' + str(job))
            run([sys.executable, ROOT / 'tools/m31/card_launch.py', '--unit',
                 'codex-m31-' + plan['arm'] + '-' + role, job])
            await_result(job)
        run(['flock', '-s', '-w', '86400', '/mnt/ai/logs/m31-port/.build.lock',
             sys.executable, Path(__file__).with_name('port7_verify_gate.py'),
             'compare', '--directory', directory])


if __name__ == '__main__':
    main()
