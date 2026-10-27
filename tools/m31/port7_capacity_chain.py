#!/usr/bin/env python3
"""Continue explicitly authorized selected-long oracle capacity retries."""
import argparse
import json
from pathlib import Path
import sys

from port7_verify_chain import ROOT, await_result, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, action='append', required=True)
    parser.add_argument('--authorized', required=True)
    args = parser.parse_args()
    gate = Path(__file__).with_name('port7_verify_gate.py')
    for directory in args.directory:
        plan = json.loads((directory / 'manifest.json').read_text())
        await_result(directory / (plan['arm'] + '-native-card-job.json'))
        original = directory / (plan['arm'] + '-oracle-card-job.json')
        if not original.exists():
            run([sys.executable, gate, 'replay', '--directory', directory])
        descriptor = directory / 'capacity-equivalence-replay-v1.json'
        if not descriptor.exists():
            run([sys.executable, gate, 'capacity', '--directory', directory,
                 '--authorized', args.authorized])
        job = Path(json.loads(descriptor.read_text())['job'])
        # Existing launched retries may be pending or complete. Never resubmit.
        if not job.with_suffix('.launcher.json').exists():
            run([sys.executable, ROOT / 'tools/m31/card_launch.py', '--unit',
                 'codex-m31-' + plan['arm'] + '-oracle-capacity-v1', job])
        await_result(job)
        run(['flock', '-s', '-w', '86400', '/mnt/ai/logs/m31-port/.build.lock',
             sys.executable, gate, 'compare', '--directory', directory,
             '--capacity-replay', descriptor])


if __name__ == '__main__':
    main()
