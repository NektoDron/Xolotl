#!/usr/bin/env python3
"""Run independent prepared numerical components, preserving every result on failure."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    if os.environ.get('M31_CARD_SESSION') != '1':
        raise RuntimeError('requires the authorized aictl worker')
    spec_path = Path(sys.argv[1]).resolve()
    spec = json.loads(spec_path.read_text())
    results = []
    for index, command in enumerate(spec['commands']):
        log = spec_path.parent / f'component-{index:02d}.log'
        started = time.monotonic()
        with log.open('x') as output:
            try:
                code = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                                      timeout=spec.get('command_timeout', 300), check=False).returncode
            except subprocess.TimeoutExpired:
                code = 124
        row = {'index': index, 'argv': command, 'log': str(log), 'returncode': code,
               'wall_seconds_diagnostic': time.monotonic() - started}
        with (spec_path.parent / f'component-{index:02d}.result.json').open('x') as output:
            json.dump(row, output, indent=2)
            output.write('\n')
        results.append(row)
    result = {'format': 'm31-port7-component-batch-result-v1', 'commands': results,
              'all_commands_rc0': all(row['returncode'] == 0 for row in results),
              'scope': 'Process results only. Inspect each mathematical component result; zero exit alone is not universal model quality acceptance.'}
    with spec_path.with_suffix('.result.json').open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    return 0 if result['all_commands_rc0'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
