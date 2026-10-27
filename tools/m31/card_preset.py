#!/usr/bin/env python3
"""Install/rollback exactly one owned temporary preset, preserving every other key."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys

config = Path('/etc/aictl/config.json')
backup = Path(sys.argv[2])
name = (json.loads(Path(sys.argv[3]).read_text()).get('preset', 'm31-xolotl-check')
        if sys.argv[1] == 'put' else (sys.argv[3] if len(sys.argv) > 3 else 'm31-xolotl-check'))
if name not in ('m31-xolotl-check', 'm31-xolotl'):
    raise ValueError('preset must be one of the two owned temporary M31 keys')
unit = 'codex-m31-port-card-worker'
with open('/run/aictl.lock', 'a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    current = json.loads(config.read_text())
    if sys.argv[1] == 'put':
        if name in current['models']:
            raise RuntimeError('owned preset already present; resolve previous session first')
        if backup.exists():
            raise RuntimeError('use a fresh backup path')
        shutil.copy2(config, backup)
        job = str(Path(sys.argv[3]).resolve())
        worker = Path(job).with_suffix('.worker.py')
        if not worker.is_file():
            raise RuntimeError('missing frozen worker: ' + str(worker))
        current['models'][name] = {
            'desc': 'временный M31 experimental batch/server, только GPU0', 'gpu': '0', 'port': 8179,
            'start': ['systemd-run', '--collect', '--unit=' + unit, '-p', 'User=nekto',
                      # This unit owns every model subprocess, including nested
                      # process groups. The controller uses mixed to protect its
                      # aictl up child; the worker must terminate all readers.
                      '-p', 'LimitCORE=0', '-p', 'KillMode=control-group', '-p', 'TimeoutStopSec=600',
                      '-p', 'StandardOutput=append:/mnt/ai/logs/m31-port/card-worker.log',
                      '-p', 'StandardError=append:/mnt/ai/logs/m31-port/card-worker.log',
                      '/usr/bin/python3', str(worker), job],
            'stop': ['systemctl', 'stop', unit],
            'check': ['systemctl', 'is-active', '--quiet', unit],
            'ready_url': 'http://127.0.0.1:8179/health',
        }
    elif sys.argv[1] == 'rollback':
        previous = json.loads(backup.read_text())['models']
        if name in previous:
            current['models'][name] = previous[name]
        else:
            current['models'].pop(name, None)
    else:
        raise ValueError(sys.argv[1])
    temporary = config.with_suffix('.json.tmp-m31-port')
    temporary.write_text(json.dumps(current, ensure_ascii=False, indent=2) + '\n')
    os.chmod(temporary, config.stat().st_mode & 0o777)
    os.replace(temporary, config)
print('Бэкап:', backup)
print('Откат: sudo python3 ' + str(Path(__file__).resolve()) + ' rollback ' + str(backup) + ' ' + name)
