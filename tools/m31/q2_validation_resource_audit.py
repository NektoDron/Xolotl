#!/usr/bin/env python3
"""Read-only end-of-wave resource audit; never stop a model, miner or watcher."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.error
import urllib.request


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def run(argv):
    p = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    return {'argv': argv, 'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--phase', choices=('pre-restore', 'post-restore'), required=True)
    p.add_argument('--self-unit', required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    base = Path('/mnt/ai/logs/m31-port/q2-wave')
    before = json.loads((base / 'config-before.json').read_text())
    current_raw = Path('/etc/aictl/config.json').read_bytes()
    current = json.loads(current_raw)
    (args.out / 'config-current.json').write_bytes(current_raw)
    protected = {name: current.get('models', {}).get(name) == value
                 for name, value in before['models'].items()}
    top = {key: current.get(key) == value for key, value in before.items() if key != 'models'}
    own = ('m31-xolotl-check', 'm31-xolotl')
    result = {'format': 'm31-q2-resource-audit-v1', 'phase': args.phase,
              'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'protected_models_equal': protected, 'top_level_keys_equal': top,
              'own_temporary_presets_present': [key for key in own if key in current['models']],
              'other_new_models': sorted(set(current['models']) - set(before['models']) - set(own)),
              'raw_config_sha256': hashlib.sha256(current_raw).hexdigest(),
              'miner_hold_present': Path('/run/aictl.miner-hold').exists(),
              'v8': {}, 'own_model_processes': []}
    baseline = json.loads((base / 'baseline.json').read_text())
    for name, fp in baseline['v8_files_corrected'].items():
        actual = sha(Path(name))
        result['v8'][name] = {'expected': fp['sha256'], 'actual': actual, 'equal': actual == fp['sha256']}
    for item in Path('/proc').iterdir():
        if not item.name.isdecimal():
            continue
        try:
            exe = os.readlink(item / 'exe')
            argv = (item / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if Path(exe).name in ('m31_serve', 'm31_generate', 'm31_oracle', 'm31_serve-frozen',
                               'm31_generate-frozen', 'llama-server-frozen') and \
                ('/mnt/ai/logs/m31-port/' in argv or '/mnt/ai/Xolotl-m31/build-' in argv):
            result['own_model_processes'].append({'pid': int(item.name), 'exe': exe, 'argv': argv})
    commands = {
        'state': ['cat', '/var/lib/aictl/state.json'],
        'aictl': ['sudo', 'aictl', 'status'],
        'gpu': ['nvidia-smi', '--query-gpu=index,name,memory.used', '--format=csv,noheader'],
        'gpu0_apps': ['nvidia-smi', '-i', '0', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader'],
        'miner0': ['systemctl', 'show', 'keryx-miner.service', '--property=ActiveState', '--property=MainPID'],
        'miner1': ['systemctl', 'is-active', 'pearl-miner.service'],
        'duty': ['systemctl', 'is-active', 'llm-chat.service'],
        'worker': ['systemctl', 'is-active', 'codex-m31-port-card-worker.service'],
        'monitor': ['systemctl', 'is-active', 'codex-m31-q2-gpu-isolation-monitor-v1.service'],
        'watch': ['systemctl', 'show', 'aictl-hold-watch.service', '--property=ActiveState', '--property=MainPID'],
        'active_q2_units': ['systemctl', 'list-units', '--state=running', '--no-legend', '--plain', 'codex-m31-q2*'],
        'ports': ['ss', '-H', '-ltnp', '( sport = :8179 or sport = :8180 or sport = :8181 or sport = :8182 or sport = :8183 )'],
        'card_lock': ['flock', '-n', '/mnt/ai/logs/.xolotl-card.lock', 'true'],
    }
    for key, argv in commands.items():
        result[key] = run(argv)
    watch = dict(line.split('=', 1) for line in result['watch']['stdout'].splitlines() if '=' in line)
    pid = int(watch.get('MainPID', '0'))
    try:
        result['watch_process_state'] = next(line for line in Path(f'/proc/{pid}/status').read_text().splitlines() if line.startswith('State:'))
    except (FileNotFoundError, ProcessLookupError, StopIteration):
        result['watch_process_state'] = 'not running'
    try:
        with urllib.request.urlopen('http://127.0.0.1:8084/health', timeout=3) as response:
            health = response.status
    except urllib.error.HTTPError as error:
        health = error.code
    except OSError:
        health = None
    result['duty_health_status'] = health
    allowed = {'codex-m31-q2.service', args.self_unit.removesuffix('.service') + '.service'}
    if args.phase == 'pre-restore':
        allowed.add('codex-m31-q2-gpu-isolation-monitor-v1.service')
    units = [line.split()[0] for line in result['active_q2_units']['stdout'].splitlines() if line.strip()]
    result['unexpected_active_q2_units'] = [unit for unit in units if unit not in allowed]
    state = json.loads(result['state']['stdout'])
    stopped = 'T (' in result['watch_process_state']
    checks = {'protected_presets_equal': all(protected.values()), 'top_level_equal': all(top.values()),
              'own_presets_rolled_back': not result['own_temporary_presets_present'],
              'protected_v8_unchanged': all(value['equal'] for value in result['v8'].values()),
              'no_own_model_pid': not result['own_model_processes'],
              'no_active_own_model_state': state.get('active') not in own,
              'worker_inactive': result['worker']['stdout'].strip() != 'active',
              'no_own_experiment_units': not result['unexpected_active_q2_units'],
              'ports_empty': not result['ports']['stdout'].strip(),
              'miner1_active': result['miner1']['stdout'].strip() == 'active',
              'duty_active': result['duty']['stdout'].strip() == 'active' and health in (200, 401, 403),
              'miner_hold_retained': result['miner_hold_present'],
              'card_lock_free': result['card_lock']['returncode'] == 0,
              'watch_expected_state': stopped if args.phase == 'pre-restore' else not stopped,
              'monitor_expected_state': (result['monitor']['stdout'].strip() == 'active') == (args.phase == 'pre-restore')}
    result['checks'] = checks
    result['pass'] = all(checks.values())
    result['scope'] = 'Read-only snapshot. Foreign models/services are only observed. Watch/monitor restoration is ROOT-owned.'
    (args.out / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'out': str(args.out), 'pass': result['pass'], 'checks': checks}, indent=2))
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
