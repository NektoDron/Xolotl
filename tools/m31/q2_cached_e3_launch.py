#!/usr/bin/env python3
"""Guard one prepared cached-E3 diagnostic after BOTH baseline quality cleanups.

Without --execute this only checks small frozen inputs and reports readiness.
Only --execute, after both baseline123-row results and cleanup, hashes the ELF
under the exclusive build lock and submits the single aictl/card-lock controller.
No automatic long/performance/quality retries are ever submitted.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess


def sha_small(path):
    if path.stat().st_size > 32 << 20:
        raise ValueError('small-file hash bound: ' + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def baseline_ready(spec):
    d = Path(spec['directory'])
    stem = d.name
    checks = {}
    for label, file in [('card', stem + '-card-job.result.json'),
                        ('http', stem + '-http-session.result.json')]:
        path = d / file
        if not path.is_file():
            checks[label] = False
            continue
        result = json.loads(path.read_text())
        checks[label] = result.get('returncode') == 0
        if label == 'http':
            checks[label] &= result.get('ready') is True and result.get('server_returncode') == 0
            checks[label] &= bool(result.get('clients')) and all(c.get('returncode') == 0 for c in result['clients'])
    rows_path = d / spec['eval']
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()] if rows_path.is_file() else []
    actual = [(row.get('kind'), row.get('id')) for row in rows]
    expected = {('gsm8k', i) for i in range(100)} | {('tool', i) for i in range(20)} | {('tool_follow', i) for i in range(3)}
    checks['complete123unique_ids'] = (len(actual) == 123 and len(set(actual)) == 123 and set(actual) == expected
                                      and all(type(row.get('id')) is int and type(row.get('kind')) is str for row in rows))
    proc = subprocess.run(['systemctl', 'is-active', spec['unit']], text=True, capture_output=True)
    checks['controller_inactive'] = proc.returncode != 0 and proc.stdout.strip() in ('inactive', 'failed', 'unknown')
    return dict(directory=str(d), checks=checks, ready=all(checks.values()), scores_not_gated=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    d = a.directory.resolve(strict=True)
    mpath = d / 'manifest.json'
    if sha_small(mpath) != (d / 'manifest-seal.sha256').read_text().strip():
        raise ValueError('prepared manifest seal differs')
    m = json.loads(mpath.read_text())
    if m['maximum_native_attempts'] != 1 or m['no_followup_jobs'] is not True:
        raise ValueError('bounded single-attempt policy required')
    for name, fp in m['frozen_files'].items():
        path = d / name
        if path.stat().st_size != fp['bytes'] or sha_small(path) != fp['sha256'] or path.stat().st_mode & 0o222:
            raise ValueError('frozen small input changed: ' + name)
    elf = Path(m['elf']['path'])
    st = elf.stat()
    if st.st_size != m['elf']['bytes'] or st.st_mtime_ns != m['elf']['mtime_ns'] or st.st_mode & 0o222:
        raise ValueError('existing frozen ELF metadata/mode changed')
    baseline = [baseline_ready(spec) for spec in m['baseline_qualities']]
    worker = subprocess.run(['systemctl', 'is-active', 'codex-m31-port-card-worker'], capture_output=True, text=True)
    worker_inactive = worker.returncode != 0 and worker.stdout.strip() in ('inactive', 'failed', 'unknown')
    ready = all(row['ready'] for row in baseline) and worker_inactive
    report = dict(prepared_small_inputs_unchanged=True, baseline_qualities=baseline,
                  own_card_worker_inactive=worker_inactive, ready_for_one_trial=ready,
                  existing_large_ELF_hash_read=False, execute=a.execute)
    if not a.execute:
        print(json.dumps(report, indent=2))
        return
    if not ready:
        raise RuntimeError('both baseline quality123 rows/RC0/controller cleanup and own worker cleanup required')
    marker = d / 'ONE_ATTEMPT_STARTED.json'
    if marker.exists() or Path(m['job']).with_suffix('.result.json').exists():
        raise RuntimeError('one attempt already submitted or completed; no retry/fishing allowed')
    with Path('/mnt/ai/logs/m31-port/.build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        digest = hashlib.sha256()
        with elf.open('rb') as stream:
            for block in iter(lambda: stream.read(8 << 20), b''):
                digest.update(block)
        if digest.hexdigest() != m['elf']['sha256']:
            raise ValueError('frozen ELF SHA differs before single trial')
    report['existing_large_ELF_hash_read'] = True
    report['ELF_sha256'] = digest.hexdigest()
    report['submitted_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    job = Path(m['job'])
    unit = 'codex-m31-q2-cached-e3-code128-v1'
    command = ['sudo', 'systemd-run', '--unit=' + unit, '-p', 'User=nekto', '-p', 'KillMode=mixed',
               '-p', 'TimeoutStopSec=600', '/bin/bash', str(job.with_suffix('.runner.sh')), str(job)]
    report['submit_command'] = command
    with marker.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    subprocess.run(command, check=True)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
