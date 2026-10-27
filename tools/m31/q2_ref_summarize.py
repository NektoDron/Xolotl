#!/usr/bin/env python3
"""Audit a completed, separately preserved Q2 reference speed or quality job."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    manifest = json.loads((directory / 'manifest.json').read_text())
    job = directory / (directory.name + '-card-job.result.json')
    session = directory / (directory.name + '-http-session.result.json')
    job_result = json.loads(job.read_text())
    session_result = json.loads(session.read_text())
    checks = {'card_job_rc0': job_result.get('returncode') == 0,
              'http_session_rc0': session_result.get('returncode') == 0,
              'all_clients_rc0': bool(session_result.get('clients')) and
                  all(client.get('returncode') == 0 for client in session_result['clients'])}
    hashes = {}
    for name, expected in manifest['files'].items():
        path = Path(name)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[name] = {'expected': expected, 'actual': actual, 'equal': actual == expected}
    # The preparer can evolve after a job was written; deployed executable,
    # runner, original benchmark, original eval and data must remain frozen.
    checks['runtime_client_data_hashes_stable'] = all(v['equal'] for k, v in hashes.items()
        if not k.endswith('/q2_ref_prepare.py'))
    result = {'directory': str(directory), 'mode': manifest['mode'],
              'model': manifest['model'], 'llama_sha': manifest['llama_sha'],
              'expert_layers': manifest['expert_layers'], 'checks': checks, 'hash_audit': hashes}
    if manifest['mode'] == 'speed':
        with (directory / 'speed.tsv').open() as stream:
            rows = list(csv.DictReader(stream, delimiter='\t'))
        checks['all_three_depths'] = [row['depth'] for row in rows] == ['4K', '16K', '32K']
        checks['all_256_outputs'] = all(int(row['gen_n']) == 256 for row in rows)
        checks['positive_speeds'] = all(float(row['prefill_tps']) > 0 and float(row['gen_tps']) > 0 for row in rows)
        result['speed'] = rows
    else:
        paths = list((directory / 'eval').glob('*.jsonl'))
        checks['one_eval_file'] = len(paths) == 1
        if len(paths) != 1:
            raise RuntimeError('expected exactly one eval file')
        rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
        gsm = [row for row in rows if row['kind'] == 'gsm8k']
        tool = [row for row in rows if row['kind'] == 'tool']
        follow = [row for row in rows if row['kind'] == 'tool_follow']
        checks['gsm_ids_0_99'] = [row['id'] for row in gsm] == list(range(100))
        checks['tool_ids_0_19'] = [row['id'] for row in tool] == list(range(20))
        checks['follow_ids_0_2'] = [row['id'] for row in follow] == list(range(3))
        checks['total123'] = len(rows) == 123
        result['quality'] = {'gsm_correct': sum(row['ok'] for row in gsm), 'gsm_total': len(gsm),
                             'tools_correct': sum(row['ok'] for row in tool + follow), 'tools_total': len(tool + follow),
                             'gsm_wrong_ids': [row['id'] for row in gsm if not row['ok']],
                             'tool_wrong_ids': [row['id'] for row in tool if not row['ok']],
                             'follow_wrong_ids': [row['id'] for row in follow if not row['ok']],
                             'length_finishes': sum(row.get('finish') == 'length' for row in rows),
                             'leaks': sum(bool(row.get('leak')) for row in rows),
                             'total_seconds': sum(row['s'] for row in rows),
                             'completion_tokens': sum(row.get('tokens') or 0 for row in rows),
                             'sampling': {'temperature': 1.0, 'top_p': .95, 'max_tokens': 6000}}
    telemetry = directory / (directory.name + '-http-session.telemetry.jsonl')
    gpu_rows = [json.loads(line) for line in telemetry.read_text().splitlines()]
    used = [float(row['gpu0_used_mib']) for row in gpu_rows if row.get('gpu0_used_mib')]
    result['gpu0_peak_used_mib_observed'] = max(used) if used else None
    result['audit_pass'] = all(checks.values())
    target = directory / 'summary.json'
    with target.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['audit_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
