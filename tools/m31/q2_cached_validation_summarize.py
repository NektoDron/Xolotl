#!/usr/bin/env python3
"""Audit one cached-E3/6656 HTTP job; never reuse baseline quality."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    args = p.parse_args()
    directory = args.directory.resolve()
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('format') != 'm31-q2-http-job-v1':
        raise ValueError('expected Q2 canonical frozen HTTP manifest')
    if manifest.get('configuration') != 'cached-e3-6656':
        raise ValueError('requires a new cached-E3/6656 session, never baseline')
    prefix = directory / (directory.name + '-http-session')
    session = json.loads(prefix.with_suffix('.result.json').read_text())
    job = json.loads((directory / (directory.name + '-card-job.result.json')).read_text())
    checks = {'card_job_rc0': job.get('returncode') == 0,
              'http_session_rc0': session.get('returncode') == 0,
              'native_server_rc0': session.get('server_returncode') == 0,
              'all_clients_rc0': bool(session.get('clients')) and
                  all(client.get('returncode') == 0 for client in session['clients'])}
    hashes = {}
    expected = {name: fp['sha256'] for name, fp in manifest['frozen'].items()}
    expected.update(manifest['external_sha256'])
    expected[manifest['profile']] = manifest['profile_sha256']
    for name, digest in expected.items():
        actual = sha(Path(name))
        hashes[name] = {'expected': digest, 'actual': actual, 'equal': actual == digest}
    checks['runtime_client_data_profile_hashes_stable'] = all(item['equal'] for item in hashes.values())
    profile = json.loads(Path(manifest['profile']).read_text())
    env = profile['environment']
    cached = env.get('M31_GPU_CACHED_DECODE')
    checks['final_q2_fast_environment'] = env.get('M31_CPU_STOCK_EXPERTS') == '1' and \
        env.get('M31_CPU_FUSED') == '1' and cached == '1'
    if cached == '1':
        # The sole approved cached E3 final family, separate from baseline.
        engine = dict(zip(profile['engine_arguments'][::2], profile['engine_arguments'][1::2]))
        checks['cached_e3_final_environment'] = all(env.get(key) == value for key, value in {
            'M31_GPU_CACHED_DECODE_NATIVE': '1', 'M31_GPU_CACHED_DECODE_EXACT': '0',
            'M31_GPU_DECODE_SEED_HOT': '384'}.items()) and \
            int(engine['--chunk']) == 6656 and float(engine['--gpu-experts-gib']) == 9.88
    result = {'format': 'm31-q2-http-summary-v1', 'directory': str(directory),
              'mode': manifest['mode'], 'model': manifest['model'], 'canonical': manifest['canonical'],
              'cached_decode': cached == '1',
              'checks': checks, 'hash_audit': hashes, 'runtime_profile_sha256': manifest['profile_sha256'],
              'serve_elf_sha256': manifest['frozen'][str(directory / 'm31_serve-frozen')]['sha256'],
              'adapter_sha256': manifest['frozen'][str(directory / 'serve_m31.py')]['sha256'],
              'scope': 'One completed session; numerical/quality/speed evidence remain separately scoped.'}
    if manifest['mode'] == 'speed':
        with (directory / 'speed.tsv').open() as stream:
            rows = list(csv.DictReader(stream, delimiter='\t'))
        checks['all_three_depths'] = [row['depth'] for row in rows] == ['4K', '16K', '32K']
        checks['all_256_outputs'] = all(int(row['gen_n']) == 256 for row in rows)
        checks['positive_speeds'] = all(float(row['prefill_tps']) > 0 and float(row['gen_tps']) > 0 for row in rows)
        result['speed'] = rows
        result['speed_goals'] = {'pp32k_ge1000': any(row['depth'] == '32K' and float(row['prefill_tps']) >= 1000 for row in rows),
                                 'tg_each_ge25': {row['depth']: float(row['gen_tps']) >= 25 for row in rows}}
    else:
        files = list((directory / 'eval').glob('*.jsonl'))
        if len(files) != 1:
            raise ValueError('exactly one original eval JSONL required')
        rows = [json.loads(line) for line in files[0].read_text().splitlines()]
        gsm = [row for row in rows if row['kind'] == 'gsm8k']
        tool = [row for row in rows if row['kind'] == 'tool']
        follow = [row for row in rows if row['kind'] == 'tool_follow']
        checks.update(gsm_ids_0_99=[row['id'] for row in gsm] == list(range(100)),
                      tool_ids_0_19=[row['id'] for row in tool] == list(range(20)),
                      follow_ids_0_2=[row['id'] for row in follow] == list(range(3)),
                      total123=len(rows) == 123, boolean_scores=all(type(row['ok']) is bool for row in rows))
        result['quality'] = {'source': str(files[0]), 'gsm_correct': sum(row['ok'] for row in gsm),
                             'gsm_total': len(gsm), 'tools_correct': sum(row['ok'] for row in tool + follow),
                             'tools_total': len(tool + follow),
                             'wrong': [{'kind': row['kind'], 'id': row['id']} for row in rows if not row['ok']],
                             'length_finishes': sum(row.get('finish') == 'length' for row in rows),
                             'leaks': sum(bool(row.get('leak')) for row in rows),
                             'total_seconds': sum(row['s'] for row in rows),
                             'completion_tokens': sum(row.get('tokens') or 0 for row in rows),
                             'sampling': {'temperature': 1.0, 'top_p': .95, 'max_tokens': 6000},
                             'single_stochastic_run': True}
        result['quality']['goals_met'] = result['quality']['gsm_correct'] >= 95 and result['quality']['tools_correct'] >= 19
    telemetry = Path(str(prefix) + '.telemetry.jsonl')
    observed = [json.loads(line) for line in telemetry.read_text().splitlines()]
    memory = [float(row['gpu0_used_mib']) for row in observed if row.get('gpu0_used_mib')]
    result['gpu0_peak_used_mib_observed'] = max(memory) if memory else None
    result['audit_pass'] = all(checks.values())
    with (directory / 'summary.json').open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['audit_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
