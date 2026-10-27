#!/usr/bin/env python3
"""Join immutable final V8 evidence; pending quality can never become acceptance."""
import argparse
import json
from pathlib import Path

from port6_bench_report import quality


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.directory.resolve()
    speed = read(root / 'speed/report.json')
    speed_manifest = speed['manifest']
    manifests = {'speed': speed_manifest, 'quality': read(root / 'quality/manifest.json'),
                 'gate20': read(root / 'gate20/manifest.json'),
                 'long': read(root / 'oracle-capacity-v3/long/manifest.json'),
                 'direct': read(root / 'oracle-capacity-v3/direct/manifest.json')}
    reference = speed_manifest['build_receipt']['sha256']
    checks = {
        'same_build_receipt': all(m.get('build_receipt', {}).get('sha256') == reference for m in manifests.values()),
        'same_environment': all(m['environment'] == speed_manifest['environment'] for m in manifests.values()),
        'same_native_unsets': all(m['unset_inherited_environment'] == speed_manifest['unset_inherited_environment'] for m in manifests.values()),
        'same_final_serve_elf': manifests['quality']['engine']['sha256'] == speed_manifest['engine']['sha256'],
        'same_final_adapter': manifests['quality']['adapter']['sha256'] == speed_manifest['adapter']['sha256'],
        'same_final_runner': manifests['quality']['session_runner']['sha256'] == speed_manifest['session_runner']['sha256'],
        'final_profile_is_fast_nofusion': all(speed_manifest['environment'].get(key) == value for key, value in {
            'GGML_CUDA_DISABLE_FUSION': '1', 'M31_GPU_EXACT': '0', 'M31_GPU_FAST_FFN': '0',
            'M31_GPU_CPU_QUANTS': '0', 'M31_GPU_PREFILL_CHAIN': '1',
            'M31_GPU_REGISTER_ARENA': '1', 'M31_GPU_NATIVE_GU_CONCAT': '0',
            'M31_GPU_LAYER_BANKS': '0', 'M31_GPU_CAUSAL_FILL': '0'}.items())}
    numeric = {}
    for label, path in [('gate20', root / 'gate20/compare.json'),
                        ('long', root / 'oracle-capacity-v3/long/compare.json'),
                        ('direct', root / 'oracle-capacity-v3/direct/compare.json')]:
        value = read(path)
        summary = {'path': str(path), 'pass': value['pass'], 'scope': value['scope'],
                   'cases': [{k: c[k] for k in ('case', 'tokens', 'rows', 'top1', 'mean_kl', 'max_kl', 'max_abs')}
                             for c in value['cases']]}
        if label == 'gate20':
            checks['full_gate20_complete'] = value.get('acceptance_eligible') is True and value['scope'] == 'port6-gpu-offload-gate20-complete-logits' and len(value['cases']) == 20 and value['rows'] == 28723
        else:
            expected = [16384, 32768] if label == 'long' else [10272]
            checks[label + '_coverage'] = sorted(c['tokens'] for c in value['cases']) == expected and all(c['rows'] == 256 and c['capacity_equivalence']['authorized'] for c in value['cases'])
        checks[label + '_numeric'] = value['pass'] and all(c['top1'] >= .99 and c['mean_kl'] <= .001 and c['max_kl'] <= .05 for c in value['cases'])
        numeric[label] = summary
    speed_checks = speed['speed_acceptance']
    checks['speed_two_repetitions_and_goals'] = len(speed_checks) == 1 and all(v['complete'] and v['pp32k_ge1000'] and all(v['tg_ge25_each_depth'].values()) and v['pass'] for v in speed_checks.values())
    checks['speed_session_completed'] = speed['session_result']['returncode'] == 0 and speed['session_result'].get('server_returncode') == 0
    sessions = list((root / 'quality').glob('*-http-session.json'))
    if len(sessions) != 1:
        raise ValueError('expected exactly one frozen quality HTTP session')
    session = read(sessions[0])
    clients = session['clients']
    if len(clients) != 1 or clients[0]['argv'][-1] != '100':
        raise ValueError('quality must use original full100 client')
    client = clients[0]
    qfile = root / 'quality' / (client['argv'][2] + '.jsonl')
    qresult = sessions[0].with_suffix('.result.json')
    q = {'status': 'pending', 'path': str(qfile), 'pass': False}
    if qresult.exists():
        result = read(qresult)
        q = quality(qfile)
        q['session_returncode'] = result['returncode']
        q['server_returncode'] = result.get('server_returncode')
        q['status'] = 'complete'
        q['pass'] = q['pass'] and result['returncode'] == 0 and result.get('server_returncode') == 0
    else:
        q['flushed_rows'] = sum(1 for line in qfile.open() if line.strip()) if qfile.exists() else 0
    checks['quality_complete_and_goals'] = q['status'] == 'complete' and q['pass']
    report = {'format': 'm31-port6-final-acceptance-v1', 'pass': all(checks.values()),
              'checks': checks, 'numeric': numeric, 'quality': q,
              'speed': speed['speed']['arms'], 'speed_raw': speed['speed']['raw_speed_rows'],
              'individual_tg_below25': [r for r in speed['speed']['raw_speed_rows'] if r['gen_tps'] < 25],
              'generation_policy': 'Original two-repetition medians at each context; all individual rows retained, including below-floor samples.',
              'build_receipt': speed_manifest['build_receipt'],
              'serve_elf': speed_manifest['engine'], 'adapter': speed_manifest['adapter'],
              'gpu_peak_mib': speed['telemetry']['gpu0_peak_mib'],
              'capacity_equivalence': 'Explicitly authorized only for selected long/direct oracle gates; nativectx40000, oraclephysicalctx=case length, headers preserved; runtime contiguousKV/RoPE trace required.',
              'limits': ['A stochastic GSM/tools run is not a statistical significance claim.',
                         'Partial quality never counts as acceptance.',
                         'CPU strict128 is a separate opt-in diagnostic, not the fast oracle.']}
    with args.out.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'output': str(args.out), 'pass': report['pass'], 'checks': checks, 'quality': q}, indent=2))


if __name__ == '__main__':
    main()
