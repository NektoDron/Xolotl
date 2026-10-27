#!/usr/bin/env python3
"""Report new port6 speed/quality results and explicit acceptance criteria."""
import argparse
import json
import re
from pathlib import Path

import bench_m31
from port5_bench_report import stages, telemetry

BASELINE_TG = {'4K': 27.51, '16K': 26.385, '32K': 25.44}
LATEST_TG_MIN = 25.0
LATEST_GOALS_PROVENANCE = 'Explicit human coordinator steering for resumed port6: generation >=25 t/s at every context. This reporting override does not edit frozen manifest goals or measured timings.'



def fast_profiles(path):
    latest, previous, requests = {}, {}, []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line.startswith('M31 fast FFN profile '):
            latest = {key: float(value) for key, value in re.findall(r'(\w+)=(-?\d+(?:\.\d+)?)', line)}
        elif line.startswith('M31 request timings '):
            delta_keys = ('calls', 'experts', 'wall_ms', 'H2D_bytes', 'D2H_bytes')
            requests.append({'line': number, 'cumulative': dict(latest),
                             'delta': {key: latest[key] - previous.get(key, 0) for key in delta_keys if key in latest}})
            previous = dict(latest)
    return {'requests': requests, 'final': latest,
            'limits': 'Fast-FFN wall_ms includes enqueue/queue/kernel/synchronization/readback. It is not isolated GPU kernel time. Scratch_bytes is a capacity gauge.'}


def quality(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    expected = {('gsm8k', i) for i in range(100)} | {('tool', i) for i in range(20)} | {('tool_follow', i) for i in range(3)}
    keys = [(row['kind'], row['id']) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected or any(type(row['ok']) is not bool for row in rows):
        raise ValueError('quality must contain exactly GSM8K100 and tools23, unique IDs and boolean scores')
    gsm = sum(row['ok'] for row in rows if row['kind'] == 'gsm8k')
    tools = sum(row['ok'] for row in rows if row['kind'] in ('tool', 'tool_follow'))
    return {'source': str(path), 'complete': True, 'gsm8k': [gsm, 100], 'tools': [tools, 23],
            'pass': gsm >= 95 and tools >= 19,
            'length_stops': sum(row.get('finish') == 'length' for row in rows),
            'marker_leaks': sum(bool(row.get('leak')) for row in rows),
            'errors': [{'kind': row['kind'], 'id': row['id']} for row in rows if not row['ok']]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--quality', type=Path, help='same session original eval JSONL, if run')
    parser.add_argument('--gate', type=Path, help='final gate20 compare.json, if available')
    parser.add_argument('--long', type=Path, help='same-build selected16K/32K long compare.json')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.directory / 'manifest.json').read_text())
    if manifest.get('format') != 'm31-port6-http-plan-v1':
        raise ValueError('report requires a new port6 session, not historical results')
    sessions = list(args.directory.glob('*-http-session.json'))
    if len(sessions) != 1:
        raise ValueError('expected exactly one HTTP session')
    prefix = sessions[0].with_suffix('')
    session = json.loads(Path(str(prefix) + '.result.json').read_text())
    result = {'format': 'm31-port6-bench-report-v1', 'manifest': manifest, 'session_result': session,
              'profile': stages(Path(str(prefix) + '.server.log')),
              'fast_ffn_profile': fast_profiles(Path(str(prefix) + '.server.log')),
              'telemetry': telemetry(Path(str(prefix) + '.telemetry.jsonl')),
              'applied_goals': {'pp32k_min': 1000.0, 'tg_each_depth_min': LATEST_TG_MIN,
                                'provenance': LATEST_GOALS_PROVENANCE, 'frozen_manifest_goals': manifest.get('goals')}}
    if (args.directory / 'speed.tsv').exists():
        result['speed'] = bench_m31.summarize(argparse.Namespace(speed=[args.directory / 'speed.tsv'], quality=[]))
        arms = result['speed']['arms']
        result['speed_acceptance'] = {}
        for arm, values in arms.items():
            depths = values['depths']
            complete = values['complete_two_repetitions']
            pp = depths.get('32K', {}).get('pp_tps_median', 0) >= 1000
            tg = {depth: depths.get(depth, {}).get('tg_tps_median', 0) >= baseline
                  for depth, baseline in BASELINE_TG.items()}
            floor = {depth: depths.get(depth, {}).get('tg_tps_median', 0) >= LATEST_TG_MIN for depth in BASELINE_TG}
            result['speed_acceptance'][arm] = {'complete': complete, 'pp32k_ge1000': pp,
                                              'tg_at_least_port5_diagnostic': tg, 'tg_ge25_each_depth': floor,
                                              'pass': complete and pp and all(floor.values())}
    if args.quality:
        # A quality file from another job cannot silently masquerade as same-ELF evidence.
        if args.quality.resolve().parent != args.directory.resolve():
            raise ValueError('quality path must belong to the reported frozen session')
        result['quality'] = quality(args.quality)
    if args.gate:
        gate = json.loads(args.gate.read_text())
        if gate.get('scope') != 'port6-gpu-offload-gate20-complete-logits':
            raise ValueError('only complete new GPU-offload gate20 is final fast numeric evidence; CPU strict remains opt-in verification')
        gate_manifest = json.loads((args.gate.parent / 'manifest.json').read_text())
        keys = ('environment', 'unset_inherited_environment', 'threads', 'gpu_experts_gib')
        result['numeric_gate'] = {'source': str(args.gate), 'pass': gate['pass'],
                                  'scope': gate['scope'], 'rows': gate['rows'], 'top1': gate['top1'],
                                  'mean_kl': gate['mean_kl'], 'max_kl': gate['max_kl'],
                                  'matching_configuration': {key: gate_manifest[key] == manifest[key] for key in keys},
                                  'matching_build_receipt': gate_manifest.get('build_receipt', {}).get('sha256') ==
                                       manifest.get('build_receipt', {}).get('sha256') if gate_manifest.get('build_receipt') and manifest.get('build_receipt') else None,
                                  'geometry_note': 'Gate20 chunk2048/ctx4096 matches the GPU-offload PR oracle; selected long-context gates additionally replay native actual chunks.'}
    if args.long:
        long = json.loads(args.long.read_text())
        if long.get('format') != 'm31-port6-long-gate-v1' or sorted(c['tokens'] for c in long['cases']) != [16384, 32768]:
            raise ValueError('long evidence must cover both16K and32K')
        result['long_numeric_gate'] = {'source': str(args.long), 'pass': long['pass'],
                                      'cases': long['cases'], 'scope': long['scope'],
                                      'matching_environment': long['environment'] == manifest['environment'],
                                      'matching_unset_environment': long['unset_inherited_environment'] == manifest['unset_inherited_environment'],
                                      'matching_build_receipt': long.get('build_receipt', {}).get('sha256') == manifest.get('build_receipt', {}).get('sha256') if long.get('build_receipt') and manifest.get('build_receipt') else None}
    result['limits'] = ['Two repetitions do not establish statistical significance.',
                        'F16 native KV differs from llama.cpp reference Q8_0.',
                        'Generation acceptance floor is >=25 t/s at every depth under latest explicit steering; port5 ratios remain diagnostic comparison.',
                        'No overall acceptance is inferred from missing numeric, speed, quality or source evidence.']
    with args.out.open('x') as output:
        json.dump(result, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps({'output': str(args.out), 'speed_acceptance': result.get('speed_acceptance'),
                      'quality': result.get('quality'), 'numeric_gate': result.get('numeric_gate')}, indent=2))


if __name__ == '__main__':
    main()
