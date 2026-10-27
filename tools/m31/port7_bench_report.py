#!/usr/bin/env python3
"""Report port7 same-session speed and host stage intervals without kernel-time claims."""
import argparse
import json
from pathlib import Path

import bench_m31
from port5_bench_report import stages, telemetry
from port6_bench_report import DiagnosticLog, quality
from port5_bench_prepare import fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--quality', type=Path)
    parser.add_argument('--gate', type=Path, help='same-build complete GPU-offload gate20 compare.json')
    parser.add_argument('--long', type=Path, help='same-build selected16K/32K compare.json')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.directory / 'manifest.json').read_text())
    if manifest.get('format') != 'm31-port7-http-plan-v1':
        raise ValueError('requires a new port7 manifest')
    sessions = list(args.directory.glob('*-http-session.json'))
    if len(sessions) != 1:
        raise ValueError('expected one frozen HTTP session')
    prefix = sessions[0].with_suffix('')
    session = json.loads(Path(str(prefix) + '.result.json').read_text())
    profile = stages(DiagnosticLog(Path(str(prefix) + '.server.log')))
    result = {'format': 'm31-port7-bench-report-v1', 'manifest': manifest,
              'session_result': session, 'profile': profile,
              'telemetry': telemetry(Path(str(prefix) + '.telemetry.jsonl')),
              'decode_summary': [],
              'analysis_sources': [fingerprint(Path(__file__).with_name(name)) for name in (
                    'port7_bench_report.py', 'port6_bench_report.py', 'port5_bench_report.py',
                    'bench_m31.py', 'port5_bench_prepare.py')]}
    for request in profile['requests']:
        decode = request['stages'].get('decode', {})
        n = request['decode_forward_count']
        summary = {'line': request['line'], 'timings': request['timings'], 'decode_forward_count': n,
                   'per_forward': {category: values['mean_per_call'] for category, values in decode.items()}}
        result['decode_summary'].append(summary)
    if (args.directory / 'speed.tsv').exists():
        result['speed'] = bench_m31.summarize(argparse.Namespace(speed=[args.directory / 'speed.tsv'], quality=[]))
        result['speed_acceptance'] = {}
        for arm, value in result['speed']['arms'].items():
            depths = value['depths']
            comparisons = {}
            for depth in ('4K', '16K', '32K'):
                current = depths.get(depth, {})
                pp = current.get('pp_tps_median', 0)
                tg = current.get('tg_tps_median', 0)
                baseline = manifest['v8_comparison']
                comparisons[depth] = {'pp_v8_ratio': pp / baseline['pp_median'][depth],
                    'tg_v8_ratio': tg / baseline['tg_median'][depth],
                    'pp_ge_v8': pp >= baseline['pp_median'][depth], 'tg_ge35': tg >= 35,
                    'tg_ge40': tg >= 40}
            complete = value['complete_two_repetitions']
            result['speed_acceptance'][arm] = {'complete_two_repetitions': complete,
                'depths': comparisons, 'pass': complete and
                  all(row['pp_ge_v8'] and row['tg_ge35'] for row in comparisons.values())}
    tiny = args.directory / 'tinydecode.jsonl'
    if tiny.exists():
        rows = [json.loads(line) for line in tiny.read_text().splitlines()]
        result['tiny_decode'] = {'path': str(tiny), 'repetitions': len(rows),
            'message_hashes': [row['message_sha256'] for row in rows],
            'identical_messages': len({row['message_sha256'] for row in rows}) == 1,
            'scope': 'Fixed greedy message hash, not raw token IDs or full-logit evidence.'}
    if args.quality:
        if args.quality.resolve().parent != args.directory.resolve():
            raise ValueError('quality must be in the same frozen session')
        result['quality'] = quality(args.quality)
    if args.gate:
        gate = json.loads(args.gate.read_text())
        gate_manifest = json.loads((args.gate.parent / 'manifest.json').read_text())
        if gate.get('scope') != 'port6-gpu-offload-gate20-complete-logits':
            raise ValueError('only complete GPU-offload gate20 is final numeric evidence')
        matches = {key: gate_manifest.get(key) == manifest.get(key)
                   for key in ('environment', 'threads', 'gpu_experts_gib')}
        matches['build_receipt'] = bool(manifest.get('build_receipt')) and (
            gate_manifest.get('build_receipt', {}).get('sha256') == manifest['build_receipt']['sha256'])
        result['numeric_gate'] = {'source': str(args.gate), 'pass': gate['pass'],
             'top1': gate['top1'], 'mean_kl': gate['mean_kl'], 'max_kl': gate['max_kl'],
             'matching_configuration': matches,
             'scope': gate['scope'], 'rows': gate['rows']}
    if args.long:
        long = json.loads(args.long.read_text())
        if long.get('format') != 'm31-port6-long-gate-v1' or sorted(case['tokens'] for case in long['cases']) != [16384, 32768]:
            raise ValueError('long evidence must include selected16K and32K')
        matches = {'environment': long.get('environment') == manifest['environment'],
                   'build_receipt': bool(manifest.get('build_receipt')) and
                    long.get('build_receipt', {}).get('sha256') == manifest['build_receipt']['sha256']}
        result['long_numeric_gate'] = {'source': str(args.long), 'pass': long['pass'],
              'cases': long['cases'], 'matching_configuration': matches}
    checks = {'session': session.get('returncode') == 0 and session.get('server_returncode') == 0,
              'speed': bool(result.get('speed_acceptance')) and
                   all(arm['pass'] for arm in result.get('speed_acceptance', {}).values()),
              'quality': result.get('quality', {}).get('pass') is True,
              'gate20': result.get('numeric_gate', {}).get('pass') is True and
                   all(result.get('numeric_gate', {}).get('matching_configuration', {}).values()),
              'long': result.get('long_numeric_gate', {}).get('pass') is True and
                   all(result.get('long_numeric_gate', {}).get('matching_configuration', {}).values())}
    result['overall_acceptance'] = {'checks': checks, 'pass': all(checks.values()),
        'scope': 'Serial target engine only. Speculative decoding additionally requires raw greedy token equality and rollback evidence.'}
    result['limits'] = ['No overall acceptance is inferred; exact same-build gate20 and full final quality are separate requirements.',
        'One speed round is diagnostic only; final speed requires two unchanged om_bench rounds.',
        'Dense compute_ms is host enqueue duration and read_ms includes queued GPU execution and synchronization.',
        'Stage intervals can overlap; summing categories is not a wall-time decomposition.',
        'Greedy HTTP message hashes cannot replace raw token comparisons for DSpark verification.',
        'Profiler runs must remain separate from reported speed measurements.']
    with args.out.open('x') as output:
        json.dump(result, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps({'output': str(args.out), 'speed': result.get('speed'),
                      'decode_summary': result['decode_summary']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
