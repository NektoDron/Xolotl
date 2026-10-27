#!/usr/bin/env python3
"""Read-only postprocessing of saved same-request timings and TTFT trace JSONL."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from ttft_session import breakdown


def profile_traffic(log, cases):
    """Difference cumulative counters at each prefill, not at every decode log."""
    if not log.exists():
        return []
    group, groups = [], []
    for line in log.read_text().splitlines():
        group.append(line)
        if line.startswith('M31 request timings '):
            groups.append(group)
            group = []
    if len(groups) != len(cases):
        raise ValueError(f'{log}: {len(groups)} native requests vs {len(cases)} client cases')
    counters = {'expert': 0, 'prefetch': 0, 'direct': 0}
    profiles = [('expert', re.compile(r'^M31 GPU expert profile .*? bytes=(\d+)')),
                ('prefetch', re.compile(r'^M31 GPU prefetch profile .*? bytes=(\d+)')),
                ('direct', re.compile(r'^M31 GPU upload profile .*? bytes=(\d+)'))]
    records = []
    demand_re = re.compile(r'^M31 GPU route demand layer=(\d+) n=(\d+) required=(\d+) required_bytes=(\d+) predicted_used=(\d+) predicted_unused=(\d+) predicted_bytes=(\d+) history=(\d+)')
    for case, lines in zip(cases, groups):
        first = {}
        latest = dict(counters)
        prefix_seen = False
        demand = []
        for line in lines:
            if line.startswith('M31 forward pos=0 '):
                prefix_seen = True
            match = demand_re.match(line)
            if match:
                layer, n, required, required_bytes, used, unused, predicted_bytes, history = map(int, match.groups())
                expert_bytes = required_bytes // required if required else 0
                demand.append({'layer': layer, 'rows': n, 'required_experts': required,
                               'required_bytes': required_bytes, 'predicted_unused': unused,
                               'predicted_unused_bytes': unused * expert_bytes,
                               'predicted_bytes': predicted_bytes, 'history': bool(history)})
            for key, pattern in profiles:
                match = pattern.match(line)
                if match:
                    value = int(match[1])
                    latest[key] = value
                    if prefix_seen:
                        first.setdefault(key, value)
        delta = {key: first[key]-counters[key] if key in first else 0 for key in counters}
        if any(value < 0 for value in delta.values()):
            raise ValueError('GPU counters unexpectedly reset')
        if delta['expert'] != delta['prefetch'] + delta['direct']:
            raise ValueError('expert DMA differs from prefetch plus direct uploads')
        records.append({'name': case['name'], 'prompt_rows': case.get('expected_rows'),
                        'prefill_dma_bytes': delta['expert'], 'prefill_prefetch_bytes': delta['prefetch'],
                        'prefill_direct_bytes': delta['direct'],
                        'required_weight_bytes': sum(d['required_bytes'] for d in demand),
                        'unused_prefetch_bytes': sum(d['predicted_unused_bytes'] for d in demand),
                        'required_experts': sum(d['required_experts'] for d in demand),
                        'routed_layers': len(demand),
                        'actual_prefill_gb': delta['expert']/1e9,
                        'required_weight_gb': sum(d['required_bytes'] for d in demand)/1e9,
                        'unused_prefetch_gb': sum(d['predicted_unused_bytes'] for d in demand)/1e9,
                        'required_dma_floor_s_at_57GBps': sum(d['required_bytes'] for d in demand)/57e9})
        counters = latest
    return records


def summarize(directory):
    source = directory/'results.json'
    result = json.loads(source.read_text())
    trace_file = Path(result.get('trace_file', directory/'trace.jsonl'))
    traces = {}
    plan_file = directory/'plan.json'
    plan = json.loads(plan_file.read_text()) if plan_file.exists() else {}
    cache_flags = {case['name']: case['body'].get('cache_prompt') for case in plan.get('cases', [])}
    if trace_file.exists():
        for line in trace_file.read_text().splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            traces[item['id']] = item
    cases = []
    for case in result['cases']:
        trace = case.get('trace') or traces.get(case.get('id'))
        row = {'name': case['name'], 'pass': case.get('pass'), 'trace_present': trace is not None}
        if trace is None:
            row['error'] = case.get('error', 'missing trace')
            cases.append(row)
            continue
        value = {**case, 'trace': trace}
        native = case.get('native_timings') or trace['details'].get('native_timings', {})
        value['native_timings'] = native
        encoder = trace['details'].get('encoder', {})
        adapter = encoder.get('adapter_timings_ms', {})
        stages = trace['stages_ms']
        row.update({'prompt_rows': native.get('prompt_n'), 'ttft_ms': case['ttft_s']*1000,
                    'body_cache_prompt': case.get('body_cache_prompt', cache_flags.get(case['name'])),
                    'native_prompt_cache': native.get('prompt_cache', {}),
                    'prompt_cache_optout_checks': case.get('prompt_cache_optout_checks'),
                    'visible_ttft_ms': None if case.get('visible_ttft_s') is None else case['visible_ttft_s']*1000,
                    'server_prepare_ms': stages.get('prepare_total_ms'),
                    'media_prepare_ms': stages.get('media_prepare_ms'),
                    'template_rpc_ms': stages.get('chat_template_rpc_ms'),
                    'encoder_python_ms': stages.get('encoder_python_total_ms'),
                    'encoder_worker_started': encoder.get('worker_started'),
                    'encoder_artifact_hit': encoder.get('cache_hit'),
                    'encoder_stage_ms': encoder.get('stage_ms', {}),
                    'adapter_timings_ms': adapter,
                    'native_stages_ms': native.get('ttft_stages_ms', {}),
                    'actual_chunks': native.get('actual_chunks'),
                    'breakdown': breakdown(value)})
        cases.append(row)
    return {'format': 'm31-ttft-breakdown-v1', 'source': str(source),
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'trace_file': str(trace_file),
            'source_session_pass': result.get('pass'),
            'session_complete': 'server_returncode' in result,
            'environment': result.get('environment'), 'cases': cases,
            'profile_traffic': profile_traffic(directory/'server.log', result['cases']),
            'notes': 'Saved raw results and traces are not modified. Nested timers are not added to parent milestones.'}


def markdown(summary):
    lines = ['| Запрос | TTFT, мс | Prepare | Encoder | Prefill forward | Reset | F32 load/validate | Token→delta |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    def f(value):
        return '—' if value is None else f'{value:.3f}'
    for case in summary['cases']:
        if not case['trace_present']:
            lines.append(f'| {case["name"]} | ERROR | | | | | | |')
            continue
        native = case['native_stages_ms']
        partition = case['breakdown']['critical_path_ms']
        values = [case['ttft_ms'], case.get('server_prepare_ms'), case.get('encoder_python_ms'),
                  native.get('prefill_forward_ms'), native.get('reset_ms'),
                  native.get('mixed_manifest_read_validate_ms'), partition['token_parse_to_first_delta_ms']]
        lines.append('| '+case['name']+' | '+' | '.join(f(v) for v in values)+' |')
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--markdown', type=Path)
    args = parser.parse_args()
    summary = summarize(args.directory.resolve())
    output = args.output or args.directory/'breakdown.json'
    with output.open('x') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    table = markdown(summary)
    if args.markdown:
        with args.markdown.open('x') as stream:
            stream.write(table)
    print(table, end='')
    return 0 if all(c['trace_present'] for c in summary['cases']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
