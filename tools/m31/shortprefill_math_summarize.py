#!/usr/bin/env python3
"""Summarize a saved single-expert CUDA/CPU diagnostic; no inference or gate."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def summarize(path):
    data = json.loads(path.read_text())
    if data.get('format') != 'm31-expert-check-v1' or not data.get('short_cpu_check'):
        raise ValueError('requires --short-cpu-check report')
    modes = ('cuda_id_auto', 'cpu_id', 'cpu_cuda_mmq')
    rows = {}
    maxima = defaultdict(lambda: {'max_relative_l2': 0, 'max_abs': 0, 'nonfinite': 0,
                                 'bitwise_stages': 0, 'stages': 0, 'worst_relative_l2_case': None})
    for case in data['cases']:
        if case['tokens'] not in (16, 128):
            continue
        if case['reference_mode'] != 'cuda_id_auto':
            raise ValueError('summary expects native cuda_id_auto reference')
        for result in case['results']:
            mode, stage = result['mode'], result['stage']
            if mode not in modes:
                continue
            metrics = result['metrics']
            key = (case['layer'], case['tokens'], mode)
            row = rows.setdefault(key, {'layer': case['layer'], 'tokens': case['tokens'],
                                       'mode': mode, 'type_names': case['type_names'],
                                       'external_scales': case['scales'], 'primitive': {}, 'full_ffn': {}})
            selected = {name: metrics[name] for name in ('count', 'bitwise', 'relative_l2', 'max_abs',
                                                        'nonfinite_reference', 'nonfinite_candidate')}
            if case['operation'] == 'full':
                row['full_ffn'][stage] = selected
            else:
                row['primitive'].setdefault(case['operation'], {})[stage] = selected
            maximum = maxima[(mode, 'full_ffn' if case['operation'] == 'full' else 'primitive')]
            maximum['stages'] += 1
            maximum['bitwise_stages'] += metrics['bitwise']
            maximum['nonfinite'] += metrics['nonfinite_reference']+metrics['nonfinite_candidate']
            maximum['max_abs'] = max(maximum['max_abs'], metrics['max_abs'])
            if metrics['relative_l2'] > maximum['max_relative_l2']:
                maximum['max_relative_l2'] = metrics['relative_l2']
                maximum['worst_relative_l2_case'] = {'layer': case['layer'], 'tokens': case['tokens'],
                                                     'operation': case['operation'], 'stage': stage}
    return {'format': 'm31-shortprefill-single-expert-summary-v1',
            'source': str(path.resolve()), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'model': data['model'], 'seed': data['seed'], 'expert': data['expert'],
            'scope': 'Synthetic single-expert diagnostic at n=16/128. Native cuda_id_auto is the reference. No full-model numerical acceptance or TTFT acceptance is implied. Raw NVFP4 outputs precede external scale; scaled maxima are also retained.',
            'maxima': [{'mode': key[0], 'scope': key[1], **value} for key, value in sorted(maxima.items())],
            'rows': [rows[key] for key in sorted(rows)]}


def markdown(value):
    lines = ['# Микропроверка экспертов: n16/n128', '',
             'Эталон — native `cuda_id_auto`. Это синтетическая проверка одного эксперта; допуск всей модели и TTFT по ней не объявляются.', '',
             '| Слой | N | Режим | primitive gate relL2 | primitive up relL2 | primitive down relL2 | full OAI relL2 | full down scaled relL2 | full down scaled maxAbs |',
             '|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in value['rows']:
        primitive = row['primitive']
        values = [primitive[role]['mm_raw']['relative_l2'] for role in ('gate', 'up', 'down')]
        values += [row['full_ffn']['oai']['relative_l2'], row['full_ffn']['down_scaled']['relative_l2'],
                   row['full_ffn']['down_scaled']['max_abs']]
        lines.append(f"| {row['layer']} | {row['tokens']} | {row['mode']} | "+' | '.join(f'{number:.6g}' for number in values)+' |')
    lines += ['', 'Raw/scaled значения и исходные внешние масштабы сохранены в JSON. Их нельзя смешивать при сравнении maxAbs NVFP4. Небольшая ошибка на одной матрице может влиять на routing и усиливаться в полной модели.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    value = summarize(args.report)
    args.out.with_suffix('.json').write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    args.out.with_suffix('.md').write_text(markdown(value))
    print(json.dumps({'rows': len(value['rows']), 'maxima': value['maxima']}))


if __name__ == '__main__':
    main()
