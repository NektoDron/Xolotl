#!/usr/bin/env python3
"""Prepare native short-prefill plans and owned card jobs; never launch a GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[2]
LOGS = Path('/mnt/ai/logs/m31-vision')


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def arms(names):
    return [{'name': name, 'env': {
        'M31_SHORT_PREFILL_MODE': name,
        'M31_SHORT_PREFILL_MAX_N': '2048',
        'M31_SHORT_PREFILL_CPU_MAX_N': '128',
        'M31_SHORT_PREFILL_MATH': 'auto',
        'M31_SHORT_PREFILL_NVFP4_CPU': '0'}} for name in names]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--oracle', type=Path, default=LOGS/'final-build/m31_oracle')
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(exist_ok=False)
    profile = json.loads((ROOT/'tools/m31/profiles/vision-ttft.json').read_text())
    base_env = profile['environment']
    reference = json.loads((LOGS/'ttft-gate-plan-v1.json').read_text())
    model = reference['model']
    exact = json.loads((LOGS/'ttft/baseline-v1/text-inputs.json').read_text())
    inputs = args.out/'inputs'
    inputs.mkdir()
    for count in (515, 3863):
        ids = exact[str(count)]['tokens']
        if len(ids) != count:
            raise ValueError('saved text length differs')
        (inputs/f'text{count}.i32').write_bytes(struct.pack('<'+'i'*count, *ids))
    code = '/mnt/ai/logs/m31-port/ids-code128-v1/code128.i32'
    corpus = str(inputs/'text3863.i32')
    code3000 = '/mnt/ai/logs/m31-port/ids-long-v1/code_18_3000.i32'
    quick = [{'name': f'text{n}', 'ids_file': str(inputs/'text515.i32'), 'take': n}
             for n in (16, 128, 515)]
    rules = {'bitwise': False, 'min_top1': .99, 'max_mean_kl': .001, 'max_kl': .05}
    gates = [{**item, 'gate': rules} for item in reference['cases']]
    gates.insert(1, {'name': 'text515', 'ids_file': str(inputs/'text515.i32'),
                     'oracle': str(args.out/'oracle-text515/text515'), 'gate': rules})
    gates.append({'name': 'code3000', 'ids_file': code3000,
                  'oracle': str(args.out/'oracle-code3000/code_18_3000'), 'gate': rules})
    wide = [{'name': f'text{n}', 'ids_file': corpus, 'take': n}
            for n in (16, 32, 64, 100, 128, 256, 300, 512, 515, 1024, 2048)]
    wide += [{'name': f'prefix515-new{n}', 'ids_file': corpus, 'take': n,
              'prefix_ids_file': str(inputs/'text515.i32')} for n in (100, 300)]
    long_input = '/mnt/ai/logs/m31-port/port6r4-final-v8-acceptance-v1/long/ids/corpus_32768.i32'
    long_cases = [{'name': f'text{n}', 'ids_file': long_input, 'take': n}
                  for n in (4096, 16384, 32768)]
    specs = {
        'quick-gate': ([gates[-3]], ('off', 'cpu'), 0, 1, {}),
        'quick-sweep': (quick, ('off', 'cpu'), 1, 2, {}),
        'wide': (wide, ('off', 'cpu', 'hybrid'), 1, 3, {}),
        'gate': (gates, ('off', 'auto'), 0, 1, {}),
        'long': (long_cases, ('off', 'auto'), 1, 2,
                 {'context': 40000, 'max_chunk': 32768}),
    }
    # Keep the first gate unambiguously the accepted exact code128 case.
    specs['quick-gate'] = ([next(c for c in gates if c['name'] == 'code128')],
                           ('off', 'cpu', 'hybrid'), 0, 1, {})
    receipt = {'inputs': {}, 'plans': {}, 'launch': 'Use vision_card_launch.py; no GPU was launched by preparation.'}
    for label, (cases, modes, warmups, repeats, changes) in specs.items():
        config = {'context': 4096, 'max_chunk': 4096, 'threads': 80,
                  'arena_gib': 170, 'gpu_experts_gib': 7, 'profile': False,
                  'record_routes': False, **changes}
        plan = {'format': 'm31-shortprefill-plan-v1', 'model': model,
                'configuration': config, 'warmups': warmups, 'repeats': repeats,
                'arms': arms(modes), 'cases': cases}
        path = args.out/f'{label}.plan.json'
        write(path, plan)
        command = ['/usr/bin/env']
        for name in profile.get('unset_environment', []):
            command += ['-u', name]
        command += [f'{key}={value}' for key, value in base_env.items()]
        command += [str(args.engine.resolve()), '--plan', str(path), '--out', str(args.out/f'{label}-v1')]
        job = {'preset': 'm31-vision-test', 'measurement': True,
               'command_timeout_seconds': 14400, 'commands': [command]}
        write(args.out/f'shortprefill-{label}-v1.json', job)
        receipt['plans'][label] = str(path)
    command = ['/usr/bin/env', 'GGML_CUDA_DISABLE_FUSION=1', str(args.oracle.resolve())]
    common = ['--model', model, '--backend', 'cuda', '--ctx', '4096',
              '--threads', '80', '--oracle-op-offload', '1', '--rows', 'all', '--taps', 'none']
    oracle_job = {'preset': 'm31-vision-test', 'measurement': False,
                  'command_timeout_seconds': 5400, 'commands': []}
    for n, ids, label in [(515, str(inputs/'text515.i32'), 'text515'),
                           (3000, code3000, 'code3000')]:
        oracle_job['commands'].append(command+common+['--ids-file', ids, '--chunk', str(n),
                                                     '--out', str(args.out/f'oracle-{label}')])
    write(args.out/'shortprefill-oracles-v1.json', oracle_job)
    for path in inputs.iterdir():
        receipt['inputs'][str(path)] = {'bytes': path.stat().st_size,
                                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    write(args.out/'preparation.json', receipt)
    print(json.dumps({'prepared': str(args.out), 'plans': list(specs), 'launched': False}))


if __name__ == '__main__':
    main()
