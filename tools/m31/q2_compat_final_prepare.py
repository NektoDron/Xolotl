#!/usr/bin/env python3
"""Reviewed logs-only Q2 final gate preparation; no model/GPU launch."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT = Path('/mnt/ai/Xolotl-m31')
spec = importlib.util.spec_from_file_location('q2prep', ROOT / 'tools/m31/q2_compat_prepare.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--directory', required=True, type=Path)
p.add_argument('--canonical', required=True, type=Path)
p.add_argument('--profile', required=True, type=Path)
a = p.parse_args()
canonical = a.canonical.resolve(strict=True)
build = canonical / 'build-receipt.json'
receipt = json.loads(build.read_text())
if not receipt.get('compile_complete') or not receipt.get('source_stable'):
    raise ValueError('requires completed source-stable canonical')
source_before = canonical / 'source-before.json'
source_final = canonical / 'final-source.json'
if json.loads(source_before.read_text()) != json.loads(source_final.read_text()):
    raise ValueError('canonical before/final source snapshots differ')
source = {str(path): helper.fp(path) for path in (source_before, source_final)}
if not source:
    # Canonical receipt may identify immutable source through before/final files.
    snapshots = list(canonical.glob('*source*json'))
    source = {str(path): helper.fp(path) for path in snapshots}
if not source:
    raise ValueError('canonical source provenance absent')
flat_path = a.directory.parent / (a.directory.name + '-canonical-flat-receipt.json')
helper.write(flat_path, dict(format='m31-q2-canonical-flat-adapter-v1', elf={name: info['sha256'] for name, info in receipt['modes']['cuda']['elf'].items()},
             source=source, canonical_build_receipt=helper.fp(build), policy='Derived representation only; actual build receipt retained.'))
profile = json.loads(a.profile.read_text())
if profile.get('inherits') or profile['environment'].get('M31_CPU_STOCK_EXPERTS') != '1' or profile['environment'].get('M31_CPU_FUSED') != '1':
    raise ValueError('requires final resolved q2-native profile')
helper.prepare(argparse.Namespace(directory=a.directory, engine=canonical / 'bin-cuda/m31_generate',
               build_receipt=flat_path, oracle=ROOT / 'build-port6-bench-oracle/replay-v3/build/m31_oracle',
               oracle_receipt=ROOT / 'build-port6-bench-oracle/replay-v3/build-receipt.json',
               profile=a.profile, cpu_stock_experts=1))
r = a.directory.resolve()
helper.freeze(a.profile, r / 'production-profile-frozen.json')
helper.freeze(Path(__file__), r / 'final-gate-prepare-frozen.py')
m_path = r / 'manifest.json'
m = json.loads(m_path.read_text())
helper.write(r / 'pre-production-geometry-manifest.json', m)
engine = profile['engine_arguments']
production = dict(zip(engine[::2], engine[1::2]))
expected = {'--ctx': '40000', '--chunk': '32768', '--threads': '80', '--gpu-experts-gib': '7',
            '--gpu-experts-min-tokens': '32', '--reuse-allocations': '1', '--persistent-cpu-pool': '1',
            '--cpu-residual': '1', '--cpu-decode-cache': '1', '--dense-decode-cache': '1', '--profile': '1'}
for key, value in expected.items():
    if production.get(key) != value:
        raise ValueError('unexpected production engine argument: ' + key)
if float(production['--arena-gib']) != 140:
    raise ValueError('Q2 arena budget must be140GiB')
for case in m['cases']:
    d = r / case['name']
    plan_path = d / 'plan.json'
    job_path = Path(case['native_job'])
    job = json.loads(job_path.read_text())
    helper.write(d / 'pre-production-geometry-plan.json', json.loads(plan_path.read_text()))
    helper.write(d / 'pre-production-geometry-native-job.json', job)
    cmd = job['commands'][0]
    for key in ('--ctx', '--chunk'):
        cmd[cmd.index(key) + 1] = production[key]
        case['oracle_common'][case['oracle_common'].index(key) + 1] = production[key]
    case['context'], case['chunk'] = 40000, 32768
    actual = {key: cmd[cmd.index(key) + 1] for key in production}
    if any(float(actual[key]) != float(value) if key == '--arena-gib' else actual[key] != value for key, value in production.items()):
        raise ValueError('native gate CLI differs from final production engine arguments')
    for key, value in profile['environment'].items():
        if key + '=' + value not in cmd:
            raise ValueError('native gate environment differs from production: ' + key)
    plan_path.write_text(json.dumps(case, indent=2) + '\n')
    job_path.write_text(json.dumps(job, indent=2) + '\n')
m['format'] = 'm31-q2-final-selected-production-gate-v1'
m['production_profile'] = helper.fp(r / 'production-profile-frozen.json')
m['geometry_derivation'] = helper.fp(r / 'final-gate-prepare-frozen.py')
m['runtime_guards'] = ['Final stderr must identify M31 fused CPU MoE enabled=1.',
                       'All four selected cases use productionctx40000/maxchunk32768/arena140; actual adaptive chunks replayed into fresh Q2 oracle.',
                       'Tapsnone; code128 and long cases contain32actualsingleton teacher-forced inputs, not only PP.',
                       'Full environment and engine CLI match final HTTP production profile; head/rows/input/output correctness options are additional.']
m_path.write_text(json.dumps(m, indent=2) + '\n')
print(json.dumps({'directory': str(r), 'native_jobs': [c['native_job'] for c in m['cases']]}, indent=2))
