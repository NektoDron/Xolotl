#!/usr/bin/env python3
"""Prepare one same-geometry complete-logit diagnostic case against saved GPU oracle."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

from port5_bench_prepare import fingerprint
from server_make_session import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--case', default='code_18_3000')
    parser.add_argument('--profile', type=Path, default=ROOT / 'tools/m31/profiles/native-gpu.json')
    options, passthrough = parser.parse_known_args()
    if not re.fullmatch(r'[A-Za-z0-9_]+', options.case):
        parser.error('invalid case name')
    if not passthrough or '--help' in passthrough:
        return subprocess.call([sys.executable, str(ROOT / 'tools/m31/port6_validate.py'), 'prepare', '--help'])
    profile = json.loads(options.profile.read_text())
    environment = [item for key, value in sorted(profile['environment'].items())
                   for item in ('--env', key + '=' + value)]
    result = json.loads(subprocess.check_output([sys.executable, str(ROOT / 'tools/m31/port6_validate.py'),
                       'prepare', '--gate', 'code128', '--seed-hot', '0', *environment, *passthrough], text=True))
    job_path = Path(result['job'])
    directory = job_path.parent
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    job = json.loads(job_path.read_text())
    # The saved code128 link points into the independent full suite selected by port6.
    parent_suite = (Path(manifest['oracle']) / 'code_02_128').resolve().parent
    reference_case = parent_suite / options.case
    reference_meta = json.loads((reference_case / 'manifest.json').read_text())
    if not reference_meta.get('oracle_op_offload') or reference_meta.get('cuda_fusion_disabled') is not True:
        raise ValueError('requires accepted no-fusion GPU-offload oracle')
    selected_reference = directory / 'single-case-reference'
    selected_reference.mkdir()
    (selected_reference / options.case).symlink_to(reference_case, target_is_directory=True)
    ids = Path('/mnt/ai/logs/m31-port/ids-v1') / (options.case + '.i32')
    if not ids.is_file() or ids.stat().st_size // 4 != reference_meta['tokens']:
        raise ValueError('case IDs or reference token count mismatch')
    native = job['commands'][0]
    native[native.index('--ids-file') + 1] = str(ids)
    comparator = job['commands'][1]
    comparator[comparator.index('compare') + 1] = str(selected_reference)
    unset = sorted(set(manifest['unset_inherited_environment']) | set(profile['unset_environment']))
    extra = set(unset) - set(manifest['unset_inherited_environment'])
    native[1:1] = [item for key in sorted(extra) for item in ('-u', key)]
    manifest.update(gate='single-case-diagnostic', selected_case=options.case,
                    selected_ids=fingerprint(ids), selected_reference=str(selected_reference),
                    oracle=str(selected_reference), acceptance_eligible=False,
                    unset_inherited_environment=unset, runtime_profile=fingerprint(options.profile))
    manifest['fusion_policy'] = {
        'oracle': 'Accepted GPU-offload reference with CUDA fusion disabled.',
        'target_env_default_disabled': manifest['environment'].get('GGML_CUDA_DISABLE_FUSION') == '1',
        'target_decode_override': manifest['environment'].get('M31_GPU_DECODE_FUSION', '0'),
        'note': 'Legacy per-case cuda_fusion_disabled records the environment default; an explicit n1 override is separately recorded here.'}
    manifest['limits'].append('One complete-logit case including sparse MSA selection; never substitutes for final gate20 or quality.')
    for path, value in ((job_path, job), (manifest_path, manifest)):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
