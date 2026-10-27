#!/usr/bin/env python3
"""Audit one completed selected resident-placement Q2 oracle diagnostic."""
import argparse
import json
from pathlib import Path
import re

from compare_logits import compare_case, execution_chunks
from q2_compat_prepare import completed, fp, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve(strict=True)
    plan = json.loads((directory / 'manifest.json').read_text())
    result = completed(Path(plan['job']))
    if fp(Path(plan['oracle']['path'])) != plan['oracle']:
        raise ValueError('placement oracle ELF changed')
    native = Path(plan['native_case'])
    oracle = directory / 'oracle' / native.name
    placement = json.loads((directory / 'oracle/oracle-placement.json').read_text())
    wanted = {int(x) for x in plan['layers'].split(',')}
    if wanted not in ({10, 11, 12}, {10, 11, 12, 13}):
        raise ValueError('unreviewed expert placement')
    if placement['gpu_expert_layers'] != plan['layers'] or placement['context_requested'] != 40000 or not placement['oracle_op_offload'] or placement['kv'] != 'f16':
        raise ValueError('runtime placement sidecar differs from plan')
    command = result['commands'][0]['argv']
    if 'M31_ORACLE_GPU_EXPERT_LAYERS=' + plan['layers'] not in command:
        raise ValueError('actual command does not select intended placement')
    log = Path(result['commands'][0]['log'])
    # PR loader independently reports the actual override chosen for each role.
    runtime_log = log.read_text(errors='replace')
    physical_contexts = re.findall(r'^llama_context: n_ctx\s*=\s*(\d+)\s*$', runtime_log, re.M)
    if physical_contexts != ['40192']:
        raise ValueError('oracle actual physical context is not expected PAD(40000,256)')
    observations = re.findall(r'^tensor (blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight) .*?buffer type overridden to (\S+)', runtime_log, re.M)
    if len(observations) != 171 or len({row[0] for row in observations}) != 171:
        raise ValueError('missing/duplicate actual routed weight placement evidence')
    for name, layer, role, buffer in observations:
        if buffer != ('CUDA0' if int(layer) in wanted else 'CUDA_Host'):
            raise ValueError('unexpected runtime storage for ' + name + ': ' + buffer)
    nm = json.loads((native / 'manifest.json').read_text())
    om = json.loads((oracle / 'manifest.json').read_text())
    for header in (nm, om):
        if header['context'] != 40000 or header['kv'] != 'f16' or header.get('cuda_mmq_precision') != 'auto' or not header['cuda_fusion_disabled']:
            raise ValueError('physical context/KV/fusion contract differs')
    if not om['oracle_op_offload'] or om['engine'] != 'llama.cpp-PR31192' or nm['engine'] != 'xolotl-m31-correctness':
        raise ValueError('wrong source engines/placement policy')
    if nm.get('m31_decode_fusion_override_requested') or nm.get('m31_decode_fusion_unfenced_requested'):
        raise ValueError('native has an unreviewed fusion override')
    if execution_chunks(nm) != execution_chunks(om) or execution_chunks(nm) != [(0, 96), *[(p, 1) for p in range(96, 128)]]:
        raise ValueError('requires actual PP96 plus32 singleton teacher-forced inputs')
    metrics = compare_case(oracle, native)
    metrics.update(format='m31-q2-best-resident-placement-code128-v1',
                   acceptance_eligible=False, selected_expert_gpu_layers=sorted(wanted),
                   actual_gpu_expert_roles=sum(int(layer) in wanted for _, layer, _, _ in observations),
                   context_requested=40000, oracle_physical_context_actual=40192,
                   score_implementation=fp(Path(__file__)), comparator_implementation=fp(Path(__file__).with_name('compare_logits.py')),
                   runtime_placement_log=fp(log), runtime_placement_sidecar=fp(directory / 'oracle/oracle-placement.json'),
                   native_header=fp(native / 'manifest.json'), oracle_header=fp(oracle / 'manifest.json'),
                   thresholds=dict(top1=.99, mean_kl=.001, max_kl=.05),
                   limits='Selected code128 teacher-forced diagnostic only. Matches chosen reference expert storage, but uses F16 KV/ctx40000/fusionoff; reference HTTP is Q8 KV/ctx65536/fusiondefault. No universal greedy/long/fullquality equivalence.')
    metrics['pass'] = metrics['top1'] >= .99 and metrics['mean_kl'] <= .001 and metrics['max_kl'] <= .05
    write(directory / 'compare.json', metrics)
    print(json.dumps(metrics, indent=2))
    return 0 if metrics['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
