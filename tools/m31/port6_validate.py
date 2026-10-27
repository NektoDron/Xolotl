#!/usr/bin/env python3
"""Prepare port6 GPU-oracle acceptance and opt-in CPU verification with agreed tolerances.

Preparation never reserves a GPU. Comparison reads logits and must run as a long
systemd task; --require-suite is necessary before declaring the final gate20.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys

import compare_logits

LOG = Path('/mnt/ai/logs/m31-port')
THRESHOLDS = dict(top1=.99, mean_kl=1e-3, max_kl=.05)



def validate_full_suite(reference, candidate, expected_offload):
    cases = sorted(path for path in reference.iterdir() if path.is_dir())
    if len(cases) != 20 or not {'ru', 'en', 'code', 'math'} <= {path.name.split('_', 1)[0] for path in cases}:
        raise ValueError('gate20 requires exactly 20 RU/EN/code/math prompts')
    lengths = []
    for case in cases:
        a, _ = compare_logits.load_case(case)
        b, _ = compare_logits.load_case(candidate / case.name)
        if a['engine'] != 'llama.cpp-PR31192' or b['engine'] != 'xolotl-m31-correctness':
            raise ValueError('gate20 requires independent PR and Xolotl engines')
        if bool(a.get('oracle_op_offload', False)) != expected_offload:
            raise ValueError('oracle placement contradicts requested gate20 contract')
        if expected_offload and (a.get('cuda_fusion_disabled') is not True or b.get('cuda_fusion_disabled') is not True):
            raise ValueError('final GPU gate20 requires CUDA fusion disabled in both engines')
        if compare_logits.execution_chunks(a) != compare_logits.execution_chunks(b):
            raise ValueError('gate20 actual chunk geometry differs from oracle')
        if a.get('head_tail', 0) or b.get('head_tail', 0) or a['positions'] != list(range(a['tokens'])) or b['positions'] != list(range(b['tokens'])):
            raise ValueError('gate20 requires every position and full head')
        lengths.append(a['tokens'])
    if min(lengths) != 64 or max(lengths) != 3000 or any(n < 64 or n > 3000 for n in lengths) or sum(n > 2048 for n in lengths) < 4:
        raise ValueError('gate20 must cover 64..3000 with enough sparse-selection prompts')


def compare(args):
    contract = getattr(args, 'oracle_contract', 'gpu-offload')
    expected_offload = contract == 'gpu-offload'
    # Refuse an inappropriate placement oracle before reading any large logits.
    for path in args.reference.iterdir():
        if path.is_dir():
            manifest = json.loads((path / 'manifest.json').read_text())
            if manifest['engine'] != 'llama.cpp-PR31192' or bool(manifest.get('oracle_op_offload', False)) != expected_offload:
                raise ValueError('oracle placement metadata contradicts explicit ' + contract + ' contract')
            if not compare_logits.full_last_layer_before_crop(manifest):
                raise ValueError('refuses legacy bounded oracle cropping FFN59')
    if args.require_suite:
        validate_full_suite(args.reference, args.candidate, expected_offload)
    result = compare_logits.compare_suite(args.reference, args.candidate,
                                          require_suite=False, **THRESHOLDS)
    result['pass'] = (result['top1'] >= THRESHOLDS['top1']
                      and result['mean_kl'] <= THRESHOLDS['mean_kl']
                      and result['max_kl'] <= THRESHOLDS['max_kl'])
    result['per_prompt_violations'] = [c['case'] for c in result['cases']
        if c['top1'] < THRESHOLDS['top1'] or c['mean_kl'] > THRESHOLDS['mean_kl']
        or c['max_kl'] > THRESHOLDS['max_kl']]
    result['format'] = 'm31-port6-numeric-gate-v1'
    result['threshold_comparison'] = 'aggregate top1 >= .99; aggregate mean KL <= .001; max KL over all positions <= .05; per-prompt violations diagnostic'
    result['reference'] = str(args.reference.resolve())
    result['candidate'] = str(args.candidate.resolve())
    result['oracle_contract'] = contract
    result['acceptance_eligible'] = bool(args.require_suite and expected_offload)
    result['scope'] = (('port6-gpu-offload-gate20-complete-logits' if expected_offload
                        else 'port6-cpu-strict-optin-complete-logits') if args.require_suite
                       else 'port6-gpu-offload-diagnostic' if expected_offload else 'port6-cpu-strict-optin-diagnostic')
    with args.out.open('x') as output:
        json.dump(result, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=2))
    return 0 if result['pass'] else 1



def dual_compare(args):
    outcomes = {}
    for contract, reference, filename in [('cpu-strict', args.reference_cpu, 'compare-cpu-strict.json'),
                                           ('gpu-offload', args.reference_gpu, 'compare-gpu-offload.json')]:
        out = args.out.parent / filename
        local = argparse.Namespace(reference=reference, candidate=args.candidate, out=out,
                                   require_suite=False, oracle_contract=contract)
        rc = compare(local)
        value = json.loads(out.read_text())
        outcomes[contract] = {'output': str(out), 'comparison_exit_code': rc, 'metric_pass': value['pass'],
                              'scope': value['scope'], 'top1': value['top1'], 'rows': value['rows'],
                              'mean_kl': value['mean_kl'], 'max_kl': value['max_kl']}
    result = {'format': 'm31-port6-dual-oracle-diagnostic-v1', 'scope': 'diagnostic-only',
              'comparisons': outcomes, 'acceptance_oracle_contract': 'gpu-offload',
              'full_gate20_accepted': False,
              'oracle_choice_pending_user_clarification': False,
              'oracle_policy_provenance': 'Official coordinator decision after review: GPU-offload llama.cpp PR31192 is primary; CPU strict is opt-in verification.',
              'limits': 'Both placements are reported on the same native outputs. GPU is primary by explicit coordinator decision; code128 does not substitute for full20, long accuracy or GSM/tools. Exit0 means diagnostics completed.'}
    with args.out.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))
    return 0


def prepare(args):
    from port5_bench_prepare import fingerprint
    from port6_bench_receipt import verify_receipt
    from server_make_session import MODEL, ROOT
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,150}', args.arm):
        raise ValueError('invalid arm')
    if args.dual_oracle and args.gate != 'code128':
        raise ValueError('dual placement mode is code128 diagnostic only; final gate20 uses the explicit GPU-offload contract')
    if not args.engine.is_file() or args.threads < 1 or args.gpu_experts_gib <= 0:
        raise ValueError('invalid engine/threads/cache')
    environment = json.loads((LOG / 'port5-gpu-v4-full-summary-v1.json').read_text())['environment']
    environment.update(M31_GPU_EXACT='0', M31_GPU_FAST_FFN='0', M31_GPU_CPU_QUANTS='0',
                       M31_GPU_FAST_FFN_DOTS='cublas', M31_GPU_FAST_FFN_LIBM_EXPF='0', M31_GPU_FAST_FFN_EXP_DOUBLE='0')
    for item in args.env:
        key, sep, value = item.partition('=')
        if not sep or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key):
            raise ValueError('--env requires NAME=VALUE')
        if key in ('CUDA_VISIBLE_DEVICES', 'M31_CARD_SESSION'):
            raise ValueError('device/session environment belongs to card_session')
        environment[key] = value
    args.directory = args.directory.resolve()
    args.directory.mkdir(parents=True, exist_ok=False)
    frozen = args.directory / 'm31_generate-frozen'
    original = fingerprint(args.engine)
    with args.engine.open('rb') as source, frozen.open('xb') as target:
        shutil.copyfileobj(source, target, 1 << 20)
    frozen.chmod(0o555)
    if fingerprint(frozen)['sha256'] != original['sha256']:
        raise ValueError('engine changed during freeze')
    reference = (args.reference.resolve() if args.reference else LOG /
                 ('oracle-ggml-gpuoracle-suite20-c2048-d32-v1' if args.oracle_contract == 'gpu-offload'
                  else 'oracle-integrated-suite20-c2048-d32-v1'))
    chunk, head_tail, rows = 2048, 0, 'all'
    ids = LOG / 'ids-v1'
    if args.gate == 'code128':
        selected = args.directory / 'reference'
        selected.mkdir()
        (selected / 'code_02_128').symlink_to(reference / 'code_02_128', target_is_directory=True)
        reference = selected
        case_args = ['--ids-file', str(ids / 'code_02_128.i32')]
    elif args.gate == 'large3000':
        if args.oracle_contract == 'gpu-offload' and not args.reference:
            raise ValueError('GPU large3000 requires explicit --reference with matching bounded head/chunks; generate the oracle first')
        reference = args.reference.resolve() if args.reference else LOG / 'port5-oracle-full-last-tail64-v2/oracle-c8192'
        chunk, head_tail, rows = args.large_chunk, 64, 'last:64'
        if args.large_chunk < 2968:
            raise ValueError('large3000 oracle requires one 2968-token prefill chunk')
        case_args = ['--ids-file', str(ids / 'code_18_3000.i32')]
    else:
        case_args = ['--cases', str(ids)]
    # Metadata/stat checks only: preparation does not read multi-GB logits.
    for path in reference.iterdir():
        if path.is_dir():
            manifest = json.loads((path / 'manifest.json').read_text())
            if manifest['engine'] != 'llama.cpp-PR31192' or bool(manifest.get('oracle_op_offload', False)) != (args.oracle_contract == 'gpu-offload'):
                raise ValueError('reference placement contradicts explicit requested contract')
            if not compare_logits.full_last_layer_before_crop(manifest):
                raise ValueError('refuses legacy bounded oracle cropping FFN59')
    output = args.directory / 'native'
    native = [str(frozen), '-m', MODEL, '--backend', 'cuda', '--ctx', '4096',
              '--chunk', str(chunk), '--threads', str(args.threads), '--reuse-allocations', '1',
              '--persistent-cpu-pool', '1', '--cpu-residual', '1', '--cpu-decode-cache', '1',
              '--dense-decode-cache', '1', '--profile', '1', '--arena-gib', '170',
              '--gpu-experts-gib', str(args.gpu_experts_gib), '--gpu-experts-min-tokens', '32',
              '--taps', 'none', '--decode-tail', '32',
              *(['--head-tail', str(head_tail)] if head_tail else []),
              '--rows', rows, *case_args, '--out', str(output)]
    slots = int(args.gpu_experts_gib * (1 << 30)) // 25909632
    if not 0 <= args.seed_hot < slots:
        raise ValueError('seed-hot must leave streaming slots')
    if args.seed_hot:
        native = [sys.executable, str(ROOT / 'tools/m31/cache_seed_launch.py'),
                  '--rank', str(LOG / 'cache-ranking-v3.json'), '--binary', str(LOG / 'cache-ranking-v3.bin'),
                  '--tensors', str(LOG / 'refs/gguf-tensors.json'),
                  '--integrity', '/mnt/ai/logs/m31-ref/integrity-ud-original.json',
                  '--slots', str(slots), '--hot', str(args.seed_hot),
                  '--output', str(args.directory / 'seed-preflight.json'), '--', *native]
    compare_dir = args.directory / 'comparator'
    compare_dir.mkdir()
    for name in ('port6_validate.py', 'compare_logits.py'):
        target = compare_dir / name
        with (ROOT / 'tools/m31' / name).open('rb') as source, target.open('xb') as stream:
            shutil.copyfileobj(source, stream)
        target.chmod(0o444)
    compare_argv = [sys.executable, str(compare_dir / 'port6_validate.py'), 'compare', str(reference), str(output),
                    '--out', str(args.directory / 'compare.json'), '--oracle-contract', args.oracle_contract]
    if args.gate == 'gate20':
        compare_argv.append('--require-suite')
    reference_gpu = None
    if args.dual_oracle:
        if args.oracle_contract != 'cpu-strict':
            raise ValueError('--dual-oracle code128 preparation requires explicit --oracle-contract cpu-strict so each placement is kept distinct')
        reference_gpu = args.directory / 'reference-gpu'
        reference_gpu.mkdir()
        gpu_case = LOG / 'oracle-ggml-gpuoracle-suite20-c2048-d32-v1/code_02_128'
        gpu_metadata = json.loads((gpu_case / 'manifest.json').read_text())
        if not gpu_metadata.get('oracle_op_offload', False):
            raise ValueError('secondary oracle is not GPU op-offload')
        (reference_gpu / 'code_02_128').symlink_to(gpu_case, target_is_directory=True)
        compare_argv = [sys.executable, str(compare_dir / 'port6_validate.py'), 'dual-compare',
                        '--reference-cpu', str(reference), '--reference-gpu', str(reference_gpu),
                        '--candidate', str(output), '--out', str(args.directory / 'dual-compare.json')]
    job_path = args.directory / (args.arm + '-card-job.json')
    unset_inherited = ['CUDA_DEVICE_MAX_CONNECTIONS', 'CUDA_DEVICE_MAX_COPY_CONNECTIONS']
    unset_argv = [item for name in unset_inherited for item in ('-u', name)]
    job = {'preset': 'm31-xolotl-check', 'measurement': False,
           'commands': [['/usr/bin/env', *unset_argv, *[k + '=' + v for k, v in sorted(environment.items())], *native], compare_argv]}
    manifest = {'format': 'm31-port6-gate-plan-v1', 'gate': args.gate, 'status': 'prepared-only',
                'engine': fingerprint(frozen), 'original_engine': original, 'environment': environment,
                'unset_inherited_environment': unset_inherited, 'thresholds': THRESHOLDS, 'oracle': str(reference), 'chunk': chunk, 'head_tail': head_tail,
                'context': 4096, 'threads': args.threads, 'gpu_experts_gib': args.gpu_experts_gib,
                'seed_hot': args.seed_hot, 'source_head': 'See candidate build receipt; workspace may change.',
                'launch_argv': [sys.executable, str(ROOT / 'tools/m31/card_launch.py'), str(job_path), '--unit', args.arm],
                'limits': ['Diagnostic gates do not substitute for full gate20.',
                           'gate20 has chunk2048 for oracle geometry; large3000 separately covers full-batch prefill.',
                           'Thresholds are inclusive: aggregate top1/mean KL, maximum KL of all positions; per-prompt violations diagnostic.']}
    manifest['oracle_contract'] = args.oracle_contract
    manifest['secondary_oracle'] = str(reference_gpu) if reference_gpu else None
    manifest['acceptance_eligible'] = args.gate == 'gate20' and args.oracle_contract == 'gpu-offload' and not args.dual_oracle
    manifest['oracle_policy_provenance'] = 'Official coordinator decision: GPU-offload PR31192 primary; CPU strict verification is opt-in.'
    if args.dual_oracle:
        manifest['limits'].append('Dual code128 diagnostic reports both placements. GPU is primary by explicit coordinator decision; full20 and long accuracy remain required.')
    manifest['comparator'] = [fingerprint(path) for path in sorted(compare_dir.iterdir())]
    if args.build_receipt:
        manifest['build_receipt'] = verify_receipt(args.build_receipt, manifest['engine']['sha256'], 'm31_generate')
    for path, value in ((job_path, job), (args.directory / 'manifest.json', manifest)):
        with path.open('x') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    print(json.dumps({'job': str(job_path), 'launch_argv': manifest['launch_argv'],
                      'sha256': manifest['engine']['sha256']}, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    prep = modes.add_parser('prepare')
    prep.add_argument('--directory', type=Path, required=True)
    prep.add_argument('--arm', required=True)
    prep.add_argument('--engine', type=Path, required=True)
    prep.add_argument('--build-receipt', type=Path)
    prep.add_argument('--oracle-contract', choices=['gpu-offload', 'cpu-strict'], default='gpu-offload')
    prep.add_argument('--reference', type=Path, help='explicit oracle suite path, required for a newly generated GPU bounded large case')
    prep.add_argument('--gate', choices=['code128', 'large3000', 'gate20'], required=True)
    prep.add_argument('--threads', type=int, default=80)
    prep.add_argument('--gpu-experts-gib', type=float, default=7)
    prep.add_argument('--seed-hot', type=int, default=20)
    prep.add_argument('--large-chunk', type=int, default=8192)
    prep.add_argument('--dual-oracle', action='store_true', help='code128 diagnostic against CPU and GPU placement oracles, one native pass')
    prep.add_argument('--env', action='append', default=[])
    comp = modes.add_parser('compare')
    comp.add_argument('reference', type=Path)
    comp.add_argument('candidate', type=Path)
    comp.add_argument('--out', type=Path, required=True)
    comp.add_argument('--require-suite', action='store_true')
    comp.add_argument('--oracle-contract', choices=['cpu-strict', 'gpu-offload'], default='gpu-offload')
    dual = modes.add_parser('dual-compare')
    dual.add_argument('--reference-cpu', type=Path, required=True)
    dual.add_argument('--reference-gpu', type=Path, required=True)
    dual.add_argument('--candidate', type=Path, required=True)
    dual.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.mode == 'prepare':
            return prepare(args)
        return dual_compare(args) if args.mode == 'dual-compare' else compare(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f'port6_validate: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
