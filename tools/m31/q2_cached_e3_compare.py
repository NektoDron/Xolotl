#!/usr/bin/env python3
"""Prepare/score one bounded cached-E3 code128 candidate; never launch a model."""
from __future__ import annotations
import argparse
import hashlib
import datetime
import json
from pathlib import Path
import re

from cache_seed_preflight import decode_rank, seal_receipt, verify
from compare_logits import compare_case, execution_chunks
from q2_compat_prepare import completed, fp, freeze, write

ROOT = Path('/mnt/ai/Xolotl-m31')
LOG = Path('/mnt/ai/logs/m31-port')
GEN_SHA = '2b51de98bbae67d6f9d14a83da1d530d6cf04fa47cdbe7ab917bd99c466e9a8c'
STRIDE, SLOTS, HOT = 27615616, 384, 384
SEED_BYTES = 6964641792


def require(condition, text):
    if not condition:
        raise ValueError(text)


def expected_chunks():
    return [(0, 96), *[(p, 1) for p in range(96, 128)]]


def small_inputs(directory):
    return {name: directory / name for name in ('runtime-profile.json', 'rank.json', 'rank.bin',
             'tensors.json', 'integrity.json', 'preflight.json', 'seed.receipt')}


def validate_seed(directory):
    paths = small_inputs(directory)
    rank = json.loads(paths['rank.json'].read_text())
    pre = json.loads(paths['preflight.json'].read_text())
    # Independently repeat the existing header/stat/identity verifier, retaining
    # its historical prefill reserve. Decode-only permission is separately sealed.
    baseline = verify(paths['rank.json'], paths['rank.bin'], paths['tensors.json'],
                      paths['integrity.json'], SLOTS, 0)
    for key in ('identity_sha256', 'rank_json_sha256', 'rank_binary_sha256',
                'tensor_header_sha256', 'checked_shards'):
        require(pre[key] == baseline[key], 'decode preflight differs from current model: ' + key)
    require(pre.get('verified') is True and pre.get('mode') == 'decode-only', 'requires verified decode-only preflight')
    require(pre['budget']['slots'] == SLOTS and pre['budget']['requested_hot'] == HOT
            and pre['budget']['max_hot'] == HOT and pre['budget']['stream_slots'] == 0,
            'decode-only seed must occupy the complete384slot pool')
    require(seal_receipt(pre) == paths['seed.receipt'].read_bytes(), 'runtime seed receipt differs from trusted preflight')
    ids = decode_rank(paths['rank.bin'].read_bytes(), rank['identity_sha256'], 60, 128, 3)
    require(set(ids[:HOT]) == set(range(10 * 128, 13 * 128)), 'first384 seed keys must be all E3 layer experts and no others')
    require(len(ids[:HOT]) == len(set(ids[:HOT])) == HOT, 'duplicate seed keys')
    return pre


def prepare(args):
    directory = args.directory.resolve(strict=True)
    job = json.loads(args.job.read_text())
    require(len(job['commands']) == 1 and job.get('measurement') is False, 'one correctness command required')
    command = job['commands'][0]
    profile = json.loads((directory / 'runtime-profile.json').read_text())
    env = profile['environment']
    required_env = dict(M31_GPU_CACHED_DECODE='1', M31_GPU_CACHED_DECODE_NATIVE='1',
         M31_GPU_CACHED_DECODE_EXACT='0', M31_GPU_DECODE_SEED_HOT='384', M31_GPU_SEED_HOT='0',
         M31_CPU_STOCK_EXPERTS='1', M31_CPU_FUSED='1', M31_GPU_EXPERT_PROFILE='0',
         M31_GPU_EXACT='0', M31_GPU_CPU_QUANTS='0', M31_GPU_FAST_FFN='0', M31_GPU_LAYER_BANKS='0',
         GGML_CUDA_DISABLE_FUSION='1', M31_GPU_DECODE_FUSION='0', M31_GPU_RMS_FUSION='0',
         M31_DECODE_FUSION_UNFENCED='0')
    for key, value in required_env.items():
        require(env.get(key) == value and key + '=' + value in command, 'candidate environment mismatch: ' + key)
    for key, value in env.items():
        require(key + '=' + value in command, 'profile/command environment differs: ' + key)
    required_args = {'--backend': 'cuda', '--ctx': '40000', '--chunk': '32768', '--threads': '80',
         '--gpu-experts-gib': '9.88', '--arena-gib': '140', '--gpu-experts-min-tokens': '32',
         '--reuse-allocations': '1', '--persistent-cpu-pool': '1', '--cpu-residual': '1',
         '--cpu-decode-cache': '1', '--dense-decode-cache': '1', '--profile': '1',
         '--decode-tail': '32', '--rows': 'all', '--taps': 'none'}
    for key, value in required_args.items():
        require(key in command, 'missing candidate CLI: ' + key)
        actual = command[command.index(key) + 1]
        require(float(actual) == float(value) if key in ('--arena-gib', '--gpu-experts-gib') else actual == value,
                'candidate CLI differs: ' + key)
    pre = validate_seed(directory)
    for key, expected in {'M31_GPU_SEED_PATH': str((directory / 'rank.bin').resolve()),
                          'M31_GPU_SEED_IDENTITY': pre['identity_sha256'],
                          'M31_GPU_SEED_SHA256': pre['rank_binary_sha256'],
                          'M31_GPU_SEED_RECEIPT': str((directory / 'seed.receipt').resolve()),
                          'M31_GPU_SEED_RECEIPT_SHA256': fp(directory / 'seed.receipt')['sha256']}.items():
        require(env.get(key) == expected, 'runtime seed identity differs: ' + key)
    elf = Path(args.engine).resolve(strict=True)
    require(str(elf) in command, 'selected candidate ELF absent from command')
    require(command[0] == '/usr/bin/env', 'requires direct reviewed env-to-ELF launch')
    assignments, unset_names, prefix = {}, [], command[1:command.index(str(elf))]
    i = 0
    while i < len(prefix):
        if prefix[i] == '-u':
            require(i + 1 < len(prefix), 'truncated unset argument')
            unset_names.append(prefix[i + 1])
            i += 2
            continue
        require('=' in prefix[i], 'unreviewed launcher prefix')
        key, value = prefix[i].split('=', 1)
        require(key not in assignments, 'duplicate explicit environment assignment')
        assignments[key] = value
        i += 1
    require(assignments == env, 'effective explicit launch environment differs from profile')
    require(len(unset_names) == len(set(unset_names)) and set(unset_names) == set(profile['unset_environment']), 'effective unset policy differs from profile')
    # This known immutable ELF was sealed earlier; hash again during score, after
    # both baseline quality sessions, rather than scan610MB in preparation.
    gold = args.gold.resolve(strict=True)
    gm = json.loads((gold / 'manifest.json').read_text())
    require(gm['engine'] == 'llama.cpp-PR31192' and gm['oracle_op_offload']
            and gm['context'] == 40000 and gm['kv'] == 'f16'
            and gm['cuda_fusion_disabled'] and execution_chunks(gm) == expected_chunks(),
            'requires existing same-geometry F16 E3 code128 gold')
    ids = Path(command[command.index('--ids-file') + 1])
    require(ids.read_bytes() == (gold / 'tokens.i32').read_bytes(), 'candidate input IDs differ from E3 gold')
    require(command[command.index('-m') + 1] == gm['model'], 'candidate model differs from E3 gold')
    seal_path = LOG / 'q2-final-selected-byte-seal-v3.json'
    seal = json.loads(seal_path.read_text())
    segments = next(row for row in seal['resident_E3_phase_review']['segments'] if row['name'] == 'resident_e3_oracle')
    compiled = args.canonical.resolve(strict=True)
    source_before = json.loads((compiled / 'source-before.json').read_text())
    source_proof = {}
    for relative in ('tools/m31/gpu_cached_decode_impl.hpp', 'tools/m31/m31_gpu_experts.cpp', 'tools/m31/m31_engine.cpp'):
        path = compiled / 'source-tree' / relative
        actual = fp(path)
        require(actual['sha256'] == source_before[relative]['sha256'], 'canonical source proof changed')
        source_proof[relative] = actual
    comparison = directory / 'comparison-v2'
    comparison.mkdir(exist_ok=False)
    for name in ('q2_cached_e3_compare.py', 'q2_compat_prepare.py', 'compare_logits.py', 'cache_seed_preflight.py'):
        freeze(ROOT / 'tools/m31' / name, comparison / name)
    write(directory / 'comparison-plan-v2.json', dict(format='m31-q2-cached-e3-code128-gate-v1',
        prepared_only=True, acceptance_eligible=False, job=fp(args.job),
        frozen_scorer=fp(comparison / 'q2_cached_e3_compare.py'), bounded_launcher=fp(directory / 'launch_once.py'), engine=dict(path=str(elf), sha256=GEN_SHA),
        candidate_case_stem=ids.stem, gold=str(gold), gold_header=fp(gold / 'manifest.json'), gold_tokens=fp(gold / 'tokens.i32'),
        gold_segments=segments, prior_gold_byte_seal=fp(seal_path),
        files={key: fp(path) for key, path in small_inputs(directory).items()},
        compiled_source_proof=source_proof,
        required_completed_baseline_jobs=[fp(args.baseline_reference_job), fp(args.baseline_native_job)],
        expected=dict(slots=SLOTS, stride=STRIDE, pool_bytes=SLOTS * STRIDE, seed_bytes=SEED_BYTES,
             seed_primes=1, decode_inputs=32, route_lookups=7296, resident_routes=384, hybrid_forwards=96),
        thresholds=dict(top1=.99, mean_kl=.001, max_kl=.05),
        additional_phase_criterion='Both aggregate128 and actualsingleton tail32 independently meet thresholds; PP96 cannot dilute decode errors. This additional bounded-candidate requirement does not alter earlier baseline criteria.',
        per_layer_residency_proof='Observed384slots + successful384key decode-only prime, rankprefix exactlyall128experts oflayers10/11/12, and compiled no-admission-on-miss native cached code imply the entire pool contains exactly E3. Per-layer all4hit/other54allmiss is a source/full-pool inference, not an observed per-layer counter.',
        stop_policy='One bounded candidate after BOTH baseline quality jobs; no automatic retry/long/perf/quality. KL failure ends this candidate; a pass only enables root to choose further work. No baseline quality acceptance inherited.'))
    plan_path = directory / 'comparison-plan-v2.json'
    plan_path.chmod(0o444)
    seal_path = directory / 'comparison-plan-v2.sha256'
    with seal_path.open('x') as stream:
        stream.write(fp(plan_path)['sha256'] + '\n')
    seal_path.chmod(0o444)
    print(plan_path)


def fields(line):
    return {k: float(v) for k, v in re.findall(r'([a-z_]+)=([0-9]+(?:\.[0-9]+)?)', line)}


def score(args):
    import numpy as np
    directory = args.directory.resolve(strict=True)
    plan_path = directory / 'comparison-plan-v2.json'
    require(fp(plan_path)['sha256'] == (directory / 'comparison-plan-v2.sha256').read_text().strip(), 'comparison plan seal changed')
    plan = json.loads(plan_path.read_text())
    require(fp(Path(__file__))['sha256'] == plan['frozen_scorer']['sha256'], 'scorer differs from frozen pre-trial criterion')
    require(fp(Path(plan['bounded_launcher']['path'])) == plan['bounded_launcher'], 'single-trial launcher changed')
    require(fp(Path(plan['job']['path'])) == plan['job'], 'candidate planned job changed')
    result = completed(Path(plan['job']['path']))
    for descriptor in plan['files'].values():
        require(fp(Path(descriptor['path'])) == descriptor, 'candidate seed/profile file changed')
    validate_seed(directory)
    require(fp(Path(plan['engine']['path']))['sha256'] == plan['engine']['sha256'] == GEN_SHA, 'candidate ELF changed')
    for descriptor in plan['compiled_source_proof'].values():
        require(fp(Path(descriptor['path'])) == descriptor, 'cached source proof changed')
    gold = Path(plan['gold'])
    for name in ('gold_header', 'gold_tokens', 'prior_gold_byte_seal'):
        require(fp(Path(plan[name]['path'])) == plan[name], 'gold provenance changed: ' + name)
    gold_raw = (gold / 'logits.f32').read_bytes()
    cut = plan['gold_segments']['PP96_bytes']
    require(len(gold_raw) == cut + plan['gold_segments']['tail32_bytes'], 'gold logits extent changed')
    require(hashlib.sha256(gold_raw[:cut]).hexdigest() == plan['gold_segments']['PP96_sha256']
            and hashlib.sha256(gold_raw[cut:]).hexdigest() == plan['gold_segments']['tail32_sha256'], 'existing E3 gold payload changed')
    command = result['commands'][0]['argv']
    native = Path(command[command.index('--out') + 1]) / plan['candidate_case_stem']
    nm = json.loads((native / 'manifest.json').read_text())
    require(nm['engine'] == 'xolotl-m31-correctness' and nm['context'] == 40000
            and nm['kv'] == 'f16' and nm['cuda_fusion_disabled']
            and nm.get('cuda_mmq_precision') == 'auto' and not nm.get('m31_decode_fusion_override_requested')
            and not nm.get('m31_decode_fusion_unfenced_requested')
            and execution_chunks(nm) == expected_chunks(), 'candidate actual engine/geometry/fusion differs')
    log = Path(result['commands'][0]['log'])
    text = log.read_text(errors='replace')
    marker_path = directory / 'ONE_ATTEMPT_STARTED.json'
    marker = json.loads(marker_path.read_text())
    require(marker.get('ready_for_one_trial') is True and marker.get('execute') is True
            and marker.get('own_card_worker_inactive') is True, 'bounded launcher did not attest both baseline cleanups before submission')
    require(len(marker.get('baseline_qualities', [])) == 2
            and all(row.get('ready') is True and all(row['checks'].values()) for row in marker['baseline_qualities']),
            'missing baseline completion at actual single-trial launch')
    launch_ns = int(datetime.datetime.fromisoformat(marker['submitted_utc']).timestamp() * 1_000_000_000)
    prerequisites = []
    for descriptor in plan['required_completed_baseline_jobs']:
        baseline_job = Path(descriptor['path'])
        require(fp(baseline_job) == descriptor, 'baseline quality chain job changed')
        completed(baseline_job)
        baseline_result = baseline_job.with_suffix('.result.json')
        require(baseline_result.stat().st_mtime_ns <= launch_ns <= log.stat().st_mtime_ns, 'candidate submission timestamp precedes required baseline completion or follows candidate end')
        prerequisites.append(fp(baseline_result))
    require(text.count('M31 fused CPU MoE: enabled=1') == 1, 'fused CPU wrapper not actually active')
    init = re.findall(r'M31 GPU experts: fixed_slots=(\d+) slot_stride=(\d+) pool_bytes=(\d+) budget=(\d+)', text)
    require(len(init) == 1, 'missing actual slot-capacity receipt')
    slots, stride, pool, budget = map(int, init[0])
    require((slots, stride, pool) == (SLOTS, STRIDE, SLOTS * STRIDE) and budget // stride == SLOTS,
            'actual cache capacity differs from full-pool proof')
    seeds = re.findall(r'M31 decode-only seed: hot=(\d+) upload_bytes=(\d+) seconds=([0-9.]+)', text)
    require(len(seeds) == 1 and tuple(map(int, seeds[0][:2])) == (HOT, SEED_BYTES), 'actual decode-only primer differs')
    hybrid = [fields(line) for line in text.splitlines() if line.startswith('M31 GPU cached hybrid label=forward ')]
    expert = [fields(line) for line in text.splitlines() if line.startswith('M31 GPU expert profile label=forward ')]
    require(len(hybrid) == len(expert) == 33, 'expected exactly1PP+32n1 profile snapshots')
    for index, (h, e) in enumerate(zip(hybrid, expert)):
        require(h['forwards'] == 3 * index and h['route_lookups'] == 228 * index
                and h['resident_routes'] == 12 * index and h['seed_primes'] == int(index > 0)
                and h['seed_bytes'] == (SEED_BYTES if index else 0), 'actual resident coverage counters differ')
        require(e['forwards'] == 57 + 3 * index and e['cached'] == 0 and e['fast'] == 0,
                'unexpected GPU execution path')
        require(e['bytes'] == expert[0]['bytes'] + (SEED_BYTES if index else 0), 'weight uploads/admissions occurred during cached decode')
    metrics = compare_case(gold, native)
    candidate = np.memmap(native / 'logits.f32', dtype='<f4', mode='r', shape=(128, 200064))
    reference = np.frombuffer(gold_raw, dtype='<f4').reshape(128, 200064)
    phases = []
    for name, first, last in (('prefill96', 0, 96), ('decode32', 96, 128)):
        a = np.asarray(reference[first:last], dtype=np.float64)
        b = np.asarray(candidate[first:last], dtype=np.float64)
        maxabs = float(np.abs(a - b).max())
        matches = int(np.count_nonzero(a.argmax(1) == b.argmax(1)))
        a -= a.max(1, keepdims=True); b -= b.max(1, keepdims=True)
        a -= np.log(np.exp(a).sum(1, keepdims=True)); b -= np.log(np.exp(b).sum(1, keepdims=True))
        kl = np.maximum((np.exp(a) * (a - b)).sum(1), 0)
        phases.append(dict(name=name, rows=last - first, top1_matches=matches,
             top1=matches / (last - first), mean_kl=float(kl.mean()), max_kl=float(kl.max()), max_abs=maxabs,
             pass_=matches / (last - first) >= .99 and float(kl.mean()) <= .001 and float(kl.max()) <= .05))
    aggregate_pass = metrics['top1'] >= .99 and metrics['mean_kl'] <= .001 and metrics['max_kl'] <= .05
    report = dict(format='m31-q2-cached-e3-code128-result-v1', acceptance_eligible=False,
        numeric_aggregate=metrics, aggregate_pass=aggregate_pass, phases=phases,
        additional_decode_phase_pass=phases[1]['pass_'], pass_=aggregate_pass and phases[1]['pass_'],
        actual_slots=slots, actual_pool_bytes=pool, seed_profile=seeds, final_hybrid=hybrid[-1],
        resident_coverage_policy=plan['per_layer_residency_proof'],
        plan=fp(plan_path), actual_log=fp(log), single_attempt_start_receipt=fp(marker_path), completed_baseline_quality_results=prerequisites,
        native_header=fp(native / 'manifest.json'), gold_payload_sha256=hashlib.sha256(gold_raw).hexdigest(),
        native_payload_sha256=fp(native / 'logits.f32')['sha256'],
        threshold_policy=plan['additional_phase_criterion'], stop_policy=plan['stop_policy'])
    report['pass'] = report.pop('pass_')
    write(directory / 'strict-compare.json', report)
    print(json.dumps(report, indent=2))
    return 0 if report['pass'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action', required=True)
    prep = subs.add_parser('prepare')
    prep.add_argument('--directory', type=Path, required=True)
    prep.add_argument('--job', type=Path, required=True)
    prep.add_argument('--engine', type=Path, default=LOG / 'q2-compat-final-v1/m31_generate-frozen')
    prep.add_argument('--gold', type=Path, default=LOG / 'q2-best-placement-code128-e3-v1/oracle/code_02_128')
    prep.add_argument('--canonical', type=Path, default=ROOT / 'build-port7-canonical/q2-v1')
    prep.add_argument('--baseline-reference-job', type=Path, default=LOG / 'q2-reference-quality-e3-v1/q2-reference-quality-e3-v1-card-job.json')
    prep.add_argument('--baseline-native-job', type=Path, default=LOG / 'q2-native-quality-v1/q2-native-quality-v1-card-job.json')
    score_parser = subs.add_parser('score')
    score_parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    return globals()[args.action](args)


if __name__ == '__main__':
    raise SystemExit(main())
