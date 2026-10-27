#!/usr/bin/env python3
"""Prepare/replay/audit final fast-tail20 using real Engine.verify and GPU oracle.

Preparation never launches a model. This gate supplements full ordinary gate20;
its tail intentionally uses different target/oracle execution batching.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path('/mnt/ai/Xolotl-m31')
LOG = Path('/mnt/ai/logs/m31-port')
MODEL = '/mnt/ai/models/MiniMax-M3.1-Flash-GGUF/UD-IQ3_S/MiniMax-M3.1-Flash-UD-IQ3_S-00001-of-00005.gguf'


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def fingerprint(path):
    path = path.resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            digest.update(part)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def freeze(source, target, executable=False):
    before = fingerprint(source)
    with source.open('rb') as stream, target.open('xb') as output:
        shutil.copyfileobj(stream, output, 1 << 20)
    target.chmod(0o555 if executable else 0o444)
    if fingerprint(target)['sha256'] != before['sha256']:
        raise ValueError('source changed while freezing ' + str(source))
    return fingerprint(target)


def receipt(path, elf, name):
    value = json.loads(path.read_text())
    if value.get('elf', {}).get(name) != elf['sha256'] or not value.get('source'):
        raise ValueError('receipt does not identify selected ' + name + '/source')
    return fingerprint(path)


def policy(prefix, chunk):
    result, position = [], 0
    while position < prefix:
        n = min(chunk, prefix - position)
        result.append((position, n))
        position += n
    return result


def prepare(args):
    original_profile = fingerprint(args.profile)
    profile = json.loads(args.profile.read_text())
    env = profile['environment']
    unset = profile.get('unset_environment', profile.get('unset_inherited_environment', []))
    if args.diagnostic_input and args.diagnostic_case:
        raise ValueError('diagnostic-input and diagnostic-case are mutually exclusive')
    if env.get('GGML_CUDA_DISABLE_FUSION') != '1':
        raise ValueError('this oracle plan uses accepted no-fusion GPU reference')
    files = ([args.diagnostic_input] if args.diagnostic_input else sorted(args.ids.glob('*.i32')))
    lengths = [path.stat().st_size // 4 for path in files]
    if not args.diagnostic_input and (len(files) != 20 or min(lengths) != 64 or max(lengths) != 3000
            or not {'ru', 'en', 'code', 'math'} <= {path.stem.split('_')[0] for path in files}):
        raise ValueError('requires original20 prompts64..3000 RU/EN/code/math')
    if args.diagnostic_case:
        files = [path for path in files if path.stem == args.diagnostic_case]
        if len(files) != 1:
            raise ValueError('diagnostic case is not in original gate20')
    arm = args.arm or args.directory.name
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{1,140}', arm):
        raise ValueError('arm must be unique and suitable for card job basename')
    diagnostic_long = args.diagnostic_input is not None
    if diagnostic_long and (len(files) != 1 or lengths[0] not in (16384, 32768)
                            or args.tail != 32 or args.chunk < 256):
        raise ValueError('long diagnostic requires one16K/32K input,32 tail and chunk>=256')
    if (args.context < max(lengths) + args.tail or
            (not diagnostic_long and (args.tail != 128 or args.chunk != 2048))):
        raise ValueError('final fast-tail20 requires128 tail,2048 chunk and sufficient context')
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    frozen_profile = freeze(args.profile, directory / 'runtime-profile.json')
    if frozen_profile['sha256'] != original_profile['sha256']:
        raise ValueError('profile changed during preparation')
    ids = directory / 'ids'
    ids.mkdir()
    cases = []
    for source in files:
        if source.stat().st_size % 4:
            raise ValueError('invalid prompt ID bytes')
        frozen = ids / source.name
        freeze(source, frozen)
        cases.append({'name': source.stem, 'prompt_tokens': source.stat().st_size // 4,
                      'ids': fingerprint(frozen)})
    native = freeze(args.engine, directory / 'port7_verify_dump-frozen', True)
    native_receipt = receipt(args.build_receipt, native, 'port7_verify_dump')
    oracle = freeze(args.oracle, directory / 'm31_oracle-frozen', True)
    oracle_receipt = receipt(args.oracle_receipt, oracle, 'm31_oracle')
    freeze(args.build_receipt, directory / 'native-build-receipt.json')
    freeze(args.oracle_receipt, directory / 'oracle-build-receipt.json')
    comparator = directory / 'comparator'
    comparator.mkdir()
    for name in ('port7_verify_gate.py', 'compare_logits.py'):
        freeze(ROOT / 'tools/m31' / name, comparator / name)
    cli = [native['path'], '-m', MODEL, '--backend', 'cuda', '--ctx', str(args.context),
           '--chunk', str(args.chunk), '--threads', str(args.threads), '--arena-gib', '170',
           '--gpu-experts-gib', str(args.cache_gib), '--gpu-experts-min-tokens', '32',
           '--reuse-allocations', '1', '--persistent-cpu-pool', '1', '--cpu-residual', '1',
           '--cpu-decode-cache', '1', '--dense-decode-cache', '1', '--profile', '1',
           '--taps', 'none', '--cases', str(ids), '--out', str(directory / 'native'),
           '--generate', str(args.tail), '--require-bitwise', '0']
    prefix = json.loads(args.prefix_json.read_text()) if args.prefix_json else []
    if not isinstance(prefix, list) or not all(isinstance(item, str) for item in prefix):
        raise ValueError('prefix-json must contain a reviewed/frozen launcher argv list')
    if profile.get('cache_seed', {}).get('hot', 0) and not prefix:
        raise ValueError('ranked cache profile requires its reviewed/frozen seed launcher prefix')
    command = ['/usr/bin/env', *[item for key in unset for item in ('-u', key)],
               *[key + '=' + value for key, value in sorted(env.items())], *prefix, *cli]
    job = directory / (arm + '-native-card-job.json')
    write(job, {'preset': 'm31-xolotl-check', 'measurement': False, 'commands': [command]})
    write(directory / 'manifest.json', {'format': 'm31-port7-verify-tail20-plan-v1',
          'arm': arm, 'expected_cases': len(cases), 'acceptance_eligible': len(cases) == 20,
          'explicit_oracle_replay': diagnostic_long,
          'native': native, 'build_receipt': native_receipt, 'oracle': oracle,
          'oracle_build_receipt': oracle_receipt, 'profile': frozen_profile,
          'environment': env, 'unset_environment': unset, 'threads': args.threads,
          'cache_gib': args.cache_gib, 'context': args.context, 'chunk': args.chunk,
          'tail': args.tail, 'rows_per_case': args.tail + 1, 'cases': cases,
          'prefix_launcher': prefix, 'prefix_launcher_receipt': fingerprint(args.prefix_json) if args.prefix_json else None,
          'thresholds': {'top1_min': .99, 'mean_kl_max': .001, 'max_kl_max': .05},
          'internal_greedy': 'Every ordinary/verified argmax must match; bit equality/features are reported separately.',
          'position_contract': 'Row0 anchorP-1. Remaining rows are logits after feeding continuation IDs atP..P+tail-1; final row predicts the next un-emitted token.',
          'scope': 'Final actual Engine.verify/partial-commit/rewind numeric gate. Supplements ordinary fullgate20 and actual speculative raw greedy checks; not speed or GSM/tools.'})
    print(job)


def replay(args):
    directory = args.directory.resolve()
    manifest = json.loads((directory / 'manifest.json').read_text())
    selected, native_result = completed(directory, manifest, 'native')
    runtime_log = Path(selected['log']).read_text(errors='replace')
    endings = list(re.finditer(r'verify-tail case=(\S+) prompt=(\d+) committed=(\d+) calls=(\d+) exact=([01])', runtime_log))
    if [match[1] for match in endings] != [case['name'] for case in manifest['cases']]:
        raise ValueError('native runtime log does not contain the completed case sequence')
    ids_dir = directory / 'oracle-ids'
    ids_dir.mkdir()
    plans = []
    previous = 0
    for case, ending in zip(manifest['cases'], endings):
        path = directory / 'native' / case['name']
        metadata = json.loads((path / 'verify-metadata.json').read_text())
        chunks = [(row['position'], row['tokens']) for row in metadata['prefix_actual_chunks']]
        cursor = 0
        for position, n in chunks:
            if position != cursor or not 0 < n <= manifest['chunk']:
                raise ValueError('native prefix has noncontiguous/out-of-range chunks')
            cursor += n
        if cursor != case['prompt_tokens']:
            raise ValueError('native prefix recipe does not cover its prompt')
        if (not manifest.get('explicit_oracle_replay')
                and chunks != policy(case['prompt_tokens'], manifest['chunk'])):
            raise ValueError('native actual prefix differs from fixed oracle policy; explicit replay implementation required')
        observed = [tuple(map(int, item)) for item in re.findall(
            r'M31 prefill-plan pos=(\d+) requested=\d+ planned=(\d+) keys=(\d+)', runtime_log[previous:ending.start()])]
        declared = [(row['position'], row['tokens'], row['nk']) for row in metadata['prefix_actual_chunks']]
        if observed != declared:
            raise ValueError('padding-derived prefix nk/chunks do not match actual runtime planner log')
        previous = ending.end()
        combined = path / 'tokens.i32'
        if combined.stat().st_size != (case['prompt_tokens'] + manifest['tail']) * 4:
            raise ValueError('combined prompt/greedy continuation byte length mismatch')
        (ids_dir / (case['name'] + '.i32')).symlink_to(combined.resolve())
        plans.append({'case': case['name'], 'combined_ids': fingerprint(combined),
                      'actual_prefix_chunks': metadata['prefix_actual_chunks'],
                      'oracle_policy': 'Replay actual native PP boundaries; then singleton continuation inputs.'})
    oracle = [str(directory / 'm31_oracle-frozen'), '-m', MODEL, '--backend', 'cuda',
              '--ctx', str(manifest['context']), '--chunk', str(manifest['chunk']),
              '--threads', str(manifest['threads']), '--decode-tail', str(manifest['tail']),
              '--head-tail', '1', '--rows', 'last:' + str(manifest['tail'] + 1),
              '--oracle-op-offload', '1', '--cases', str(ids_dir), '--out', str(directory / 'oracle')]
    job = directory / (manifest['arm'] + '-oracle-card-job.json')
    oracle_environment = ['-u', 'M31_ORACLE_CHUNK_PLAN']
    if manifest.get('explicit_oracle_replay'):
        if len(manifest['cases']) != 1:
            raise ValueError('explicit oracle replay requires exactly one input')
        replay_path = directory / 'oracle-chunk-plan.tsv'
        chunks = [(row['position'], row['tokens']) for row in plans[0]['actual_prefix_chunks']]
        p = manifest['cases'][0]['prompt_tokens']
        chunks += [(p + i, 1) for i in range(manifest['tail'])]
        with replay_path.open('x') as stream:
            for position, n in chunks:
                stream.write(f'{position}\t{n}\n')
        plans[0]['chunk_plan'] = fingerprint(replay_path)
        oracle_environment = ['M31_ORACLE_CHUNK_PLAN=' + str(replay_path)]
    write(job, {'preset': 'm31-xolotl-check', 'measurement': False, 'commands': [
        ['/usr/bin/env', '-u', 'CUDA_DEVICE_MAX_CONNECTIONS', '-u', 'CUDA_DEVICE_MAX_COPY_CONNECTIONS',
         *oracle_environment, 'GGML_CUDA_DISABLE_FUSION=1', *oracle]]})
    write(directory / 'replay-receipt.json', {'format': 'm31-port7-verify-tail-replay-v1',
          'native_atomic_result': native_result, 'native_log': fingerprint(Path(selected['log'])), 'plans': plans})
    print(job)
    print(json.dumps({'after_oracle_rc0_compare_argv': [sys.executable,
          str(directory / 'comparator/port7_verify_gate.py'), 'compare', '--directory', str(directory)]}))


def completed(directory, plan, role, capacity=None):
    job_path = (Path(capacity['job']) if capacity and role == 'oracle' else
                directory / (plan['arm'] + '-' + role + '-card-job.json'))
    job = json.loads(job_path.read_text())
    result_path = job_path.with_suffix('.result.json')
    result = json.loads(result_path.read_text())
    selected = [row for row in result['commands'] if row['argv'] == job['commands'][0]]
    if (len(selected) != 1 or type(selected[0].get('returncode')) is not int
            or selected[0]['returncode'] != 0):
        raise ValueError(role + ' selected frozen command did not complete with rc0')
    binary = plan['native'] if role == 'native' else plan['oracle']
    if fingerprint(Path(binary['path']))['sha256'] != binary['sha256']:
        raise ValueError(role + ' frozen executable changed')
    return selected[0], fingerprint(result_path)


def capacity(args):
    directory = args.directory.resolve()
    plan = json.loads((directory / 'manifest.json').read_text())
    if not args.authorized or plan['acceptance_eligible'] or len(plan['cases']) != 1:
        raise ValueError('capacity equivalence requires explicit authorization and selected single long case')
    p = plan['cases'][0]['prompt_tokens']
    if p not in (16384, 32768) or plan['tail'] != 32 or not plan.get('explicit_oracle_replay'):
        raise ValueError('capacity equivalence is limited to prepared long16K/32K tail32')
    completed(directory, plan, 'native')
    original = directory / (plan['arm'] + '-oracle-card-job.json')
    command = json.loads(original.read_text())['commands'][0].copy()
    physical = p + plan['tail']
    command[command.index('--ctx') + 1] = str(physical)
    output = directory / 'oracle-capacity-v1'
    command[command.index('--out') + 1] = str(output)
    command.insert(command.index(plan['oracle']['path']), 'M31_ORACLE_CAPACITY_EQUIVALENCE=1')
    job = directory / (plan['arm'] + '-oracle-capacity-v1-card-job.json')
    write(job, {'preset': 'm31-xolotl-check', 'measurement': False, 'commands': [command]})
    write(directory / 'capacity-equivalence-replay-v1.json', {
        'authorized': True, 'authorization': args.authorized, 'job': str(job),
        'job_receipt': fingerprint(job), 'original_job': fingerprint(original),
        'oracle_output': str(output), 'physical_context_requested': physical,
        'physical_context_expected': ((physical + 255) // 256) * 256,
        'native_context_requested': plan['context'],
        'policy': 'Port6 selected-long capacity equivalence, explicitly reaffirmed by coordinator. Require runtime contiguous KV, identical padded active keys and trained linear RoPE. Original headers are preserved.'})
    print(job)


def capacity_trace(reference, header, expected, native_context, descriptor):
    physical = descriptor['physical_context_requested']
    trace_path = reference / 'capacity-equivalence.jsonl'
    trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
    if (descriptor.get('authorized') is not True or header['context'] != physical
            or len(trace) != len(expected)):
        raise ValueError('capacity context/authorization/trace coverage mismatch')
    for row, (position, n) in zip(trace, expected):
        end = position + n
        if (row['position'] != position or row['tokens'] != n or row['seq_min'] != 0
                or row['seq_max'] != end - 1
                or row['active_keys_padded_from_verified_positions'] != ((end + 255) // 256) * 256
                or row['physical_context_requested'] != physical
                or row['physical_context_actual'] != descriptor['physical_context_expected']
                or row['rope_ext_factor'] != 0 or row['rope_orig_context'] != 1048576
                or row['rope_frequency_scale'] != 1):
            raise ValueError('capacity trace does not prove contiguous KV/key padding/trained RoPE')
    return {'authorized': True, 'native_context_requested': native_context,
            'oracle_context_requested': physical, 'oracle_context_actual': trace[0]['physical_context_actual'],
            'runtime_trace': fingerprint(trace_path), 'runtime_chunks': trace,
            'authorization': descriptor['authorization']}


def metrics(a, b, positions):
    import numpy as np
    count = same = 0
    total = maximum = max_abs = 0.0
    for start in range(0, len(positions), 8):
        x, y = np.asarray(a[start:start + 8], dtype=np.float64), np.asarray(b[start:start + 8], dtype=np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('nonfinite raw logits')
        same += int((x.argmax(1) == y.argmax(1)).sum())
        max_abs = max(max_abs, float(np.abs(x - y).max()))
        x -= x.max(1, keepdims=True)
        y -= y.max(1, keepdims=True)
        x -= np.log(np.exp(x).sum(1, keepdims=True))
        y -= np.log(np.exp(y).sum(1, keepdims=True))
        kl = (np.exp(x) * (x - y)).sum(1)
        if not np.isfinite(kl).all() or float(kl.min()) < -1e-10:
            raise ValueError('invalid KL calculation')
        kl = np.maximum(kl, 0)
        total += float(kl.sum())
        maximum = max(maximum, float(kl.max()))
        count += len(kl)
    return {'rows': count, 'top1_matches': same, 'top1': same / count,
            'mean_kl': total / count, 'max_kl': maximum, 'max_abs': max_abs}


def compare(args):
    import numpy as np
    import compare_logits
    directory = args.directory.resolve()
    plan = json.loads((directory / 'manifest.json').read_text())
    capacity_descriptor = json.loads(args.capacity_replay.read_text()) if args.capacity_replay else None
    if capacity_descriptor:
        if fingerprint(Path(capacity_descriptor['job'])) != capacity_descriptor['job_receipt']:
            raise ValueError('capacity retry job changed after preparation')
    native_run, native_result = completed(directory, plan, 'native')
    oracle_run, oracle_result = completed(directory, plan, 'oracle', capacity_descriptor)
    records = []
    for case in plan['cases']:
        path = directory / 'native' / case['name']
        reference = ((Path(capacity_descriptor['oracle_output']) if capacity_descriptor else directory / 'oracle')
                     / case['name'])
        a, av = compare_logits.load_case(reference)
        b, bv = compare_logits.load_case(path)
        meta = json.loads((path / 'verify-metadata.json').read_text())
        p, tail = case['prompt_tokens'], plan['tail']
        positions = list(range(p - 1, p + tail))
        if (a['engine'] != 'llama.cpp-PR31192' or not a.get('oracle_op_offload')
                or not compare_logits.full_last_layer_before_crop(a)
                or b['engine'] != 'xolotl-m31-verify-tail-gate'):
            raise ValueError('incorrect oracle/fast-target provenance or last-layer crop policy')
        if a.get('backend') != 'cuda' or b.get('backend') != 'cuda':
            raise ValueError('fast-tail oracle and target must both use CUDA')
        if b['context'] != plan['context'] or (not capacity_descriptor and a['context'] != plan['context']):
            raise ValueError('unapproved native/oracle requested context difference')
        for key in ('model', 'tokens', 'vocab', 'positions', 'kv', 'head_tail'):
            if a[key] != b[key]:
                raise ValueError('oracle/native header mismatch ' + key)
        if a['positions'] != positions or a['tokens'] != p + tail or a['head_tail'] != 1:
            raise ValueError('anchor/tail raw position contract mismatch')
        raw_ids = (path / 'tokens.i32').read_bytes()
        if raw_ids != (reference / 'tokens.i32').read_bytes():
            raise ValueError('oracle and target teacher-forced IDs differ')
        if raw_ids[:p * 4] != Path(case['ids']['path']).read_bytes():
            raise ValueError('prompt IDs differ from prepared frozen input')
        prefix = [(row['position'], row['tokens']) for row in meta['prefix_actual_chunks']]
        oracle_chunks = compare_logits.execution_chunks(a)
        expected = prefix + [(p + i, 1) for i in range(tail)]
        if (not plan.get('explicit_oracle_replay') and prefix != policy(p, plan['chunk'])) or oracle_chunks != expected:
            raise ValueError('actual native PP/oracle singleton geometry mismatch')
        equivalence = (capacity_trace(reference, a, expected, b['context'], capacity_descriptor)
                       if capacity_descriptor else None)
        if (meta['anchor_position'], meta['tail_start'], meta['tail_rows']) != (p - 1, p, tail):
            raise ValueError('fast metadata position contract mismatch')
        cursor = p
        sizes, partial, full = set(), 0, 0
        for call in meta['verify_calls']:
            n, commit = call['input_tokens'], call['committed']
            if (call['position'] != cursor or not 1 <= commit <= n <= 8
                    or cursor + n > p + tail or call['rewound_to'] != cursor + commit):
                raise ValueError('invalid actual verify/commit/rewind trace')
            sizes.add(n)
            partial += commit < n
            full += commit == n
            cursor += commit
        if cursor != p + tail or not {1, 2, 3, 4, 7, 8} <= sizes or not partial or not full:
            raise ValueError('fast verify trace has incomplete coverage')
        shape = (tail + 1, a['vocab'])
        serial_path = path / meta['serial_logits_file']
        if serial_path.stat().st_size != int(np.prod(shape)) * 4:
            raise ValueError('serial reference logits length mismatch')
        serial = np.memmap(serial_path, dtype='<f4', mode='r', shape=shape)
        ids = np.frombuffer(raw_ids, dtype='<i4')
        if not np.array_equal(serial[:tail].argmax(1), ids[p:]):
            raise ValueError('continuation IDs are not ordinary serial greedy decisions, including anchor')
        internal = metrics(serial, bv, positions)
        if internal['top1'] != 1 or any(value['nonfinite'] for value in meta['comparisons'].values()):
            raise ValueError('verified argmax/finite invariants differ from ordinary serial')
        if meta['comparisons']['all_verified_logits']['top1_different']:
            raise ValueError('uncommitted verify row differs from ordinary serial argmax')
        feature_shape = (tail + 1, meta['feature_dim'])
        if meta['feature_layers'] != [4, 18, 32, 46, 60] or not meta['feature_rows_include_anchor']:
            raise ValueError('feature capture branch does not match actual M3 configuration')
        feature_arrays = []
        for key in ('serial_features_file', 'verified_features_file'):
            feature_file = path / meta[key]
            if feature_file.stat().st_size != int(np.prod(feature_shape)) * 4:
                raise ValueError('feature artifact length mismatch')
            data = np.memmap(feature_file, dtype='<f4', mode='r', shape=feature_shape)
            if not np.isfinite(data).all():
                raise ValueError('nonfinite captured feature artifact')
            feature_arrays.append(data)
        feature_bitwise = np.array_equal(feature_arrays[0].view('<u4'), feature_arrays[1].view('<u4'))
        record = {'case': case['name'], 'prompt_tokens': p, 'anchor': metrics(av[:1], bv[:1], positions[:1]),
                  'tail': metrics(av[1:], bv[1:], positions[1:]), 'internal': internal,
                  'verify_calls': len(meta['verify_calls']), 'partial_commits': partial,
                  'committed_feature_files_bitwise': bool(feature_bitwise),
                  'bitwise_parity': meta['bitwise_parity_pass'], 'feature_comparisons': meta['comparisons'],
                  'raw_ids': fingerprint(path / 'tokens.i32'), 'metadata': fingerprint(path / 'verify-metadata.json')}
        if equivalence:
            record['capacity_equivalence'] = equivalence
        records.append(record)
    if len(records) != plan['expected_cases']:
        raise ValueError('fasttail case set is incomplete')
    rows = sum(row['tail']['rows'] for row in records)
    top1 = sum(row['tail']['top1_matches'] for row in records) / rows
    mean = sum(row['tail']['mean_kl'] * row['tail']['rows'] for row in records) / rows
    maximum = max(row['tail']['max_kl'] for row in records)
    passed = top1 >= .99 and mean <= .001 and maximum <= .05
    passed &= all(row['anchor']['top1'] >= .99 and row['anchor']['mean_kl'] <= .001
                  and row['anchor']['max_kl'] <= .05 for row in records)
    result = {'format': ('m31-port7-verify-tail20-gate-v1' if plan['acceptance_eligible']
                        else 'm31-port7-verify-tail-diagnostic-v1'),
              'acceptance_eligible': plan['acceptance_eligible'], 'pass': bool(passed), 'cases': records,
              'tail_rows': rows, 'anchor_rows': len(records), 'top1': top1, 'mean_kl': mean,
              'max_kl': maximum, 'build_receipt': plan['build_receipt'],
              'environment': plan['environment'], 'unset_environment': plan['unset_environment'],
              'atomic_results': {'native': native_result, 'oracle': oracle_result},
              'audit_sources': {'runner': fingerprint(Path(__file__)),
                                'metrics': fingerprint(Path(__file__).with_name('compare_logits.py'))},
              'scope': 'Actual Engine.verify numeric gate with partial commit/rewind. Selected long cases are diagnostics, not full long-context acceptance. Does not replace ordinary fullgate20, full sidecar acceptance or final GSM/tools.'}
    write(directory / 'compare.json', result)
    print(json.dumps({key: result[key] for key in ('pass', 'tail_rows', 'anchor_rows', 'top1', 'mean_kl', 'max_kl')}, indent=2))
    return 0 if passed else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    prep = modes.add_parser('prepare')
    for name in ('directory', 'engine', 'build-receipt', 'oracle', 'oracle-receipt', 'profile'):
        prep.add_argument('--' + name, type=Path, required=True)
    prep.add_argument('--ids', type=Path, default=LOG / 'ids-v1')
    prep.add_argument('--prefix-json', type=Path)
    prep.add_argument('--arm')
    prep.add_argument('--diagnostic-case', help='single original gate20 case; never final acceptance')
    prep.add_argument('--diagnostic-input', type=Path, help='single frozen16K/32K prompt for long verify diagnostic')
    prep.add_argument('--context', type=int, default=4096)
    prep.add_argument('--chunk', type=int, default=2048)
    prep.add_argument('--tail', type=int, default=128)
    prep.add_argument('--threads', type=int, default=80)
    prep.add_argument('--cache-gib', type=float, default=7)
    for mode in ('replay', 'compare'):
        sub = modes.add_parser(mode)
        sub.add_argument('--directory', type=Path, required=True)
        if mode == 'compare':
            sub.add_argument('--capacity-replay', type=Path)
    cap = modes.add_parser('capacity')
    cap.add_argument('--directory', type=Path, required=True)
    cap.add_argument('--authorized', required=True, help='explicit coordinator authorization reference/scope')
    args = parser.parse_args()
    return {'prepare': prepare, 'replay': replay, 'compare': compare, 'capacity': capacity}[args.mode](args) or 0


if __name__ == '__main__':
    raise SystemExit(main())
