#!/usr/bin/env python3
"""Compare archived greedy IDs only after mandatory target/source/geometry checks.

Exit0 means equal,2 means valid greedy mismatch,3 means invalid reference.
Logs must contain actual M31 forward and prefill-plan lines. Receipts are read
from each job's adjacent manifest; frozen inputs and receipts are never edited.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct

PREFLIGHT_KEYS = ('executable_sha256', 'source_receipt_sha256', 'source_engine_sha256',
    'model_path', 'precision', 'prompt_ids_sha256', 'prompt_tokens', 'actual_prefix_chunks',
    'target_settings', 'explicit_environment', 'unset_environment')


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def once(argv, name):
    indices = [i for i, value in enumerate(argv) if value == name]
    if len(indices) != 1 or indices[0] + 1 >= len(argv):
        raise ValueError('missing or repeated target argument ' + name)
    return argv[indices[0] + 1]


def describe(job_path, log_path, ids_path, profile, command_index=0, traversal_index=0,
             output_case=None, declared_receipt=None, result_path=None):
    manifest_path = job_path.parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    command = json.loads(job_path.read_text())['commands'][command_index]
    result_path = job_path.with_suffix('.result.json') if result_path is None else result_path
    atomic_result = json.loads(result_path.read_text())
    completed = [entry for entry in atomic_result['commands'] if entry['argv'] == command]
    if len(completed) != 1 or completed[0].get('returncode') != 0:
        raise ValueError('selected command is not uniquely completed with rc0 in atomic job result')
    if Path(completed[0]['log']).resolve() != log_path.resolve():
        raise ValueError('selected log is not bound to command in atomic job result')
    indices = [i for i, value in enumerate(command) if Path(value).name == 'm31_dspark_generate']
    if len(indices) != 1:
        raise ValueError('expected one explicit m31_dspark_generate executable')
    index = indices[0]
    prefix, argv = command[:index], command[index:]
    executable = Path(argv[0]).resolve(strict=True)
    exe_sha = digest(executable)
    if exe_sha != manifest['executable_sha256']:
        raise ValueError('executable differs from run manifest')
    receipt_path = Path(manifest['source_receipt']).resolve(strict=True)
    receipt = json.loads(receipt_path.read_text())
    receipt_sha = digest(receipt_path)
    if manifest.get('source_receipt_sha256', receipt_sha) != receipt_sha:
        raise ValueError('build receipt differs from run manifest')
    if receipt['elf']['m31_dspark_generate'] != exe_sha:
        raise ValueError('build receipt does not identify selected executable')
    source = receipt_path.parent / 'source/m31_engine.cpp'
    if digest(source) != receipt['source']['m31_engine.cpp']:
        raise ValueError('frozen Engine source differs from build receipt')
    driver_source = receipt_path.parent / 'source/m31_dspark_generate.cpp'
    if digest(driver_source) != receipt['source']['m31_dspark_generate.cpp']:
        raise ValueError('frozen driver source differs from build receipt')
    text = source.read_text()
    # Current M31 has fixed F16 main/index KV, no runtime precision override.
    for role in ('k', 'v', 'index'):
        if not re.search(r'kv\.' + role + r'\s*=\s*ggml_new_tensor_\w+\([^;\n]*GGML_TYPE_F16', text):
            raise ValueError('unsupported or unproven KV precision for ' + role)
    environment = {}
    for value in prefix:
        match = re.fullmatch(r'([A-Za-z_][A-Za-z0-9_]*)=(.*)', value, flags=re.S)
        if match:
            environment[match[1]] = match[2]
    if not set(profile['environment']) <= set(environment):
        raise ValueError('target profile is not explicitly set by job')
    unset = sorted({prefix[i + 1] for i, item in enumerate(prefix[:-1]) if item == '-u'})
    keys = ('--backend', '--ctx', '--chunk', '--threads', '--arena-gib', '--gpu-experts-gib',
            '--gpu-experts-min-tokens', '--reuse-allocations', '--persistent-cpu-pool',
            '--cpu-residual', '--cpu-decode-cache', '--dense-decode-cache', '--generate')
    config = {name: once(argv, name) for name in keys}
    if config['--backend'] != 'cuda':
        raise ValueError('this GPU speculative comparison requires explicit CUDA target backend')
    mode = once(argv, '--mode')
    model = Path(once(argv, '--model' if '--model' in argv else '-m')).resolve(strict=True)
    if '--ids-file' in argv:
        prompt = Path(once(argv, '--ids-file')).resolve(strict=True)
    else:
        cases = sorted(Path(once(argv, '--cases')).glob('*.i32'))
        if len(cases) != 1:
            raise ValueError('this comparison requires exactly one prompt case')
        prompt = cases[0].resolve(strict=True)
    size = prompt.stat().st_size
    if not size or size % 4:
        raise ValueError('prompt IDs must be nonempty little-endian int32')
    prompt_sha = digest(prompt)
    if prompt_sha != manifest['case_sha256']:
        raise ValueError('actual prompt IDs differ from run manifest')
    prompt_tokens = size // 4
    log = log_path.read_text(errors='replace')
    plans, traversals = {}, []
    chunks, position = [], 0
    for line in log.splitlines():
        plan = re.search(r'M31 prefill-plan pos=(\d+) requested=\d+ planned=(\d+) keys=(\d+)', line)
        if plan:
            pos, n, nk = map(int, plan.groups())
            plans[pos, n] = nk
        forward = re.search(r'M31 forward pos=(\d+) n=(\d+)', line)
        if not forward:
            continue
        pos, n = map(int, forward.groups())
        if pos >= prompt_tokens:
            continue
        if pos != position or n < 1 or pos + n > prompt_tokens:
            raise ValueError('actual prompt chunks are not a single contiguous traversal')
        if (pos, n) not in plans or plans[pos, n] < pos + n:
            raise ValueError('actual nk is absent or invalid for a prompt chunk')
        chunks.append({'position': pos, 'tokens': n, 'nk': plans[pos, n]})
        position += n
        if position == prompt_tokens:
            traversals.append(chunks)
            chunks, position = [], 0
            plans = {}
    if position or chunks or not 0 <= traversal_index < len(traversals):
        raise ValueError('runtime log does not cover the full prompt')
    chunks = traversals[traversal_index]
    output_case = prompt.stem if output_case is None else output_case
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', output_case):
        raise ValueError('invalid output case label')
    if declared_receipt is not None:
        expected_chunks = [[row['position'], row['tokens'], row['nk']] for row in chunks]
        if (declared_receipt['case'] != output_case or declared_receipt['input_case'] != prompt.stem
                or declared_receipt['context'] != int(config['--ctx'])
                or declared_receipt['max_chunk'] != int(config['--chunk'])
                or declared_receipt['chunks'] != expected_chunks):
            raise ValueError('structured arm receipt contradicts actual log/config/input case')
    expected_output = Path(once(argv, '--out')) / (output_case + '.generated.i32')
    if ids_path.resolve() != expected_output.resolve():
        raise ValueError('selected generated IDs are not the output bound to this command/case')
    if not ids_path.is_file():
        raise ValueError('generated IDs are absent')
    summary_path = Path(once(argv, '--out')) / 'summary.jsonl'
    summaries = [json.loads(line) for line in summary_path.read_text().splitlines()]
    if '--sweep' in argv and declared_receipt is None:
        raise ValueError('a sweep arm requires an explicit structured prefill receipt')
    if declared_receipt is not None:
        arm_receipts = [row for row in summaries if row.get('mode') == 'prefill_receipt']
        matches = [i for i, row in enumerate(arm_receipts) if row.get('case') == output_case]
        if (len(matches) != 1 or matches[0] != traversal_index
                or arm_receipts[matches[0]] != declared_receipt):
            raise ValueError('output case is not bound to selected traversal in actual summary order')
    serial_records = [row for row in summaries if row.get('case') == output_case
                      and row.get('mode') == 'serial' and row.get('tokens') == int(config['--generate'])]
    return {'executable_sha256': exe_sha, 'source_receipt_sha256': receipt_sha,
            'source_engine_sha256': digest(source), 'source_driver_sha256': digest(driver_source),
            'model_path': str(model), 'mode': mode, 'output_directory': str(Path(once(argv, '--out')).resolve()),
            'input_case': prompt.stem, 'sweep_command': '--sweep' in argv,
            'source_receipt_manifest_bound': 'source_receipt_sha256' in manifest,
            'environment_scope': 'Explicit job assignments/unsets; inherited runtime environment is not attested by this legacy run format.',
            'serial_artifact_records': serial_records, 'summary_sha256': digest(summary_path),
            'precision': {'main_kv': 'F16', 'index_kv': 'F16',
                          'evidence': 'Explicit KV tensor types in hash-verified frozen Engine source.'},
            'prompt_ids_sha256': prompt_sha, 'prompt_tokens': prompt_tokens,
            'actual_prefix_chunks': chunks, 'target_settings': config,
            'selected_command': command_index, 'selected_traversal': traversal_index,
            'selected_artifact_case': output_case,
            'observed_traversals': len(traversals),
            'explicit_environment': environment, 'unset_environment': unset,
            'generated_path': str(ids_path.resolve()), 'log_sha256': digest(log_path),
            'atomic_result_path': str(result_path.resolve()), 'atomic_result_sha256': digest(result_path),
            'job_sha256': digest(job_path), 'manifest_sha256': digest(manifest_path)}


def validate_roles(reference, candidate):
    if reference['mode'] not in ('serial', 'parity') or len(reference['serial_artifact_records']) != 1:
        raise ValueError('reference is not an explicitly recorded ordinary serial-generated artifact')
    if reference['selected_traversal'] != 0:
        raise ValueError('ordinary serial artifact must use its first prompt traversal')
    if candidate['mode'] not in ('spec', 'sweep'):
        raise ValueError('candidate is not an actual speculative/sweep command')
    if (reference['output_directory'] == candidate['output_directory']
            or reference['generated_path'] == candidate['generated_path']):
        raise ValueError('reference and speculative output artifacts must be distinct')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference-job', 'candidate-job', 'reference-log', 'candidate-log',
                 'reference-ids', 'candidate-ids', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--profile', type=Path, default=Path(__file__).with_name('profiles') / 'native-gpu.json')
    parser.add_argument('--reference-command', type=int, default=0)
    parser.add_argument('--candidate-command', type=int, default=0)
    parser.add_argument('--reference-traversal', type=int, default=0)
    parser.add_argument('--candidate-traversal', type=int, default=0)
    parser.add_argument('--reference-result', type=Path, help='explicit outer bundle result when source command ran inside a frozen wrapper')
    parser.add_argument('--candidate-result', type=Path, help='explicit outer bundle result when source command ran inside a frozen wrapper')
    args = parser.parse_args()
    result = {'format': 'm31-port7-strict-greedy-compare-v1', 'comparison_valid': False,
              'status': 'invalid_reference', 'greedy_equal': None, 'score_computed': False,
              'scope': 'Selected deterministic greedy sequence only; not full model quality acceptance.'}
    code = 3
    try:
        profile = json.loads(args.profile.read_text())
        reference = describe(args.reference_job, args.reference_log, args.reference_ids, profile,
                             args.reference_command, args.reference_traversal, result_path=args.reference_result)
        candidate = describe(args.candidate_job, args.candidate_log, args.candidate_ids, profile,
                             args.candidate_command, args.candidate_traversal, result_path=args.candidate_result)
        validate_roles(reference, candidate)
        checks = {key: reference[key] == candidate[key] for key in PREFLIGHT_KEYS}
        result.update(reference=reference, candidate=candidate, preflight=checks)
        if not all(checks.values()):
            result['reason'] = 'Mandatory preflight mismatch: ' + ', '.join(key for key, ok in checks.items() if not ok)
        else:
            expected = int(reference['target_settings']['--generate'])
            a, b = args.reference_ids.read_bytes(), args.candidate_ids.read_bytes()
            if len(a) != 4 * expected or len(b) != 4 * expected:
                raise ValueError('generated ID lengths differ from required generation length')
            x, y = struct.unpack('<' + 'i' * expected, a), struct.unpack('<' + 'i' * expected, b)
            different = [i for i, (left, right) in enumerate(zip(x, y)) if left != right]
            result.update(comparison_valid=True, score_computed=True, greedy_equal=not different,
                          status='pass' if not different else 'greedy_mismatch', tokens=expected,
                          different=len(different), first_different=different[0] if different else None,
                          reference_ids_sha256=hashlib.sha256(a).hexdigest(),
                          candidate_ids_sha256=hashlib.sha256(b).hexdigest())
            code = 2 if different else 0
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        result['reason'] = str(error)
    result['comparator_sha256'] = digest(Path(__file__))
    with args.out.open('x') as output:
        json.dump(result, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps({key: result.get(key) for key in ('status', 'comparison_valid', 'score_computed',
          'greedy_equal', 'tokens', 'different', 'first_different', 'reason')}, ensure_ascii=False, indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
