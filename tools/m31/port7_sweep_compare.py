#!/usr/bin/env python3
"""Audit every completed arm of a one-prompt DSpark sweep before comparing IDs."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import struct

import port7_greedy_compare as greedy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference-job', 'reference-log', 'reference-ids', 'candidate-job',
                 'candidate-log', 'summary', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-arms', type=int, default=20)
    parser.add_argument('--reference-result', type=Path)
    parser.add_argument('--candidate-result', type=Path)
    parser.add_argument('--reference-command', type=int, default=0)
    parser.add_argument('--candidate-command', type=int, default=0)
    parser.add_argument('--profile', type=Path, default=Path(__file__).with_name('profiles') / 'native-gpu.json')
    args = parser.parse_args()
    try:
        return compare(args)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        result = {'format': 'm31-port7-sweep-greedy-audit-v1', 'complete': False,
                  'comparison_valid': False, 'pass': False, 'score_computed': False,
                  'status': 'invalid_reference', 'reason': str(error)}
        with args.out.open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
        print(json.dumps(result, indent=2))
        return 3


def compare(args):
    # These inputs must be completed/immutable. Recheck stat identity on cache hits
    # so an accidental change invalidates the cached fingerprint.
    uncached_digest = greedy.digest

    @lru_cache(maxsize=128)
    def cached(path, size, mtime_ns, inode):
        return uncached_digest(Path(path))

    def digest(path):
        path = path.resolve(strict=True)
        stat = path.stat()
        return cached(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ino)

    greedy.digest = digest
    profile = json.loads(args.profile.read_text())
    candidate_job = json.loads(args.candidate_job.read_text())
    command = candidate_job['commands'][args.candidate_command]
    index = next(i for i, item in enumerate(command) if Path(item).name == 'm31_dspark_generate')
    output_dir = Path(greedy.once(command[index:], '--out'))
    if args.summary.resolve() != (output_dir / 'summary.jsonl').resolve():
        raise ValueError('provided summary is not bound to selected command output directory')
    declared_arms = greedy.once(command[index:], '--sweep').split(',')
    if len(declared_arms) != args.expected_arms:
        raise ValueError('expected arm count contradicts selected command sweep grid')
    rows = [json.loads(line) for line in args.summary.read_text().splitlines()]
    receipts = [row for row in rows if row.get('mode') == 'prefill_receipt']
    spec_rows = [row for row in rows if row.get('mode') == 'spec']
    specs = {row['case']: row for row in spec_rows}
    if len(specs) != len(spec_rows):
        raise ValueError('duplicate spec case in selected command summary')
    reference = greedy.describe(args.reference_job, args.reference_log, args.reference_ids, profile,
                                 command_index=args.reference_command,
                                 result_path=args.reference_result)
    expected = int(reference['target_settings']['--generate'])
    raw_reference = args.reference_ids.read_bytes()
    if len(raw_reference) != expected * 4:
        raise ValueError('reference raw length differs from declared generation length')
    arms = []
    seen = set()
    for traversal, receipt in enumerate(receipts):
        arm = {'case': receipt['case'], 'receipt': receipt, 'status': 'invalid_reference',
               'comparison_valid': False, 'score_computed': False, 'greedy_equal': None}
        try:
            if receipt['case'] in seen:
                raise ValueError('duplicate sweep arm receipt')
            seen.add(receipt['case'])
            if receipt['case'] not in specs:
                raise ValueError('sweep arm has no completed spec summary')
            path = output_dir / (receipt['case'] + '.generated.i32')
            candidate = greedy.describe(args.candidate_job, args.candidate_log, path, profile,
                 command_index=args.candidate_command,
                 traversal_index=traversal, output_case=receipt['case'], declared_receipt=receipt,
                 result_path=args.candidate_result)
            greedy.validate_roles(reference, candidate)
            checks = {key: reference[key] == candidate[key] for key in greedy.PREFLIGHT_KEYS}
            arm.update(preflight=checks, target=candidate, diagnostic=specs[receipt['case']])
            if not all(checks.values()):
                raise ValueError('mandatory target/source/geometry/profile preflight mismatch')
            raw = path.read_bytes()
            if len(raw) != expected * 4 or specs[receipt['case']]['tokens'] != expected:
                raise ValueError('sweep raw/summary generation length mismatch')
            a, b = struct.unpack('<' + 'i' * expected, raw_reference), struct.unpack('<' + 'i' * expected, raw)
            bad = [i for i, (left, right) in enumerate(zip(a, b)) if left != right]
            arm.update(status='pass' if not bad else 'greedy_mismatch', comparison_valid=True,
                       score_computed=True, greedy_equal=not bad, different=len(bad),
                       first_different=bad[0] if bad else None, generated_sha256=digest(path))
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            arm['reason'] = str(error)
        arms.append(arm)
    complete = len(arms) == args.expected_arms and len(specs) == args.expected_arms
    valid = complete and all(arm['comparison_valid'] for arm in arms)
    passed = valid and all(arm['greedy_equal'] for arm in arms)
    result = {'format': 'm31-port7-sweep-greedy-audit-v1', 'reference': reference,
              'summary_sha256': digest(args.summary), 'expected_arms': args.expected_arms,
              'complete': complete, 'comparison_valid': valid, 'pass': passed, 'arms': arms,
              'sources': {str(Path(__file__)): digest(Path(__file__)),
                          str(Path(greedy.__file__)): digest(Path(greedy.__file__))},
              'scope': 'Mandatory per-arm source/profile/actual geometry and raw greedy audit. Timing is diagnostic under concurrent compilation, never om acceptance.'}
    with args.out.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'complete': complete, 'valid': valid, 'pass': passed,
         'arms': [{'case': arm['case'], 'status': arm['status'], 'reason': arm.get('reason'),
                   'tps_diagnostic': arm.get('diagnostic', {}).get('tps')} for arm in arms]}, indent=2))
    return 0 if passed else 2 if valid else 3


if __name__ == '__main__':
    raise SystemExit(main())
