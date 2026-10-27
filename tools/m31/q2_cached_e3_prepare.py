#!/usr/bin/env python3
"""Prepare ONE opt-in existing-ELF E3 residency diagnostic; never launch a model.

The 384-slot pool is deliberately full of the 384 E3 experts after priming.
A larger pool would retain incidental prefill entries: native cached decode has
no layer whitelist. This controlled rank describes placement, not calibration.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct

from cache_seed_preflight import verify, seal_receipt, tensor_header

ROOT = Path('/mnt/ai/Xolotl-m31')
LOG = Path('/mnt/ai/logs/m31-port')
MODEL = Path('/mnt/ai/models/MiniMax-M3.1-Flash-GGUF/UD-Q2_K_XL/MiniMax-M3.1-Flash-UD-Q2_K_XL-00001-of-00004.gguf')
REVISION = 'b17b7f5459c0b8f127ea12ee774409b282521d6a'
STRIDE = 27615616
ELF_SHA = '2b51de98bbae67d6f9d14a83da1d530d6cf04fa47cdbe7ab917bd99c466e9a8c'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def small_fp(path):
    path = path.resolve(strict=True)
    size = path.stat().st_size
    if size > 32 << 20:
        raise ValueError('small-file hash bound: ' + str(path))
    return dict(path=str(path), bytes=size, sha256=sha(path.read_bytes()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--profile', type=Path, default=LOG / 'q2-native-speed-v2/runtime-profile.json')
    a = p.parse_args()
    directory = a.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    if directory.parent != LOG or directory.name != 'q2-cached-e3-code128-v1':
        raise ValueError('this bounded trial has one fixed unique directory/name')
    base = json.loads(a.profile.read_text())
    if base.get('inherits') or base['environment'].get('M31_CPU_STOCK_EXPERTS') != '1':
        raise ValueError('resolved canonical Q2-native profile required')
    write(directory / 'base-profile.json', base)
    headers = []
    metadata = {}
    tensors = []
    expected_sha = {}
    for line in Path('/mnt/ai/logs/m31-q2kxl-download.tsv').read_text().splitlines():
        name, digest = line.split('\t')
        if len(bytes.fromhex(digest)) != 32:
            raise ValueError('invalid inherited user-verified digest')
        expected_sha[Path(name).name] = digest
    shards = sorted(MODEL.parent.glob('*.gguf'))
    if len(shards) != 4 or set(expected_sha) != {path.name for path in shards}:
        raise ValueError('exact four Q2 shards/digests required')
    quants = {0: (1, 4), 8: (32, 34), 12: (256, 144), 13: (256, 176),
              14: (256, 210), 17: (256, 74), 18: (256, 98), 23: (256, 136)}
    integrity = []
    for path in shards:
        meta, rows, start, header_sha = tensor_header(path)
        metadata.update(meta)
        st = path.stat()
        headers.append(dict(path=str(path.resolve()), bytes=st.st_size, mtime_ns=st.st_mtime_ns,
                            reference_sha256=expected_sha[path.name], revision=REVISION))
        integrity.append(dict(file=path.name, actual=expected_sha[path.name], expected=expected_sha[path.name],
                              match=True, revision=REVISION,
                              provenance='Inherited user confirmation: dl_hf.sh/sha256 OK; TSV supplies digests. Not freshly hashed here.'))
        for row in rows:
            block, block_bytes = quants[row['type_id']]
            if row['shape'][0] % block:
                raise ValueError('nonintegral quant row')
            row['bytes'] = math.prod(row['shape']) // block * block_bytes
            row['shard'] = str(path.resolve())
            tensors.append(row)
    keys = {'layers': 'block_count', 'embd': 'embedding_length', 'experts': 'expert_count',
            'used': 'expert_used_count', 'dense_lead': 'leading_dense_block_count',
            'expert_ff': 'expert_feed_forward_length', 'heads': 'attention.head_count',
            'kv_heads': 'attention.head_count_kv', 'head_dim': 'attention.key_length',
            'index_dim': 'attention.indexer.key_length', 'index_heads': 'attention.indexer.head_count',
            'block': 'attention.indexer.block_size', 'top_blocks': 'attention.indexer.top_k'}
    geometry = {key: metadata['minimax-m3.' + name] for key, name in keys.items()}
    geometry['model'] = str(MODEL)
    if (geometry['layers'], geometry['experts'], geometry['dense_lead'], geometry['used']) != (60, 128, 3, 4):
        raise ValueError('unexpected Q2 geometry')
    write(directory / 'tensors.json', tensors)
    write(directory / 'integrity.json', integrity)
    identity = dict(geometry=geometry, tensors_sha256=small_fp(directory / 'tensors.json')['sha256'], shards=headers)
    identity_sha = sha(json.dumps(identity, sort_keys=True).encode())
    records = [dict(layer=l, expert=e, count=0, weight_sum=float(l in (10, 11, 12)),
                    placement_priority=int(l in (10, 11, 12)))
               for l in range(3, 60) for e in range(128)]
    records.sort(key=lambda row: (-row['count'], -row['weight_sum'], row['layer'], row['expert']))
    ids = [row['layer'] * 128 + row['expert'] for row in records]
    if ids[:384] != list(range(1280, 1664)) or len(set(ids)) != 7296:
        raise ValueError('controlled placement permutation differs')
    rank = dict(format='m31-expert-ranking-v1', ranking_only=True, identity=identity,
                identity_sha256=identity_sha, records=records, training_route_count=0,
                score_semantics=dict(count='All zero: no measured or fabricated usage counts.',
                                     weight_sum='Legacy sort field holds synthetic placement priority0/1, NOT observed route weights.'),
                calibration='none', source_trace=None,
                placement_priority=dict(target_layers=[10, 11, 12], target_priority=1, other_priority=0),
                scope='Synthetic fixed E3 residency permutation, not calibration/hot-frequency ranking. No usage observations or quality calibration.')
    write(directory / 'rank.json', rank)
    with (directory / 'rank.bin').open('xb') as stream:
        stream.write(struct.pack('<8sIIII', b'M31RANK\0', 1, 60, 128, 7296) + bytes.fromhex(identity_sha))
        stream.write(struct.pack('<7296I', *ids))
    receipt = verify(directory / 'rank.json', directory / 'rank.bin', directory / 'tensors.json',
                     directory / 'integrity.json', 384, 0)
    receipt['original_prefill_budget'] = receipt['budget'].copy()
    receipt['budget'].update(max_hot=384, requested_hot=384, stream_slots=0,
                             recommendation_scope='Decode-only exact E3 residency; no admission of misses.')
    receipt['mode'] = 'decode-only'
    receipt['semantic_scope'] = 'Integrity/geometry/model-identity preflight only; not routing calibration, usage frequency or quality proof.'
    receipt['runtime_contract'] = ['Physical slots MUST equal384, not merely>=384.',
                                  'Decode-only pins are cleared before every n>1 GPU prefill.',
                                  'No additional expert keys may remain after384-key priming.']
    write(directory / 'preflight.json', receipt)
    with (directory / 'seed.receipt').open('xb') as stream:
        stream.write(seal_receipt(receipt))
    write(directory / 'rank-metadata-audit.json', dict(
        format='m31-q2-synthetic-placement-metadata-audit-v1',
        all_usage_counts_zero=all(row['count'] == 0 for row in records),
        training_route_count_zero=rank['training_route_count'] == 0,
        calibration_none=rank['calibration'] == 'none', source_trace_null=rank['source_trace'] is None,
        legacy_weight_field_is_explicit_synthetic_priority=True,
        record_placement_priority_matches=all(row['placement_priority'] == int(row['layer'] in (10, 11, 12))
                                              and row['weight_sum'] == row['placement_priority'] for row in records),
        prefix384_exact=ids[:384] == list(range(1280, 1664)),
        complete_unique_routed_permutation=set(ids) == set(range(384, 7680)) and len(ids) == 7296,
        preflight_verified=receipt['verified'], preflight_scope=receipt['semantic_scope'],
        model_payload_hash_fresh=False,
        inherited_integrity='User confirmed payload SHA256/dl_hf.sh OK; the four TSV digests are inherited, current headers/sizes/mtime are checked.'))
    profile = json.loads(json.dumps(base))
    env = profile['environment']
    env.update(M31_GPU_CACHED_DECODE='1', M31_GPU_CACHED_DECODE_NATIVE='1',
               M31_GPU_CACHED_DECODE_EXACT='0', M31_GPU_DECODE_SEED_HOT='384', M31_GPU_SEED_HOT='0',
               M31_GPU_SEED_PATH=str(directory / 'rank.bin'), M31_GPU_SEED_IDENTITY=identity_sha,
               M31_GPU_SEED_SHA256=small_fp(directory / 'rank.bin')['sha256'],
               M31_GPU_SEED_RECEIPT=str(directory / 'seed.receipt'),
               M31_GPU_SEED_RECEIPT_SHA256=small_fp(directory / 'seed.receipt')['sha256'])
    profile['unset_environment'] = sorted(set(profile['unset_environment']) - set(env))
    argv = profile['engine_arguments']
    argv[argv.index('--gpu-experts-gib') + 1] = '9.88'
    profile['name'] = 'q2-cached-e3-code128-v1'
    profile['cache_seed'] = dict(mode='decode-only', hot=384, physical_slots=384, target_layers=[10, 11, 12])
    profile['provenance'] = dict(status='PREPARED_ONLY, not a replacement for baseline quality or performance.')
    write(directory / 'runtime-profile.json', profile)
    source_elf = LOG / 'q2-compat-final-v1/m31_generate-frozen'
    source_manifest = json.loads((LOG / 'q2-compat-final-v1/manifest.json').read_text())
    declared = source_manifest['native']
    st = source_elf.stat()
    if declared['sha256'] != ELF_SHA or declared['bytes'] != st.st_size or st.st_mode & 0o222:
        raise ValueError('existing frozen canonical ELF identity/mode mismatch')
    os.link(source_elf, directory / 'm31_generate-frozen')
    source_ids = LOG / 'q2-compat-final-v1/code128/code_02_128.i32'
    shutil.copy2(source_ids, directory / 'code_02_128.i32')
    if source_ids.stat().st_size != 512:
        raise ValueError('code128 IDs extent')
    command = ['/usr/bin/env', *[x for key in profile['unset_environment'] for x in ('-u', key)],
               *[key + '=' + str(value) for key, value in sorted(env.items())],
               str(directory / 'm31_generate-frozen'), '-m', str(MODEL),
               '--ids-file', str(directory / 'code_02_128.i32'), '--rows', 'all', '--taps', 'none',
               '--decode-tail', '32', *argv, '--out', str(directory / 'native')]
    job = directory / (directory.name + '-card-job.json')
    write(job, dict(preset='m31-xolotl-check', measurement=False, command_timeout_seconds=300,
                    commands=[command]))
    helpers = {}
    for name in ('card_session.sh', 'card_worker.py', 'card_preset.py'):
        suffix = {'card_session.sh': '.runner.sh', 'card_worker.py': '.worker.py', 'card_preset.py': '.preset.py'}[name]
        target = job.with_suffix(suffix)
        shutil.copy2(ROOT / 'tools/m31' / name, target)
        helpers[name] = small_fp(target)
    shutil.copy2(ROOT / 'tools/m31/q2_cached_e3_launch.py', directory / 'launch_once.py')
    shutil.copy2(Path(__file__), directory / 'prepare-frozen.py')
    for name in ('cache_seed_preflight.py',):
        shutil.copy2(ROOT / 'tools/m31' / name, directory / name)
    frozen = {path.name: small_fp(path) for path in directory.iterdir()
              if path.is_file() and path.name != 'm31_generate-frozen'}
    for path in directory.iterdir():
        if path.is_file() and path.name != 'm31_generate-frozen':
            path.chmod(0o444)
    manifest = dict(format='m31-q2-cached-e3-bounded-trial-v1', status='PREPARED_ONLY',
                    maximum_native_attempts=1, start_after_both_baseline_qualities_and_cleanup=True,
                    job=str(job), profile=str(directory / 'runtime-profile.json'), frozen_files=frozen,
                    elf=dict(path=str(directory / 'm31_generate-frozen'), sha256=ELF_SHA, bytes=st.st_size,
                             mtime_ns=st.st_mtime_ns, source=str(source_elf),
                             freeze='Hardlink to existing mode0555 canonical frozen inode; no large copy/hash during baseline quality. SHA rechecked only after both baseline cleanups before trial.'),
                    gold_case=str(LOG / 'q2-best-placement-code128-e3-v1/oracle/code_02_128'),
                    baseline_qualities=[dict(directory=str(LOG / 'q2-reference-quality-e3-v1'),
                                             eval='eval/m31-q2-ref-e3.jsonl', unit='codex-m31-q2-reference-quality-e3-v1'),
                                        dict(directory=str(LOG / 'q2-native-quality-v1'),
                                             eval='eval/q2-native-quality-v1.jsonl', unit='codex-m31-q2-native-quality-v1')],
                    expected=dict(slots=384, slot_stride=STRIDE, pool_bytes=384*STRIDE,
                                  target_layers=[10, 11, 12], target_prefix_ids_inclusive=[1280, 1663],
                                  n1_calls=32, route_lookups=7296, resident_routes=384,
                                  hybrid_forwards=96, native_GPU_forwards=153,
                                  seed_primes=1, seed_bytes=6964641792, all_target_layers_all4_hits=True,
                                  no_incidental_entries='Capacity384 equals384 unique seed keys; full priming evicts every nonseed key. Aggregated counters alone are not per-layer telemetry.'),
                    primary_numeric_contract=dict(top1_min=.99, mean_kl_max=.001, max_kl_max=.05,
                                                   scope='Whole128 plus separate32-row n1 tail; selected diagnostic, no full20/long/quality acceptance.'),
                    launch_check=['/usr/bin/python3', str(directory / 'launch_once.py'), '--directory', str(directory)],
                    launch_execute=['/usr/bin/python3', str(directory / 'launch_once.py'), '--directory', str(directory), '--execute'],
                    no_followup_jobs=True, no_baseline_profile_or_runtime_edits=True)
    write(directory / 'manifest.json', manifest)
    with (directory / 'manifest-seal.sha256').open('x') as stream:
        stream.write(small_fp(directory / 'manifest.json')['sha256'] + '\n')
    (directory / 'manifest.json').chmod(0o444)
    (directory / 'manifest-seal.sha256').chmod(0o444)
    print(json.dumps(dict(directory=str(directory), job=str(job), slots=384, hot=384,
                         rank_identity=identity_sha, prepared_only=True), indent=2))


if __name__ == '__main__':
    main()
