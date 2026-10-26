#!/usr/bin/env python3
"""Negative rank/header/stat tests and CPU-only C++ reader checks."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from cache_seed_preflight import decode_rank, seed_budget, sha, tensor_header, verify, seal_receipt

sys.path.insert(0, str(Path(__file__).parent / 'tests'))
from make_fixture import make_fixture

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--reader', type=Path)
parser.add_argument('--receipt-reader', type=Path)
options, rest = parser.parse_known_args()
sys.argv = [sys.argv[0]] + rest


class SeedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='m31-seed-reader-')
        self.root = Path(self.temp.name)
        self.model = self.root / 'tiny.gguf'
        make_fixture(self.model)
        meta, tensors, _, _ = tensor_header(self.model)
        for row in tensors:
            row['bytes'] = math.prod(row['shape']) * 4
            row['shard'] = str(self.model)
        self.tensors = self.root / 'tensors.json'
        self.tensors.write_text(json.dumps(tensors))
        stat = self.model.stat()
        keys = {'layers': 'block_count', 'embd': 'embedding_length', 'experts': 'expert_count',
                'used': 'expert_used_count', 'dense_lead': 'leading_dense_block_count',
                'expert_ff': 'expert_feed_forward_length', 'heads': 'attention.head_count',
                'kv_heads': 'attention.head_count_kv', 'head_dim': 'attention.key_length',
                'index_dim': 'attention.indexer.key_length', 'index_heads': 'attention.indexer.head_count',
                'block': 'attention.indexer.block_size', 'top_blocks': 'attention.indexer.top_k'}
        geometry = {name: meta['minimax-m3.' + key] for name, key in keys.items()}
        geometry['model'] = str(self.model)
        digest = sha(self.model.read_bytes())
        identity = {'geometry': geometry, 'tensors_sha256': sha(self.tensors.read_bytes()),
                    'shards': [{'path': str(self.model), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                                'reference_sha256': digest, 'revision': 'test-fixture'}]}
        self.rank = {'format': 'm31-expert-ranking-v1', 'ranking_only': True, 'identity': identity,
                     'identity_sha256': sha(json.dumps(identity, sort_keys=True).encode()),
                     'records': [{'layer': i // 4, 'expert': i % 4, 'count': 16 - i, 'weight_sum': (16 - i) / 2}
                                 for i in range(4, 16)], 'training_route_count': sum(range(1, 13))}
        self.json = self.root / 'rank.json'
        self.json.write_text(json.dumps(self.rank))
        self.binary = self.root / 'rank.bin'
        self.original = (struct.pack('<8sIIII', b'M31RANK\0', 1, 4, 4, 12)
                         + bytes.fromhex(self.rank['identity_sha256']) + struct.pack('<12I', *range(4, 16)))
        self.binary.write_bytes(self.original)
        self.integrity = self.root / 'integrity.json'
        self.integrity.write_text(json.dumps([{'file': self.model.name, 'actual': digest, 'expected': digest,
                                               'match': True, 'revision': 'test-fixture'}]))
        self.seal = self.root / 'preflight.seal'
        self.seal.write_bytes(seal_receipt(self.preflight()))

    def tearDown(self):
        self.temp.cleanup()

    def preflight(self):
        return verify(self.json, self.binary, self.tensors, self.integrity, 8)

    def reader(self, raw=None, expected_sha=None, identity=None, extra=()):
        if not options.reader:
            self.skipTest('pass --reader after building the CPU-only C++ tool')
        if raw is not None:
            self.binary.write_bytes(raw)
        data = self.binary.read_bytes()
        return subprocess.run([str(options.reader), '--rank', str(self.binary), '--identity',
                               identity or self.rank['identity_sha256'], '--sha256', expected_sha or sha(data),
                               '--layers', '4', '--experts', '4', '--leading', '1', '--used', '2', '--slots', '8',
                               *extra], capture_output=True, text=True)

    def receipt_check(self, data=None, expected_sha=None, model=None):
        if not options.receipt_reader:
            self.skipTest('pass --receipt-reader after building the CPU header verifier')
        if data is not None:
            self.seal.write_bytes(data)
        return subprocess.run([str(options.receipt_reader),str(self.seal),
                               expected_sha or sha(self.seal.read_bytes()),str(model or self.model)],
                              capture_output=True,text=True)

    def test_current_fixture_and_safe_budget(self):
        result = self.preflight()
        self.assertTrue(result['verified'])
        self.assertEqual(result['count'], 12)
        self.assertEqual(result['budget']['requested_hot'], 2)
        self.assertEqual(result['budget']['stream_slots'], 6)
        self.assertEqual(seed_budget(248, 128, 60, 3, 4)['recommended_hot'], 20)
        with self.assertRaises(ValueError):
            seed_budget(248, 128, 60, 3, 4, 121)
        with self.assertRaises(ValueError):
            seed_budget(127, 128, 60, 3, 4)
        with self.assertRaises(ValueError):
            seed_budget(13, 4, 4, 1, 2)

    def test_duplicate_dense_out_of_range_truncated_and_trailing_binary(self):
        for ids in ([4] * 12, list(range(0, 12)), list(range(5, 17))):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                decode_rank(self.original[:56] + struct.pack('<12I', *ids), self.rank['identity_sha256'], 4, 4, 1)
        for data in (self.original[:-1], self.original + b'x'):
            with self.subTest(data=len(data)), self.assertRaises(ValueError):
                decode_rank(data, self.rank['identity_sha256'], 4, 4, 1)

    def test_valid_permutation_swapped_binary_rejected_by_json(self):
        data = bytearray(self.original)
        data[56:60], data[60:64] = data[60:64], data[56:60]
        self.binary.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'order differs'):
            self.preflight()

    def test_json_identity_and_reference_checksum_rejected(self):
        self.rank['identity']['geometry']['embd'] += 1
        self.json.write_text(json.dumps(self.rank))
        with self.assertRaisesRegex(ValueError, 'identity checksum'):
            self.preflight()
        self.rank['identity']['geometry']['embd'] -= 1
        self.json.write_text(json.dumps(self.rank))
        self.tensors.write_text(self.tensors.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'directory checksum'):
            self.preflight()

    def test_model_stat_change_rejected(self):
        stat = self.model.stat()
        os.utime(self.model, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
        with self.assertRaisesRegex(ValueError, 'size/mtime'):
            self.preflight()

    def test_model_header_change_with_same_stat_rejected(self):
        stat = self.model.stat()
        raw = self.model.read_bytes().replace(b'output.weight', b'output.weighu', 1)
        self.model.write_bytes(raw)
        os.utime(self.model, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, 'directory names'):
            self.preflight()

    def test_payload_only_preserved_stat_is_explicit_limit(self):
        stat = self.model.stat()
        raw = bytearray(self.model.read_bytes())
        raw[-1] ^= 1
        self.model.write_bytes(raw)
        os.utime(self.model, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        receipt = self.preflight()
        self.assertIn('inherited', receipt['model_content_verification'])
        self.assertIn('Payload-only', receipt['limitation'])
        self.assertNotEqual(sha(raw), self.rank['identity']['shards'][0]['reference_sha256'])

    def test_duplicate_json_key_rejected(self):
        self.json.write_text('{"format":"first","format":"m31-expert-ranking-v1"}')
        with self.assertRaisesRegex(ValueError, 'duplicate JSON'):
            self.preflight()

    def test_sparse_oversized_inputs_rejected_before_open_or_read(self):
        # Only sparse metadata is created: no 150-GiB allocation/read/download.
        # Path.open is forbidden for the oversized path to prove the guard runs
        # before either read_bytes/read_text or their bounded replacement.
        original_open = Path.open
        for path in (self.binary, self.json, self.tensors, self.integrity):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                try:
                    with path.open('wb') as stream:
                        stream.truncate(150 << 30)
                    self.assertLess(path.stat().st_blocks * 512, 1 << 20)
                    def guarded_open(candidate, *args, **kwargs):
                        if candidate == path:
                            raise AssertionError('oversized input reached open/read')
                        return original_open(candidate, *args, **kwargs)
                    with mock.patch.object(Path, 'open', guarded_open):
                        with self.assertRaisesRegex(ValueError, 'byte size'):
                            self.preflight()
                finally:
                    path.write_bytes(original)

    def test_cpp_known_sha256_vectors(self):
        if not options.reader:
            self.skipTest('C++ reader not built')
        for text in ('', 'abc', 'a' * 1000, 'unicode: привет'):
            result = subprocess.run([str(options.reader), '--hash-text', text], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), hashlib.sha256(text.encode()).hexdigest())

    def test_cpp_valid_rank_budget(self):
        result = self.reader()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {'verified': True, 'count': 12, 'hot': 2, 'stream': 6, 'max_hot': 4})
        result = self.reader(extra=['--hot', '5'])
        self.assertNotEqual(result.returncode, 0)

    def test_cpp_structural_failures_even_with_recomputed_file_hash(self):
        mutations = []
        for offset, value in [(0, 0), (8, 2), (12, 5), (16, 5), (20, 11), (56, 0), (56, 16), (60, 4)]:
            raw = bytearray(self.original)
            raw[offset:offset+4] = struct.pack('<I', value)
            mutations.append(raw)
        mutations += [self.original[:-1], self.original+b'x']
        for data in mutations:
            result = self.reader(data)
            self.assertNotEqual(result.returncode, 0, f'accepted corruption: {data}')
        result = self.reader(self.original, identity='00'*32)
        self.assertNotEqual(result.returncode, 0)

    def test_cpp_swapped_valid_permutation_rejected_by_expected_sha(self):
        data = bytearray(self.original)
        data[56:60], data[60:64] = data[60:64], data[56:60]
        result = self.reader(data, expected_sha=sha(self.original))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('content SHA256 mismatch', result.stderr)

    def test_receipt_cpp_valid(self):
        result=self.receipt_check()
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout),{'verified':True,'count':12,'slots':8,'hot':2})

    def test_receipt_hash_truncation_trailing_and_version(self):
        raw=self.seal.read_bytes()
        result=self.receipt_check(expected_sha='00'*32)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('receipt SHA256 mismatch',result.stderr)
        version=bytearray(raw);version[8:12]=struct.pack('<I',2)
        for mutation in (raw[:-1],raw+b'x',version):
            result=self.receipt_check(mutation)
            self.assertNotEqual(result.returncode,0)

    def test_receipt_current_metadata_header_change_same_stat_rejected(self):
        stat=self.model.stat()
        raw=self.model.read_bytes().replace(b'minimax-m3',b'minimax-m4',1)
        self.model.write_bytes(raw)
        os.utime(self.model,ns=(stat.st_atime_ns,stat.st_mtime_ns))
        result=self.receipt_check()
        self.assertNotEqual(result.returncode,0)
        self.assertIn('header SHA256 differs',result.stderr)

    def test_receipt_current_model_stat_and_path_rejected(self):
        import shutil
        alternate=self.root/'other.gguf';shutil.copy2(self.model,alternate)
        result=self.receipt_check(model=alternate)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('path differs',result.stderr)
        stat=self.model.stat();os.utime(self.model,ns=(stat.st_atime_ns,stat.st_mtime_ns+1))
        result=self.receipt_check()
        self.assertNotEqual(result.returncode,0)
        self.assertIn('size/mtime differs',result.stderr)

    def test_receipt_rank_changed_after_preflight_rejected(self):
        raw=bytearray(self.original);raw[56:60],raw[60:64]=raw[60:64],raw[56:60]
        self.binary.write_bytes(raw)
        result=self.receipt_check()
        self.assertNotEqual(result.returncode,0)
        self.assertIn('content SHA256 mismatch',result.stderr)


if __name__ == '__main__':
    unittest.main()
