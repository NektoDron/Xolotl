#!/usr/bin/env python3
"""Verified decode-only hot set; all pins are released before next GPU prefill.

Ordinary prefill seed preflight still reserves128 streaming slots. This separate
mode permits using the full existing pool during cached-only decode, when misses
are computed on CPU and no weight replacement is performed.
"""
import argparse
import json
from pathlib import Path
from cache_seed_preflight import verify, seal_receipt, sha
p=argparse.ArgumentParser(description=__doc__)
for name in ['rank','binary','tensors','integrity','output','seal']:p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--slots',type=int,required=True);p.add_argument('--hot',type=int,required=True);a=p.parse_args()
if not 1 <= a.hot <= a.slots: p.error('decode-only hot count must be in1..slots')
try:
    # Verify all identity, model header/stat, rank permutation/scores, integrity
    # and normal pool geometry checks with the historical prefill reserve.
    receipt=verify(a.rank,a.binary,a.tensors,a.integrity,a.slots,0)
    original=receipt['budget'].copy()
    receipt['format']='m31-seed-preflight-v1'
    receipt['original_prefill_budget']=original
    receipt['budget'].update(max_hot=a.slots,requested_hot=a.hot,stream_slots=a.slots-a.hot,
        recommendation_scope='decode-only fixed residency; no GPU admission of misses')
    receipt['mode']='decode-only'
    receipt['runtime_contract']=['Set M31_GPU_DECODE_SEED_HOT to this hot count.','M31_GPU_SEED_HOT must be0/unset.','GpuExperts releases all decode-only pins before every n>1 GPU forward.','Ordinary seed_hot must still reject counts exceeding slots-experts.']
    sealed=seal_receipt(receipt)
    with a.output.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    with a.seal.open('xb') as f:f.write(sealed)
    print(json.dumps(dict(verified=True,mode=receipt['mode'],budget=receipt['budget'],seal=str(a.seal.resolve()),seal_sha256=sha(sealed)),indent=2))
except (OSError,ValueError,KeyError,TypeError) as e:p.exit(2,f'port7 decode seed preflight: {e}\n')
