#!/usr/bin/env python3
"""Run fresh header/stat preflight, seal it, then exec a seeded job inside aictl."""
import argparse
import json
import os
from pathlib import Path
import sys
from cache_seed_preflight import verify, seal_receipt, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('rank','binary','tensors','integrity','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--slots',type=int,required=True)
    p.add_argument('--hot',type=int,required=True)
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args()
    command=a.command[1:] if a.command and a.command[0]=='--' else a.command
    if not command:
        p.error('an explicit executable/argv is required after --')
    if os.environ.get('M31_CARD_SESSION')!='1':
        p.error('execution requires the aictl/card-session worker')
    try:
        receipt=verify(a.rank,a.binary,a.tensors,a.integrity,a.slots,a.hot)
        sealed=seal_receipt(receipt)
        seal_path=a.output.with_suffix('.seal')
        with a.output.open('x') as f:
            json.dump(receipt,f,indent=2,allow_nan=False);f.write('\n')
        with seal_path.open('xb') as f:
            f.write(sealed)
        env=dict(os.environ,M31_GPU_SEED_HOT=str(a.hot),M31_GPU_SEED_PATH=receipt['rank_binary'],
                 M31_GPU_SEED_IDENTITY=receipt['identity_sha256'],M31_GPU_SEED_SHA256=receipt['rank_binary_sha256'],
                 M31_GPU_SEED_RECEIPT=str(seal_path.resolve()),M31_GPU_SEED_RECEIPT_SHA256=sha(sealed))
        print(json.dumps({'preflight':str(a.output.resolve()),'seal':str(seal_path.resolve()),
                          'hot':a.hot,'slots':a.slots,
                          'model_content_verification':receipt['model_content_verification']}),flush=True)
        os.execvpe(command[0],command,env)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        p.exit(2,f'cache_seed_launch: {exc}\n')


if __name__=='__main__':
    main()
