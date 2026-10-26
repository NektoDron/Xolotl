#!/usr/bin/env python3
"""Prepare a small F32 model/rank/receipt for the direct GPU cache probe."""
import argparse
import json
import math
from pathlib import Path
import struct
import sys
from cache_seed_preflight import sha, tensor_header, verify, seal_receipt
sys.path.insert(0,str(Path(__file__).parent/'tests'))
from make_fixture import make_fixture

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('directory',type=Path)
a=p.parse_args()
a.directory.mkdir(exist_ok=False)
model=(a.directory/'tiny.gguf').resolve()
make_fixture(model)
meta,tensors,_,_=tensor_header(model)
for row in tensors:
    row['bytes']=math.prod(row['shape'])*4
    row['shard']=str(model)
tensors_path=a.directory/'tensors.json'
tensors_path.write_text(json.dumps(tensors,indent=2)+'\n')
keys={'layers':'block_count','embd':'embedding_length','experts':'expert_count','used':'expert_used_count',
      'dense_lead':'leading_dense_block_count','expert_ff':'expert_feed_forward_length',
      'heads':'attention.head_count','kv_heads':'attention.head_count_kv','head_dim':'attention.key_length',
      'index_dim':'attention.indexer.key_length','index_heads':'attention.indexer.head_count',
      'block':'attention.indexer.block_size','top_blocks':'attention.indexer.top_k'}
geometry={key:meta['minimax-m3.'+value] for key,value in keys.items()}
geometry['model']=str(model)
stat=model.stat();content_sha=sha(model.read_bytes())
identity={'geometry':geometry,'tensors_sha256':sha(tensors_path.read_bytes()),
          'shards':[{'path':str(model),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,
                     'reference_sha256':content_sha,'revision':'synthetic-f32-probe'}]}
identity_sha=sha(json.dumps(identity,sort_keys=True).encode())
records=[{'layer':layer,'expert':expert,'count':0,'weight_sum':0.0}
         for layer in range(geometry['dense_lead'],geometry['layers']) for expert in range(geometry['experts'])]
rank={'format':'m31-expert-ranking-v1','ranking_only':True,'identity':identity,
      'identity_sha256':identity_sha,'records':records,'training_route_count':0}
rank_path=a.directory/'rank.json';rank_path.write_text(json.dumps(rank,indent=2)+'\n')
ids=[r['layer']*geometry['experts']+r['expert'] for r in records]
rank_bin=a.directory/'rank.bin'
rank_bin.write_bytes(struct.pack('<8sIIII',b'M31RANK\0',1,geometry['layers'],geometry['experts'],len(ids))
                     +bytes.fromhex(identity_sha)+struct.pack('<'+'I'*len(ids),*ids))
integrity=a.directory/'integrity.json'
integrity.write_text(json.dumps([{'file':model.name,'actual':content_sha,'expected':content_sha,
                                 'match':True,'revision':'synthetic-f32-probe'}],indent=2)+'\n')
receipt=verify(rank_path,rank_bin,tensors_path,integrity,6,2)
(a.directory/'preflight.json').write_text(json.dumps(receipt,indent=2)+'\n')
seal=seal_receipt(receipt);seal_path=a.directory/'preflight.seal';seal_path.write_bytes(seal)
info={'model':str(model),'seal':str(seal_path.resolve()),'seal_sha256':sha(seal),
      'slots':6,'hot':2,'rank':str(rank_bin.resolve()),'identity_sha256':identity_sha,'rank_sha256':sha(rank_bin.read_bytes())}
(a.directory/'fixture.json').write_text(json.dumps(info,indent=2)+'\n')
print(json.dumps(info,indent=2))
