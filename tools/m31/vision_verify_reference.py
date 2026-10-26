#!/usr/bin/env python3
"""Verify CPU mtmd preprocessing spans against saved image/video reference logs."""
import argparse
import json
from pathlib import Path
import re
from vision_batch import load

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('directory',type=Path)
p.add_argument('--reference',type=Path,default=Path('/mnt/ai/logs/m31-ref'))
p.add_argument('--out',type=Path)
a=p.parse_args();results=[]
for label,expected_frames in [('image',1),('video',12)]:
    m=load(a.directory/f'{label}-ipc/manifest.json')
    ref=a.reference/f'm31-ref-1g-vision-64-u1024.{label}.server.log'
    log=ref.read_text()
    positions=[int(x) for x in re.findall(r'encoding mtmd batch from idx = (\d+)',log)]
    token_counts=[int(x) for x in re.findall(r'encoding batch with \d+ entries and total (\d+) tokens',log)]
    prompt_counts=[int(x) for x in re.findall(r'task\.n_tokens = (\d+)',log)]
    shapes=[list(map(int,x)) for x in re.findall(r'copying image \d+/\d+ to input buffer \(nx=(\d+), ny=(\d+)\)',log)]
    images=[c for c in m['chunks'] if c['type']=='image']
    own_shapes=[[e['resized_width'],e['resized_height']] for c in images for e in c['geometry']['entries']]
    checks={'frame_count':len(images)==expected_frames,'absolute_image_spans':[c['row'] for c in images]==positions,
            'visual_token_counts':[c['rows'] for c in images]==token_counts,
            'prompt_token_total':bool(prompt_counts) and m['rows']==prompt_counts[0],
            'resized_geometry':own_shapes==shapes}
    results.append({'case':label,'pass':all(checks.values()),'checks':checks,
                    'total_rows':m['rows'],'visual_rows':m['visual_rows'],
                    'image_positions':[c['row'] for c in images], 'resized_pixels':own_shapes,
                    'reference_log':str(ref)})
report={'pass':all(r['pass'] for r in results),'scope':'CPU mtmd preprocessing/metadata; no encoder embeddings or native LLM parity',
        'cases':results}
data=json.dumps(report,ensure_ascii=False,indent=2)+'\n'
if a.out:
    with a.out.open('x') as f:f.write(data)
print(data,end='')
if not report['pass']:raise SystemExit(1)
