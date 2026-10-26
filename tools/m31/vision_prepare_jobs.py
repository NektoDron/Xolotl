#!/usr/bin/env python3
"""Create fresh CPU-only preprocessing/encode job inputs matching reference media."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('directory',type=Path)
p.add_argument('--encode',action='store_true')
a=p.parse_args();a.directory.mkdir(exist_ok=False)
root=Path('/mnt/ai/models/MiniMax-M3.1-Flash-GGUF')
media=Path('/mnt/ai/logs/m31-ref/media')
common={'model':str(root/'UD-IQ3_S/MiniMax-M3.1-Flash-UD-IQ3_S-00001-of-00005.gguf'),
        'mmproj':str(root/'mmproj-BF16.gguf'),'video_fps':1.0,'timestamp_interval_ms':5000,
        'encode':a.encode,'threads':8,'encoder_batch_tokens':4096,'enable_thinking':True,
        'chat_template_kwargs':{'reasoning_effort':'low'}}
for label,name,prompt in [
 ('image','0.png','Describe the visible shape, its color and its position. Be concise.'),
 ('video','shapes-12s.mp4','Describe the sequence of changes in this video in chronological order: the shape, color and position in each stage. Be concise.')]:
    spec={**common,'media':[str(media/name)],'messages':[{'role':'user','content':[
        {'type':'text','text':prompt},{'type':'media_marker','text':'<__media__>'}]}]}
    (a.directory/(label+'.json')).write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n')
print(a.directory)
