#!/usr/bin/env python3
"""Real standalone CLI subprocess regression; fake JSONL peers never load models."""
import base64
import http.client
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from PIL import Image

HERE = Path(__file__).resolve().parent

ENGINE = '''#!/usr/bin/env python3
import json,sys
from pathlib import Path
print(json.dumps({'event':'ready','context':8192,'vocab':200064}),flush=True)
for line in sys.stdin:
 request=json.loads(line)
 Path(__file__).with_suffix('.audit').write_text(json.dumps(request))
 print(json.dumps({'event':'token','token':10}),flush=True)
 json.loads(sys.stdin.readline())
 print(json.dumps({'event':'done','finish_reason':'length','timings':{'prompt_n':2,'predicted_n':1,'prompt_ms':1,'predicted_ms':1}}),flush=True)
'''
HELPER = '''#!/usr/bin/env python3
import json,sys
print(json.dumps({'event':'ready','vocab':200064,'eog':[2],'suppress':[]}),flush=True)
for line in sys.stdin:
 request=json.loads(line);op=request['op']
 if op=='chat': result={'tokens':[1],'prompt':request['messages'][0]['content'],'grammar_active':False}
 elif op=='piece': result={'bytes':list(b'hello')}
 elif op=='parse': result={'message':{'role':'assistant','content':request['content']}}
 else: result={'error':'unexpected op'}
 print(json.dumps(result),flush=True)
'''
ADAPTER = '''#!/usr/bin/env python3
import json,sys
from pathlib import Path
print(json.dumps({'ready':True}),flush=True)
for line in sys.stdin:
 request=json.loads(line);out=Path(request['out']);out.mkdir()
 manifest={'format':'m31-vision-ipc-v1','source_pr':'6b95a9908d2c7b242817a447708aec38e937e2c0','n_embd':6144,'llm_inference':False,'encoded':True,'encode_seconds':0,'backend':'cpu','rows':2,'visual_rows':0,'chunks':[{'index':0,'row':0,'position':0,'rows':2,'n_pos':2,'type':'text','tokens':[1,3]}]}
 path=out/'manifest.json';path.write_text(json.dumps(manifest))
 print(json.dumps({'ok':True,'manifest':str(path)}),flush=True)
'''


class StandaloneCLITest(unittest.TestCase):
    def test_real_cli_image_data_url_reaches_mixed_engine_without_common_frontend_import(self):
        with tempfile.TemporaryDirectory(prefix='m31-cli-vision-') as directory:
            directory=Path(directory)
            for name,script in [('engine',ENGINE),('helper',HELPER),('adapter',ADAPTER)]:
                path=directory/name
                path.write_text(script)
                path.chmod(0o755)
            for name in ('model','mmproj'):
                (directory/name).write_text('fake peer fixture')
            config={'adapter':str(directory/'adapter'),'mmproj':str(directory/'mmproj'),
                    'cache_dir':str(directory/'cache'),'backend':'cpu','threads':1,'cpu_workers':1}
            (directory/'vision.json').write_text(json.dumps(config))
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            # Cwd and PYTHONPATH intentionally do not provide the repository root.
            env=os.environ.copy();env.pop('PYTHONPATH',None)
            with (directory/'server.log').open('w+') as log:
                proc=subprocess.Popen([sys.executable,str(HERE/'serve_m31.py'),'--engine',str(directory/'engine'),
                    '--tokenizer',str(directory/'helper'),'--model',str(directory/'model'),'--port',str(port),
                    '--vision-config',str(directory/'vision.json')],cwd=directory,env=env,stdout=log,stderr=subprocess.STDOUT)
                connection=None
                try:
                    deadline=time.monotonic()+10
                    while time.monotonic()<deadline:
                        self.assertIsNone(proc.poll(),'standalone exited before readiness')
                        try:
                            connection=http.client.HTTPConnection('127.0.0.1',port,timeout=2)
                            connection.request('GET','/health');response=connection.getresponse();response.read()
                            if response.status==200:break
                        except OSError:time.sleep(.02)
                        finally:
                            if connection:connection.close()
                    else:self.fail('standalone readiness timeout')
                    png=io.BytesIO();Image.new('RGB',(8,8),'red').save(png,format='PNG')
                    body={'messages':[{'role':'user','content':[{'type':'image_url','image_url':{
                          'url':'data:image/png;base64,'+base64.b64encode(png.getvalue()).decode()}},
                          {'type':'text','text':'describe'}]}],'max_tokens':1}
                    connection=http.client.HTTPConnection('127.0.0.1',port,timeout=5)
                    connection.request('POST','/v1/chat/completions',json.dumps(body),{'Content-Type':'application/json'})
                    response=connection.getresponse();result=json.loads(response.read())
                    self.assertEqual(response.status,200,result)
                    self.assertEqual(result['choices'][0]['message']['content'],'hello')
                    audit=json.loads((directory/'engine.audit').read_text())
                    self.assertIn('mixed_manifest',audit)
                    self.assertEqual(audit['tokens'],[1])
                finally:
                    if connection:connection.close()
                    proc.terminate()
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                    log.seek(0)
                    self.assertNotIn('No module named',log.read())


if __name__=='__main__':unittest.main()
