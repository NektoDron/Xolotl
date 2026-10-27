#!/usr/bin/env python3
"""No-GPU validation of HTTP session profile translation and structured tool checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import vision_http_session as session
from profile_plan import load_profile


class SessionTests(unittest.TestCase):
    def test_native_recipe_becomes_valid_standalone_cli_with_exact_context(self):
        profile = load_profile(Path(__file__).with_name('profiles')/'native-gpu.json')
        arguments = session.standalone_arguments(profile['engine_arguments'],8192,None)
        self.assertEqual(arguments[arguments.index('--ctx')+1], '8192')
        self.assertIn('--dense-decode-cache',arguments)
        self.assertEqual(arguments[arguments.index('--gpu-experts-gib')+1], '7')
        self.assertEqual(session.standalone_arguments(['--profile','0','--ctx','40000'],8192,None),['--ctx','8192'])
        with self.assertRaises(ValueError):
            session.standalone_arguments(['--profile','yes'],8192,None)

    def test_tool_smoke_requires_actual_named_call_and_exact_nested_json(self):
        calls = []
        def correct(endpoint,body,timeout):
            calls.append(body)
            name = body['tools'][0]['function']['name']
            values = {'get_weather':{'city':'Paris'},'add':{'a':7,'b':5},
                      'set_config':{'config':{'enabled':True,'labels':['red','blue']}}}
            return {'choices':[{'message':{'tool_calls':[{'function':{'name':name,'arguments':json.dumps(values[name])}}]}}]}
        with tempfile.TemporaryDirectory(prefix='m31-session-test-') as directory:
            output = Path(directory)/'tools.json'
            with patch.object(session,'json_request',correct):
                self.assertTrue(session.tools_smoke('http://127.0.0.1:1',output,1))
            self.assertEqual(len(calls),3)
            self.assertTrue(json.loads(output.read_text())['pass'])
            with patch.object(session,'json_request',return_value={'choices':[{'message':{'content':'I would call add'}}]}):
                self.assertFalse(session.tools_smoke('http://127.0.0.1:1',output,1))
            self.assertFalse(json.loads(output.read_text())['pass'])

    def test_low_reference_requests_match_ref_effort_prompt_order_and_report_cache_scope(self):
        sent = []
        def response(endpoint,body,timeout):
            sent.append(body)
            return {'ttft_s':2,'content':'red circle','events':[]}
        with tempfile.TemporaryDirectory(prefix='m31-session-test-') as directory:
            output = Path(directory)/'low.json'
            with patch.object(session.V,'request',response):
                self.assertTrue(session.low_reference('http://127.0.0.1:1',output,1))
            self.assertEqual(len(sent),2)
            for request in sent:
                self.assertEqual(request['reasoning_effort'],'low')
                self.assertEqual(request['max_tokens'],1200)
                self.assertEqual(request['messages'][0]['content'][0]['type'],'text')
                self.assertFalse(request['cache_prompt'])
            self.assertIn('warm',json.loads(output.read_text())['cases'][0]['encoder_state'])


if __name__=='__main__':
    unittest.main()
