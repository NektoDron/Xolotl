#!/usr/bin/env python3
"""Real native-sampling HTTP gate for lazy tool grammar; never execute tool calls."""
import argparse
import json
from pathlib import Path
import sys

from server_smoke import Smoke

TOOLS = [{'type': 'function', 'function': {'name': 'get_weather', 'description': 'Get current weather in a city.',
          'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'},
                        'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']}}, 'required': ['city', 'unit']}}}]


class ToolSmoke(Smoke):
    def body(self, text, maximum=256, stream=False, thinking=False):
        return {'messages': [{'role': 'user', 'content': text}], 'tools': TOOLS,
                'tool_choice': 'auto', 'temperature': 0, 'max_tokens': maximum, 'stream': stream,
                'chat_template_kwargs': {'enable_thinking': thinking}}

    def tool(self, name, response):
        choice = response['choices'][0]
        self.check(name + ': tool finish', choice.get('finish_reason') == 'tool_calls', choice)
        calls = choice.get('message', {}).get('tool_calls') or []
        self.check(name + ': one call', len(calls) == 1, calls)
        call = calls[0]
        self.check(name + ': stable call id', bool(call.get('id')), call)
        self.check(name + ': function name', call.get('function', {}).get('name') == 'get_weather', call)
        args = json.loads(call['function']['arguments'])
        self.check(name + ': Unicode city', args.get('city', '').casefold() in ('казань', 'казани', 'kazan'), args)
        self.check(name + ': enum unit', args.get('unit') == 'celsius', args)
        self.check(name + ': native grammar checked', response.get('timings', {}).get('grammar_checks', 0) > 0, response.get('timings'))

    def run(self):
        prompt = 'Call get_weather for Казань with unit celsius. Use the provided tool; do not invent weather data.'
        plain = self.post_ok('auto-plain', '/v1/chat/completions', self.body('What is 17*23? Reply only with the number.'))
        content, _ = self.text_fields('auto-plain', plain['choices'][0]['message'])
        self.check('auto permits plain391', '391' in content, content)
        self.check('auto-plain grammar checked', plain.get('timings', {}).get('grammar_checks', 0) > 0, plain.get('timings'))
        response = self.post_ok('nonstream-tool', '/v1/chat/completions', self.body(prompt))
        self.tool('nonstream-tool', response)
        streamed = self.request('SSE-tool', '/v1/chat/completions', self.body(prompt, stream=True), stream=True)
        self.check('SSE tool HTTP200', streamed['status'] == 200, streamed)
        self.check('SSE tool DONE', streamed.get('done'), streamed)
        calls, final = [], None
        for event in streamed.get('events', []):
            choice = event.get('choices', [{}])[0]
            calls.extend(choice.get('delta', {}).get('tool_calls') or [])
            if choice.get('finish_reason') is not None:
                final = event
        self.check('SSE tool final', final is not None, streamed)
        assembled = {'choices': [{'message': {'tool_calls': calls}, 'finish_reason': final['choices'][0]['finish_reason']}],
                     'timings': final.get('timings', {})}
        self.tool('SSE-tool', assembled)
        cancelled = self.request('cancel-tool-thinking', '/v1/chat/completions',
                                 self.body('Think carefully about the weather in Казань before calling the weather tool.',
                                           maximum=512, stream=True, thinking=True), stream=True, cancel=True)
        self.check('cancel during grammar request', cancelled.get('cancelled'), cancelled)
        recovered = self.post_ok('tool-after-cancel', '/v1/chat/completions', self.body(prompt))
        self.tool('tool-after-cancel', recovered)
        health = self.request('health-final', '/health')
        self.check('health after grammar cancel', health['status'] == 200, health)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, default=8180)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--timeout', type=float, default=600)
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as output:
        smoke = ToolSmoke(f'http://127.0.0.1:{a.port}', a.timeout)
        error = None
        try:
            smoke.run()
        except Exception as e:
            error = f'{type(e).__name__}: {e}'
        result = {'format': 'm31-tool-http-smoke-v1', 'ok': error is None, 'error': error,
                  'checks': smoke.checks, 'requests': smoke.requests}
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps({'ok': result['ok'], 'checks': len(smoke.checks), 'error': error, 'out': str(a.out)}, ensure_ascii=False))
    return 0 if error is None else 1


if __name__ == '__main__':
    sys.exit(main())
