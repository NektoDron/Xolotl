#!/usr/bin/env python3
"""Run the original 20 tool cases + 3 continuations, without its GSM8K entrypoint.

Load definitions via AST up to the original final top-level with statement.
Never edit the reference script or execute generated tool calls.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def load_definitions(source, name, port, directory):
    data = source.read_bytes()
    tree = ast.parse(data, filename=str(source))
    final = next((i for i, node in enumerate(tree.body) if isinstance(node, ast.With)), None)
    if final is None or final != len(tree.body) - 1:
        raise ValueError('reference entrypoint must be a single final top-level with statement')
    namespace = {'__name__': 'm31_original_tools23', '__file__': str(source)}
    original_argv = sys.argv
    try:
        sys.argv = [str(source), name, str(port), str(directory), '0']
        exec(compile(ast.Module(body=tree.body[:final], type_ignores=[]), str(source), 'exec'), namespace)
    finally:
        sys.argv = original_argv
    if len(namespace['CASES']) != 20 or len(namespace['FOLLOW']) != 3:
        raise ValueError('unexpected reference tool fixture count')
    return namespace, hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('/home/nekto/eval_duty_1023.py'))
    p.add_argument('--port', type=int, default=8180)
    p.add_argument('--name', required=True)
    p.add_argument('--directory', type=Path, required=True)
    a = p.parse_args()
    if not a.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in a.name):
        p.error('invalid name')
    if not 1 <= a.port <= 65535:
        p.error('invalid port')
    a.directory.mkdir(parents=True, exist_ok=True)
    output_path = a.directory / (a.name + '.jsonl')
    raw_path = a.directory / (a.name + '.raw.jsonl')
    meta_path = a.directory / (a.name + '.summary.json')
    # Reserve all reports before any request; retain partial results after a failure.
    with output_path.open('x') as output, raw_path.open('x') as raw, meta_path.open('x') as meta:
        os.environ.update(EVAL_TEMP='1.0', EVAL_TOPP='0.95', EVAL_MAX='6000')
        os.environ.pop('EVAL_TOPK', None)
        os.environ.pop('EVAL_MINP', None)
        namespace, sha = load_definitions(a.source, a.name, a.port, a.directory)
        original_chat = namespace['chat']
        def recorded_chat(*args, **kwargs):
            response = original_chat(*args, **kwargs)
            raw.write(json.dumps({'messages': args[0], 'tools': kwargs.get('tools'), 'choice': response[0],
                                  'usage': response[1], 'seconds': response[2]}, ensure_ascii=False) + '\n')
            raw.flush()
            return response
        namespace['chat'] = recorded_chat
        started = time.monotonic()
        result = {'source': str(a.source), 'sha256': sha, 'expected': 23,
                  'gsm8k_run': False, 'tool_execution': False,
                  'sampling': {'temperature': 1.0, 'top_p': .95, 'max_tokens': 6000, 'thinking': 'model default'}}
        try:
            good, total = namespace['tools'](output)
            result.update(complete=total == 23, correct=good, total=total)
        except BaseException as error:
            result.update(complete=False, error=f'{type(error).__name__}: {error}')
            raise
        finally:
            result['wall_seconds'] = time.monotonic() - started
            json.dump(result, meta, ensure_ascii=False, indent=2)
            meta.write('\n')
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['complete'] else 1


if __name__ == '__main__':
    sys.exit(main())
