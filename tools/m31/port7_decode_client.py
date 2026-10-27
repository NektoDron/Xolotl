#!/usr/bin/env python3
"""Small deterministic HTTP decode fixture; preserves responses, not raw token IDs."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--repeat', type=int, default=2)
    args = parser.parse_args()
    if args.tokens < 1 or args.repeat < 1:
        parser.error('tokens and repeat must be positive')
    body = {'messages': [{'role': 'user', 'content':
            'Write a precise Python implementation of merge sort, then explain its time and space complexity.'}],
            'temperature': 0, 'max_tokens': args.tokens, 'ignore_eos': True,
            'stream': False, 'cache_prompt': False, 'reasoning_effort': 'low'}
    rows = []
    # Never silently overwrite an earlier measurement, even if the server fails.
    with args.out.open('x') as stream:
        for repetition in range(args.repeat):
            request = urllib.request.Request(f'http://127.0.0.1:{args.port}/v1/chat/completions',
                        json.dumps(body).encode(), {'Content-Type': 'application/json'})
            start = time.perf_counter()
            with urllib.request.urlopen(request, timeout=7200) as response:
                result = json.load(response)
            message = result['choices'][0]['message']
            message_bytes = json.dumps(message, ensure_ascii=False, sort_keys=True,
                                       separators=(',', ':')).encode()
            row = {'repetition': repetition, 'request': body, 'response': result,
                   'wall_seconds': time.perf_counter() - start,
                   'message_sha256': hashlib.sha256(message_bytes).hexdigest(),
                   'scope': 'Message JSON hash; not a raw token-ID or full-logit comparison.'}
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            print(json.dumps({'repetition': repetition, 'message_sha256': row['message_sha256'],
                              'timings': result.get('timings'), 'usage': result.get('usage')}, indent=2), flush=True)
    if len({row['message_sha256'] for row in rows}) != 1:
        raise RuntimeError('identical greedy requests produced different message hashes')
    if any(row['response']['usage']['completion_tokens'] != args.tokens for row in rows):
        raise RuntimeError('short deterministic decode fixture')


if __name__ == '__main__':
    main()
