#!/usr/bin/env python3
"""Resolve a portable M31 profile and print a reviewable plan; never execute it."""
import argparse
import json
from pathlib import Path
import re


def load_profile(path, _parents=()):
    path = Path(path).resolve()
    if path in _parents or len(_parents) >= 8:
        raise ValueError('profile inheritance cycle or depth exceeds8')
    profile = json.loads(path.read_text())
    if not isinstance(profile, dict) or profile.get('format') != 'm31-runtime-profile-v1':
        raise ValueError('unsupported M31 profile format')
    inherited = {}
    if 'inherits' in profile:
        parent = profile['inherits']
        if not isinstance(parent, str) or not parent or Path(parent).is_absolute():
            raise ValueError('inherits must be a nonempty relative profile path')
        inherited = load_profile(path.parent / parent, (*_parents, path))
    merged = {**inherited, **profile}
    for field in ('environment', 'requires'):
        values = profile.get(field, {})
        if not isinstance(values, dict):
            raise ValueError(field + ' must be an object')
        merged[field] = {**inherited.get(field, {}), **values}
    unset = profile.get('unset_environment', [])
    if not isinstance(unset, list) or not all(isinstance(key, str) for key in unset):
        raise ValueError('unset_environment must be an array of strings')
    merged['unset_environment'] = list(dict.fromkeys([*inherited.get('unset_environment', []), *unset]))
    for key, value in merged['environment'].items():
        if not isinstance(key, str) or not re.fullmatch(r'[A-Z][A-Z0-9_]*', key) or not isinstance(value, str):
            raise ValueError('environment requires uppercase names and string values')
    for key in merged['unset_environment']:
        if not isinstance(key, str) or not re.fullmatch(r'[A-Z][A-Z0-9_]*', key):
            raise ValueError('invalid unset environment name')
        if key in merged['environment']:
            raise ValueError('environment key is also unset: ' + key)
    arguments = merged.get('engine_arguments')
    if not isinstance(arguments, list) or not all(isinstance(item, str) for item in arguments):
        raise ValueError('engine_arguments must be an array of strings')
    if '--model' in arguments or '-m' in arguments or not arguments or not arguments[0].startswith('--'):
        raise ValueError('profile arguments must exclude model and executable')
    return merged


def make_plan(profile, engine, model):
    if not engine or not model:
        raise ValueError('engine and model are required separately from the profile')
    return {'format': 'm31-runtime-plan-v1', 'profile': profile['name'],
            'argv': [str(engine), '--model', str(model), *profile['engine_arguments']],
            'environment': profile['environment'], 'unset_environment': profile['unset_environment'],
            'requires': profile['requires'], 'execute': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--engine', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        plan = make_plan(load_profile(args.profile), args.engine, args.model)
        rendered = json.dumps(plan, ensure_ascii=False, indent=2) + '\n'
        if args.output:
            with args.output.open('x') as stream:
                stream.write(rendered)
        else:
            print(rendered, end='')
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
