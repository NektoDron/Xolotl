#!/usr/bin/env python3
"""Snapshot canonical build inputs and freeze completed CMake binaries with receipts."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from port6_validate_long import fingerprint, write

ROOT = Path('/mnt/ai/Xolotl-m31')


def sources():
    paths = {ROOT / 'CMakeLists.txt', ROOT / 'tools/m31/build.sh'}
    for directory in (ROOT / 'include', ROOT / 'cmake'):
        paths.update(p for p in directory.rglob('*') if p.is_file())
    for suffix in ('*.cpp', '*.hpp', '*.cuh', '*.cu'):
        paths.update((ROOT / 'tools/m31').glob(suffix))
    paths.update((ROOT / 'tools/m31/oracle').glob('*.cpp'))
    paths.add(ROOT / 'tools/m31/oracle/CMakeLists.txt')
    return {str(p.relative_to(ROOT)): fingerprint(p)['sha256'] for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('snapshot', 'freeze'))
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--unit', default='codex-m31-port6-canonical-v8')
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.command == 'snapshot':
        directory.mkdir(parents=True, exist_ok=False)
        write(directory / 'before.json', {'source': sources(), 'unit': args.unit,
              'base_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'timestamp_utc': subprocess.check_output(['date', '-u', '--iso-8601=seconds'], text=True).strip()})
        return
    before = json.loads((directory / 'before.json').read_text())
    state = subprocess.check_output(['systemctl', 'show', args.unit, '-p', 'ActiveState', '-p', 'SubState', '-p', 'ExecMainStatus'], text=True)
    if 'ActiveState=inactive' not in state or 'ExecMainStatus=0' not in state:
        raise ValueError('canonical unit has not completed successfully: ' + state)
    after = sources()
    if before['source'] != after:
        changed = sorted(k for k in set(before['source']) | set(after) if before['source'].get(k) != after.get(k))
        raise ValueError('source changed during canonical build: ' + repr(changed))
    build = ROOT / 'build-m31-cuda'
    elves = {}
    for name in ('m31_generate', 'm31_serve'):
        path = build / name
        original = fingerprint(path)
        target = directory / name
        with path.open('rb') as inp, target.open('xb') as out:
            shutil.copyfileobj(inp, out, 1 << 20)
        target.chmod(0o555)
        if fingerprint(target)['sha256'] != original['sha256']:
            raise ValueError('canonical ELF changed while copying')
        elves[name] = original['sha256']
    copied = {}
    for name in ('CMakeCache.txt', 'compile_commands.json'):
        for mode in ('cpu', 'cuda'):
            source = ROOT / f'build-m31-{mode}' / name
            target = directory / (mode + '-' + name)
            shutil.copyfile(source, target)
            target.chmod(0o444)
            copied[target.name] = fingerprint(target)
    links = {}
    for name in elves:
        links[name] = (build / f'CMakeFiles/{name}.dir/link.txt').read_text()
    vendor = {str(p.relative_to(build)): fingerprint(p)['sha256'] for p in (
        build / 'ggml/src/libggml.a', build / 'ggml/src/libggml-cpu.a',
        build / 'ggml/src/ggml-cuda/libggml-cuda.a', build / 'ggml/src/libggml-base.a')}
    receipt = {'format': 'm31-port6-canonical-cmake-build-v1', 'elf': elves,
               'source': after, 'source_before': before['source'], 'source_stable': True,
               'snapshot_timestamp_utc': before['timestamp_utc'], 'base_head': before['base_head'],
               'build_command': 'tools/m31/build.sh cpu && tools/m31/build.sh cuda',
               'systemd_unit': args.unit, 'unit_result': state, 'cmake_files': copied,
               'link_commands': links, 'vendor_libraries': vendor,
               'compiler': subprocess.check_output(['/usr/bin/g++-14', '--version'], text=True).splitlines()[0],
               'cuda_compiler': subprocess.check_output(['/usr/local/cuda/bin/nvcc', '--version'], text=True),
               'limits': ['Copied canonical CMake outputs; no manual relink.',
                          'Untracked diagnostic code is included in the broad source snapshot.',
                          'Runtime Python adapter is frozen separately after its validation ACK.']}
    write(directory / 'build-receipt.json', receipt)
    print(json.dumps({'directory': str(directory), 'elf': elves}, indent=2))


if __name__ == '__main__':
    main()
