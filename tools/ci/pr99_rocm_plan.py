#!/usr/bin/env python3
"""Read official pinned AMD package metadata; never download or install packages."""
import argparse
import gzip
import hashlib
import json
import re
import urllib.request
from pathlib import Path

URL = 'https://repo.radeon.com/rocm/apt/7.0.2/dists/noble/main/binary-amd64/Packages.gz'
ROOTS = {
    'hip-dev': '7.0.51831.70002-56~24.04',
    'rocm-device-libs': '1.0.0.70002-56~24.04',
    'hipblas-dev': '3.0.2.70002-56~24.04',
    'rocblas-dev': '5.0.2.70002-56~24.04',
    'hipblaslt-dev': '1.0.0.70002-56~24.04',
}
EXPECTED = set('comgr hip-dev hip-runtime-amd hipblas hipblas-common-dev hipblas-dev '
               'hipblaslt hipblaslt-dev hipcc hsa-rocr hsa-rocr-dev rocblas rocblas-dev '
               'rocm-core rocm-device-libs rocm-llvm rocminfo rocprofiler-register '
               'rocsolver roctracer'.split())


def plan(raw):
    packages = {}
    for block in gzip.decompress(raw).decode().split('\n\n'):
        fields, previous = {}, None
        for line in block.splitlines():
            if line.startswith(' ') and previous:
                fields[previous] += ' ' + line.strip()
            elif ': ' in line:
                previous, value = line.split(': ', 1)
                fields[previous] = value
        if fields.get('Package'):
            packages[fields['Package']] = fields
    pins = json.loads(Path(__file__).with_name('pr99_rocm_packages.json').read_text())
    assert set(pins) == EXPECTED
    for name, record in pins.items():
        assert all(packages[name][key] == value for key, value in record.items()), ('Pinned AMD package changed', name)
    for name, version in ROOTS.items():
        assert packages[name]['Version'] == version, ('Pinned root version changed', name)
    seen, todo = set(), list(ROOTS)
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        for dependency in packages[name].get('Depends', '').split(','):
            for alternative in dependency.split('|'):
                candidate = re.split(r'[ (]', alternative.strip())[0]
                if candidate in packages:
                    todo.append(candidate)
                    break
    assert seen == EXPECTED, 'AMD dependency closure changed: review before installing'
    selected = []
    for name in sorted(seen):
        record = packages[name]
        assert re.fullmatch(r'[a-z0-9+.-]+', name)
        assert re.fullmatch(r'[0-9A-Za-z.+:~_-]+', record['Version'])
        assert re.fullmatch(r'[0-9a-f]{64}', record['SHA256'])
        selected.append({key: record.get(key) for key in
                         ('Package', 'Version', 'Size', 'Installed-Size', 'SHA256', 'Filename', 'Depends')})
    unknown = [record for record in selected if record['Installed-Size'] is None]
    # An explicit planning allowance, not a measurement: AMD's rocm-llvm record
    # lacks Installed-Size. The CI also measures disk after the real installation.
    return {
        'metadata_url': URL, 'metadata_sha256': hashlib.sha256(raw).hexdigest(),
        'roots': ROOTS, 'packages': selected,
        'amd_archive_bytes': sum(int(record['Size']) for record in selected),
        'known_installed_bytes': sum(int(record['Installed-Size'] or 0) * 1024 for record in selected),
        'unknown_installed_packages': [record['Package'] for record in unknown],
        'unknown_installed_planning_allowance_bytes': sum(int(record['Size']) * 4 for record in unknown),
        'allowance_scope': '4x compressed bytes for missing Installed-Size; estimate, not measured footprint',
        'scope': 'AMD closure only; apt simulation/URI plan additionally records Ubuntu dependencies',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--packages', type=Path, required=True)
    args = parser.parse_args()
    with urllib.request.urlopen(URL, timeout=30) as response:
        raw = response.read((1 << 20) + 1)
    assert len(raw) <= 1 << 20, 'Unexpected metadata size'
    result = plan(raw)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    args.packages.write_text(''.join(record['Package'] + '=' + record['Version'] + '\n'
                                   for record in result['packages']))
    print(json.dumps({key: value for key, value in result.items() if key != 'packages'}, indent=2))


if __name__ == '__main__':
    main()
