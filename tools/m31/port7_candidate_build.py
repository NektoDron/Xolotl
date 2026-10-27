#!/usr/bin/env python3
"""Freeze port7 sources, rebuild CPU-quant CUDA TU, reuse other v8 libraries.

Invoke through systemd User=nekto under a shared .build.lock, outside speed windows.
The canonical CMake build remains the final release check.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/mnt/ai/Xolotl-m31'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--with-research', action='store_true',
                        help='Also link DSpark, calibration and prefix research probes')
    parser.add_argument('--freeze-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--cuda-override', type=Path,
                        help='Isolated backend directory containing vendor/ggml/include and libggml-cuda.a')
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / 'build-m31-cuda'
    out = args.out.resolve()
    cuda_override = args.cuda_override.resolve() if args.cuda_override else None
    if cuda_override and not (cuda_override / 'libggml-cuda.a').is_file():
        raise RuntimeError('Missing CUDA override archive')
    source = out / 'source'
    if args.resume:
        hashes = json.loads((out / 'source-sha256.json').read_text())
        actual = {str(p.relative_to(source)): sha(p) for p in sorted(source.iterdir())}
        if hashes != actual:
            raise RuntimeError('Frozen source does not match its receipt')
    else:
        out.mkdir(exist_ok=False, parents=True)
        source.mkdir()
        for path in sorted((root / 'tools/m31').iterdir()):
            if path.is_file() and path.suffix in ('.cpp', '.hpp', '.cuh', '.cu', '.inc'):
                shutil.copy2(path, source / path.name)
        hashes = {str(path.relative_to(source)): sha(path) for path in sorted(source.iterdir())}
        (out / 'source-sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
        (out / 'source-diff.patch').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=root))
        (out / 'base-head.txt').write_text(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True))
    if args.freeze_only:
        print(json.dumps({'frozen': str(source), 'files': len(hashes)}))
        return
    entries = json.loads((base / 'compile_commands.json').read_text())
    commands = {}

    # GpuDeviceDots has private scratch and exact IQ3 methods now. Reusing its
    # old object with new helper headers would silently mix implementations.
    # Compile this one CUDA TU first; the subsequent CXX pool owns three slots.
    quant_entry = next(e for e in entries if e['file'] == str(root / 'tools/m31/gpu_cpu_quants.cu'))
    cuda_argv = shlex.split(quant_entry['command'])
    expanded = []
    index = 0
    while index < len(cuda_argv):
        if cuda_argv[index] == '--options-file':
            options = Path(quant_entry['directory']) / cuda_argv[index + 1]
            expanded.extend(shlex.split(options.read_text()))
            index += 2
        else:
            expanded.append(cuda_argv[index])
            index += 1
    cuda_argv = [('-I' + str(source)) if a == '-I' + str(root / 'tools/m31') else a for a in expanded]
    cuda_argv[1:1] = ['-O3', '--ftz=false', '--fmad=true', '--prec-div=true', '--prec-sqrt=true']
    if cuda_override:
        cuda_argv.insert(1, '-I' + str(cuda_override / 'vendor/ggml/include'))
    cuda_argv[cuda_argv.index('-o') + 1] = str(out / 'gpu_cpu_quants.o')
    cuda_argv[cuda_argv.index('-c') + 1] = str(source / 'gpu_cpu_quants.cu')
    commands['gpu_cpu_quants'] = cuda_argv
    subprocess.run(cuda_argv, cwd=quant_entry['directory'], check=True)

    def compile_one(name):
        template = name if name in ('m31_engine', 'm31_gpu_experts', 'm31_generate', 'm31_serve') else (
            'm31_engine' if name in ('m31_dspark', 'gpu_cached_exact') else 'm31_generate')
        entry = next(e for e in entries if e['file'] == str(root / f'tools/m31/{template}.cpp'))
        argv = shlex.split(entry['command'])
        if cuda_override:
            argv.insert(1, '-I' + str(cuda_override / 'vendor/ggml/include'))
        argv.insert(1, '-I' + str(root / 'build-m31-cpu/_deps/strata_llamacpp-src/vendor/nlohmann'))
        if name == 'gpu_cached_exact':
            argv[1:1] = ['-frounding-math', '-fno-fast-math']
        argv = [('-I' + str(source)) if a == '-I' + str(root / 'tools/m31') else a for a in argv]
        argv[argv.index('-o') + 1] = str(out / (name + '.o'))
        argv[-1] = str(source / (name + '.cpp'))
        commands[name] = argv
        subprocess.run(argv, cwd=entry['directory'], check=True)

    # At most three compiler processes; NVCC above has already exited.
    targets = ['m31_generate', 'm31_serve']
    if args.with_research:
        targets += ['m31_dspark_generate', 'port7_decode_calibrate', 'port7_prefix_raw_check',
                    'port7_verify_dump', 'port7_capture_raw_check']
    names = ['m31_engine', 'm31_gpu_experts', 'm31_dspark'] + targets
    extra_engine_objects = ['gpu_cpu_quants']
    if (source / 'gpu_cached_exact.cpp').exists():
        names += ['gpu_cached_exact']
        extra_engine_objects += ['gpu_cached_exact']
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(compile_one, names))
    links = {}
    for name in targets:
        template = name if name in ('m31_generate', 'm31_serve') else 'm31_generate'
        argv = shlex.split((base / f'CMakeFiles/{template}.dir/link.txt').read_text())
        if cuda_override:
            argv[argv.index('ggml/src/ggml-cuda/libggml-cuda.a')] = str(cuda_override / 'libggml-cuda.a')
        ix = argv.index('-o')
        argv[ix - 1] = str(out / (name + '.o'))
        argv[ix + 1] = str(out / name)
        ix = argv.index('libm31_engine.a')
        argv[ix:ix] = [str(out / (obj + '.o')) for obj in
                       ['m31_engine', 'm31_gpu_experts'] + extra_engine_objects]
        if name in ('m31_dspark_generate', 'm31_serve'):
            argv.insert(ix, str(out / 'm31_dspark.o'))
        subprocess.run(argv, cwd=base, check=True)
        links[name] = argv
    after = {str(path.relative_to(source)): sha(path) for path in sorted(source.iterdir())}
    if hashes != after:
        raise RuntimeError('Frozen source changed during build')
    libraries = [base / 'libm31_engine.a'] + list((base / 'ggml/src').rglob('*.a'))
    if cuda_override:
        libraries += [cuda_override / 'libggml-cuda.a']
    receipt = {'format': 'm31-port7-incremental-build-v1',
               'base_head': (out / 'base-head.txt').read_text().strip(),
               'source': hashes, 'compile': commands, 'link': links,
               'libraries': {str(p): sha(p) for p in libraries},
               'elf': {name: sha(out / name) for name in links},
               'cuda_override': str(cuda_override) if cuda_override else None,
               'cuda_header_sha256': sha(cuda_override / 'vendor/ggml/include/ggml-cuda.h') if cuda_override else None,
               'scope': 'Incremental candidate; gpu_cpu_quants CUDA TU rebuilt coherently, other CUDA objects/libraries reused from v8 or recorded override. Final canonical CMake build still required.'}
    (out / 'build-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'build': str(out), 'elf': receipt['elf']}, indent=2))


if __name__ == '__main__':
    main()
