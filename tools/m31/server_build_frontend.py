#!/usr/bin/env python3
"""Link only the native server against the coordinator's completed CPU engine."""
import fcntl
import json
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parents[2]
source_build = root / 'build-cpu-decode'
out = root / 'build-server-eval'
out.mkdir(exist_ok=True)
with Path('/mnt/ai/logs/m31-port/server-build-frontend.log').open('a') as log:
    with Path('/mnt/ai/logs/m31-port/.build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        flags = {}
        for line in (source_build / 'CMakeFiles/m31_serve.dir/flags.make').read_text().splitlines():
            if ' = ' in line:
                key, value = line.split(' = ', 1)
                flags[key] = shlex.split(value)
        command = ['/usr/bin/g++-14', *flags['CXX_DEFINES'], *flags['CXX_INCLUDES'], *flags['CXX_FLAGS'],
                   '-c', str(root / 'tools/m31/m31_serve.cpp'), '-o', str(out / 'm31_serve_frontend.o')]
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        link = shlex.split((source_build / 'CMakeFiles/m31_serve.dir/link.txt').read_text())
        final = []
        for i, value in enumerate(link):
            if value.startswith('-Wl,--dependency-file='):
                continue
            if value.endswith('m31_serve.cpp.o'):
                value = str(out / 'm31_serve_frontend.o')
            elif i and link[i - 1] == '-o':
                value = str(out / 'm31_serve')
            elif value == 'libm31_engine.a':
                value = str(source_build / value)
            final.append(value)
        subprocess.run(final, cwd=source_build, check=True, stdout=log, stderr=subprocess.STDOUT)
        subprocess.run(['/usr/bin/g++-14', '-std=c++20', '-O3', str(root / 'tools/m31/server_sampling_check.cpp'),
                        '-o', str(out / 'server_sampling_check')], check=True, stdout=log, stderr=subprocess.STDOUT)
        subprocess.run([str(out / 'server_sampling_check'),
                        '/mnt/ai/logs/m31-port/cpu-dense-code128-v1/code128/logits.f32'],
                       check=True, stdout=log, stderr=subprocess.STDOUT)
        subprocess.run(['/usr/bin/python3', str(root / 'tools/m31/test_server.py')], check=True, stdout=log, stderr=subprocess.STDOUT)
        log.write('Native frontend build and greedy regression checks complete\n')
        log.flush()
