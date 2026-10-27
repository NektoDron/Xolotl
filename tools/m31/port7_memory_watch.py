#!/usr/bin/env python3
"""Run one aictl-owned child with lightweight physical-GPU0 NVML memory samples."""
import argparse
import ctypes
import ctypes.util
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time


class Memory(ctypes.Structure):
    _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong),
                ('used', ctypes.c_ulonglong)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=.1)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or not .02 <= args.interval <= 10:
        parser.error('require child command and interval .02..10 seconds')
    if os.environ.get('M31_CARD_SESSION') != '1':
        parser.error('requires the authorized aictl card worker')
    if args.out.exists() or args.summary.exists():
        parser.error('use fresh sample and summary paths')
    nvml = ctypes.CDLL(ctypes.util.find_library('nvidia-ml') or 'libnvidia-ml.so.1')
    nvml.nvmlInit_v2.restype = ctypes.c_int
    nvml.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
    nvml.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
    nvml.nvmlDeviceGetMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Memory)]
    nvml.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int

    def checked(code):
        if code:
            raise RuntimeError(f'NVML error {code}')

    checked(nvml.nvmlInit_v2())
    handle = ctypes.c_void_p()
    checked(nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)))
    stop = threading.Event()
    samples = []
    errors = []
    child = None

    def forward(signum, _frame):
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    def watch(stream):
        while True:
            try:
                value = Memory()
                checked(nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(value)))
                row = {'monotonic_ns': time.monotonic_ns(), 'wall_ns': time.time_ns(),
                       'physical_gpu': 0, 'total_bytes': value.total,
                       'free_bytes': value.free, 'used_bytes': value.used}
                samples.append(row)
                stream.write(json.dumps(row) + '\n')
                stream.flush()
            except Exception as error:
                errors.append(str(error))
                break
            if stop.wait(args.interval):
                break

    try:
        signal.signal(signal.SIGTERM, forward)
        signal.signal(signal.SIGINT, forward)
        with args.out.open('x') as stream:
            watcher = threading.Thread(target=watch, args=(stream,), daemon=True)
            watcher.start()
            try:
                child = subprocess.Popen(command)
                rc = child.wait()
            finally:
                stop.set()
                watcher.join(timeout=3)
        result = {'format': 'm31-port7-nvml-memory-watch-v1', 'command': command,
                  'child_returncode': rc, 'interval_seconds': args.interval,
                  'samples': len(samples), 'watch_errors': errors,
                  'gpu0_peak_used_bytes': max((row['used_bytes'] for row in samples), default=None),
                  'gpu0_min_free_bytes': min((row['free_bytes'] for row in samples), default=None),
                  'scope': 'Physical GPU0 total NVML memory; includes driver/other processes. Sampling can miss short allocation peaks.'}
        with args.summary.open('x') as output:
            json.dump(result, output, indent=2)
            output.write('\n')
        return rc if rc else int(bool(errors) or not samples)
    finally:
        nvml.nvmlShutdown()


if __name__ == '__main__':
    raise SystemExit(main())
