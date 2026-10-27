#!/usr/bin/env python3
"""Check isolated Python launcher argv without subprocesses or network sockets."""
import argparse
import contextlib
import importlib.util
import io
from pathlib import Path
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("dspark_launcher_fixture", args.launcher)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    checks = 0
    def check(ok, why):
        nonlocal checks
        if not ok: raise AssertionError(why)
        checks += 1
    class FakeRPC:
        calls = []
        def __init__(self, argv, **kwargs):
            self.calls.append(argv)
            self.ready = {"context": 128, "vocab": 64, "eog": []}
        def close(self): pass
    class FakeServer:
        def __init__(self, *args, **kwargs): pass
        def serve_forever(self): raise KeyboardInterrupt
        def server_close(self): pass
    base = [str(args.launcher), "--engine", "engine-fixture", "--tokenizer", "tokenizer-fixture", "--model", "target-fixture"]
    def run(extra):
        FakeRPC.calls = []
        with patch.object(module, "RPC", FakeRPC), patch.object(module, "ThreadingHTTPServer", FakeServer), \
             patch.object(module.signal, "signal"), patch.object(module.sys, "argv", base + extra), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            module.main()
        return FakeRPC.calls
    calls = run([])
    check(len(calls) == 2 and calls[0] == ["tokenizer-fixture", "target-fixture"], "default tokenizer argv")
    check(not any(arg.startswith("--dspark") for arg in calls[1]), "default worker argv enables DSpark")
    calls = run(["--dspark-model", "draft sidecar.gguf", "--dspark-depth", "5", "--dspark-p-min", "0.25"])
    check(calls[1][-6:] == ["--dspark-model", "draft sidecar.gguf", "--dspark-depth", "5", "--dspark-p-min", "0.25"], "explicit DSpark argv")
    calls = run(["--prefix-cache", "--dspark-model", "draft.gguf"])
    check(calls[1][-8:] == ["--prefix-cache", "1", "--dspark-model", "draft.gguf", "--dspark-depth", "3", "--dspark-p-min", "0.0"], "prefix opt-in argv")
    for extra in (["--dspark-depth", "0"], ["--dspark-depth", "8"], ["--dspark-p-min", "nan"], ["--dspark-p-min", "1.1"]):
        try: run(extra)
        except SystemExit as error: check(error.code == 2 and not FakeRPC.calls, "invalid config started subprocess")
        else: raise AssertionError("invalid config accepted")
    import json
    print(json.dumps({"pass": True, "checks": checks, "scope": "Python launcher argv only, no subprocess/model/network"}))

if __name__ == "__main__": main()
