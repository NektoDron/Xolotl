#!/usr/bin/env python3
"""Exercise the frozen CLI on CPU, replacing every process/server constructor."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--tree", type=Path, required=True)
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
shim_path = args.tree / "tools/m31/serve/serve_m31.py"
sys.path.insert(0, str(shim_path.parent))
spec = importlib.util.spec_from_file_location("m31_preflight_common", shim_path)
shim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shim)
config = json.loads(args.config.read_text())
engine_argv, environment, unset = shim.runtime_plan(config, args.config.parent)
captured = []


class FakeRPC:
    def __init__(self, argv, **kwargs):
        captured.append(list(argv))
        self.ready = {"context": 40000, "vocab": 200064}
        if "--backend" in argv:
            assert all(os.environ.get(key) == value for key, value in environment.items())
            assert all(key not in os.environ for key in unset)

    def close(self):
        pass


class FakeEvent:
    def wait(self): pass
    def set(self): pass


fake_http = SimpleNamespace(server_address=("127.0.0.1", 8181), shutdown=lambda: None,
                            server_close=lambda: None)
os.environ["M31_CARD_SESSION"] = "1"  # No constructor below starts an actual process.
with patch.object(shim, "PollingRPC", FakeRPC), \
        patch.object(shim, "make_service", lambda *a: SimpleNamespace(api_key="")), \
        patch.object(shim.S, "serve", lambda *a: fake_http), \
        patch.object(shim.threading, "Event", FakeEvent), \
        patch.object(sys, "argv", [str(shim_path), "--config", str(args.config), "--port", "8181"]):
    shim.main()
assert captured == [config["tokenizer_argv"], config["tokenizer_argv"], engine_argv]
assert engine_argv[engine_argv.index("--ctx") + 1] == "40000"
assert engine_argv[engine_argv.index("--gpu-experts-gib") + 1] == "7"
for key in ("--reuse-allocations", "--persistent-cpu-pool", "--cpu-residual",
            "--cpu-decode-cache", "--dense-decode-cache", "--profile"):
    assert engine_argv[engine_argv.index(key) + 1] == "1", (key, engine_argv)
client_spec = importlib.util.spec_from_file_location("m31_preflight_client", shim_path.with_name("smoke_native_client.py"))
client = importlib.util.module_from_spec(client_spec)
client_spec.loader.exec_module(client)
with tempfile.TemporaryDirectory(prefix="m31-smoke-audit-") as directory:
    audit_path = Path(directory) / "audit.jsonl"
    for tail in (b'{"nested":{}}', b'{"text":"\xf0\x9f', b''):
        audit_path.write_bytes(b'{"complete":true}\n' + tail)
        assert client.read_audit(audit_path) == [{"complete": True}]
with args.output.open("x") as stream:
    json.dump({"format": "m31-common-native-cli-preflight-v1", "pass": True, "gpu_started": False,
               "process_constructors_mocked": True, "captured_argv": captured,
               "environment": environment, "unset_environment": unset,
               "environment_count": len(environment), "audit_partial_tail_cases_pass": 3}, stream, indent=2)
    stream.write("\n")
