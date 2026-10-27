#!/usr/bin/env python3
"""Audit-only entry point for a frozen common M31 smoke tree."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time

tree = Path(os.environ["M31_COMMON_SMOKE_TREE"])
shim_path = tree / "tools/m31/serve/serve_m31.py"
sys.path.insert(0, str(shim_path.parent))
spec = importlib.util.spec_from_file_location("m31_smoke_common", shim_path)
shim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shim)
audit = Path(os.environ["M31_COMMON_SMOKE_AUDIT"])
lock = threading.Lock()


def record(value):
    with lock, audit.open("a") as stream:
        stream.write(json.dumps({"monotonic": time.monotonic(), **value}, ensure_ascii=False) + "\n")


class AuditedRPC(shim.PollingRPC):
    def __init__(self, argv, *args, **kwargs):
        self.engine = "--backend" in argv
        self.request_id = 0
        super().__init__(argv, *args, **kwargs)

    def write(self, value):
        if self.engine:
            if "tokens" in value:
                self.request_id += 1
                record({"kind": "request", "request": self.request_id, "params": value})
            elif "accepted" in value:
                record({"kind": "grammar_ack", "request": self.request_id,
                        "accepted": value["accepted"], "allowed_count": len(value.get("allowed", []))})
            elif "continue" in value:
                record({"kind": "token_ack", "request": self.request_id, "continue": value["continue"]})
        return super().write(value)

    def event(self, value):
        if self.engine and value is not None:
            record({"kind": "engine_event", "request": self.request_id, "event": value})
        return value

    def read(self, timeout=None):
        return self.event(super().read(timeout))

    def poll(self, timeout):
        return self.event(super().poll(timeout))


shim.PollingRPC = AuditedRPC
if __name__ == "__main__":
    shim.main()
