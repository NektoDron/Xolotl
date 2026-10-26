#!/usr/bin/env python3
"""Batch GPU worker started ONLY by its own aictl preset; health lasts until down."""
import http.server
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

job = Path(sys.argv[1]).resolve()
spec = json.loads(job.read_text())
result_path = job.with_suffix(".result.json")
if result_path.exists():
    raise RuntimeError("job already has a result; use a new job name")
os.environ.update(CUDA_VISIBLE_DEVICES="0", M31_CARD_SESSION="1", OPENBLAS_NUM_THREADS="1",
                  GGML_CUDA_DISABLE_FUSION="1")
state = {"status": "ok", "phase": "starting"}


class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(state).encode())

    def log_message(self, *args):
        pass


server = http.server.ThreadingHTTPServer(("127.0.0.1", 8179), Health)
threading.Thread(target=server.serve_forever, daemon=True).start()
result = {"commands": [], "returncode": 0}
try:
    for i, command in enumerate(spec["commands"]):
        state.update(phase="running", command=i)
        log = job.parent / f"{job.stem}-{i:02d}.log"
        with log.open("x") as stream:
            proc = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        result["commands"].append({"argv": command, "log": str(log), "returncode": proc.returncode})
        if proc.returncode:
            result["returncode"] = proc.returncode
            break
except Exception as exc:
    result.update(returncode=1, error=str(exc))
pending_result = job.with_suffix(".result.tmp")
with pending_result.open("x") as stream:
    json.dump(result, stream, indent=2)
os.replace(pending_result, result_path)
state.update(phase="finished", returncode=result["returncode"])
# Retain health and ownership until the controller has observed the result and calls aictl down.
threading.Event().wait()
