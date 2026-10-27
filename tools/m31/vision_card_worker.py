#!/usr/bin/env python3
"""Batch GPU worker started ONLY by its own aictl preset; health lasts until down."""
import http.server
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import threading
import time

job = Path(sys.argv[1]).resolve()
spec = json.loads(job.read_text())
result_path = job.with_suffix(".result.json")
if result_path.exists():
    raise RuntimeError("job already has a result; use a new job name")
os.environ.update(CUDA_VISIBLE_DEVICES="0", M31_CARD_SESSION="1", OPENBLAS_NUM_THREADS="1",
                  GGML_CUDA_DISABLE_FUSION="1")
state = {"status": "ok", "phase": "starting"}
command_timeout = float(spec.get("command_timeout_seconds", 86400))
controller_timeout = float(spec.get("controller_wait_seconds", 600))
if not 0 < command_timeout <= 604800 or not 0 < controller_timeout <= 3600:
    raise ValueError("invalid command/controller timeout")


class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(state).encode())

    def log_message(self, *args):
        pass


server = http.server.ThreadingHTTPServer(("127.0.0.1", 8189), Health)
threading.Thread(target=server.serve_forever, daemon=True).start()
result = {"commands": [], "returncode": 0}
stopping = threading.Event()
active_process = None

def request_stop(signum, _frame):
    stopping.set()
    proc = active_process
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

signal.signal(signal.SIGTERM, request_stop)
signal.signal(signal.SIGINT, request_stop)
try:
    for i, command in enumerate(spec["commands"]):
        state.update(phase="running", command=i)
        log = job.parent / f"{job.stem}-{i:02d}.log"
        with log.open("x") as stream:
            proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            active_process = proc
            deadline = time.monotonic() + command_timeout
            try:
                while proc.poll() is None:
                    if stopping.is_set() or time.monotonic() >= deadline:
                        try:
                            os.killpg(proc.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                        if stopping.is_set():
                            raise InterruptedError(f"command {i} stopped")
                        raise TimeoutError(f"command {i} exceeded {command_timeout} seconds")
                    try:
                        proc.wait(timeout=min(1.0, max(.01, deadline - time.monotonic())))
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                active_process = None
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
stopping.wait(controller_timeout)
server.shutdown()
