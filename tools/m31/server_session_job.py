#!/usr/bin/env python3
"""Run one HTTP server and sequential clients as descendants of the aictl worker.

The outer card_session.sh owns card/build locks, preset backup and cleanup.
This helper never invokes aictl or reserves a card by itself.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.request


def main():
    if os.environ.get("M31_CARD_SESSION") != "1":
        raise RuntimeError("requires the aictl card-session worker")
    spec_path = Path(sys.argv[1]).resolve()
    spec = json.loads(spec_path.read_text())
    prefix = spec_path.with_suffix("")
    server_log = Path(str(prefix) + ".server.log")
    result_path = Path(str(prefix) + ".result.json")
    if result_path.exists():
        raise RuntimeError("use a new session spec")
    server = None
    stop = threading.Event()
    monitor = None
    result = {"spec": spec, "clients": []}

    def interrupted(*_):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def telemetry():
        with Path(str(prefix) + ".telemetry.jsonl").open("x") as f:
            while not stop.is_set():
                gpu = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10, check=False)
                record = {"monotonic": time.monotonic(), "gpu0_used_mib": gpu.stdout.strip()}
                # Include native engine descendants, not only the Python HTTP process.
                processes = [server.pid]
                rss = {}
                for pid in processes:
                    try:
                        status = Path(f"/proc/{pid}/status").read_text()
                        rss[str(pid)] = next(line for line in status.splitlines() if line.startswith("VmRSS:"))
                        processes.extend(int(p) for p in Path(f"/proc/{pid}/task/{pid}/children").read_text().split())
                    except (FileNotFoundError, ProcessLookupError, StopIteration):
                        pass
                record["rss"] = rss
                f.write(json.dumps(record) + "\n")
                f.flush()
                stop.wait(1)

    try:
        with server_log.open("x") as stream:
            server = subprocess.Popen(spec["server"], stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            monitor = threading.Thread(target=telemetry, daemon=True)
            monitor.start()
            deadline = time.monotonic() + spec.get("ready_timeout", 600)
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"server exited during startup: {server.returncode}")
                try:
                    with urllib.request.urlopen(spec["health"], timeout=2) as response:
                        if response.status == 200:
                            break
                except (OSError, ValueError):
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError("server readiness timeout")
                time.sleep(.5)
            result["ready"] = True
            for index, client in enumerate(spec["clients"]):
                log = Path(str(prefix) + f".client-{index:02d}.log")
                environment = {**os.environ, **client.get("env", {})}
                for name in client.get("unset_env", []):
                    environment.pop(name, None)
                started = time.monotonic()
                with log.open("x") as output:
                    command = subprocess.run(client["argv"], env=environment, stdout=output, stderr=subprocess.STDOUT,
                                             check=False, timeout=client.get("timeout", 7200))
                result["clients"].append({"argv": client["argv"], "log": str(log), "returncode": command.returncode,
                                          "wall_seconds": time.monotonic() - started})
                if command.returncode:
                    raise RuntimeError(f"client {index} failed: {command.returncode}")
            result["returncode"] = 0
    except BaseException as error:
        result.update(returncode=1, error=str(error) or type(error).__name__)
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        stop.set()
        if monitor is not None:
            monitor.join(timeout=12)
        with result_path.open("x") as output:
            json.dump(result, output, indent=2)
    print(json.dumps(result, indent=2), flush=True)
    return result["returncode"]


if __name__ == "__main__":
    sys.exit(main())
