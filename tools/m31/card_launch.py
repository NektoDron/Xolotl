#!/usr/bin/env python3
"""Submit a card job through a per-job, read-only snapshot of the shell launcher."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("--unit", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,180}", args.unit):
        parser.error("invalid systemd unit name")
    job = args.job.resolve(strict=True)
    spec = json.loads(job.read_text())
    if not isinstance(spec.get("measurement", True), bool):
        parser.error("measurement must be boolean")
    if job.with_suffix(".result.json").exists():
        parser.error("job already has a result; prepare a fresh job")
    source = Path(__file__).with_name("card_session.sh")
    frozen = job.with_suffix(".runner.sh")
    data = source.read_bytes()
    with frozen.open("xb") as stream:
        stream.write(data)
    frozen.chmod(0o444)
    with job.with_suffix(".launcher.json").open("x") as stream:
        json.dump({"source": str(source), "snapshot": str(frozen), "job": str(job),
                   "sha256": hashlib.sha256(data).hexdigest(),
                   "measurement": spec.get("measurement", True)}, stream, indent=2)
        stream.write("\n")
    subprocess.run(["sudo", "systemd-run", "--unit=" + args.unit, "-p", "User=nekto",
                    "/bin/bash", str(frozen), str(job)], check=True)


if __name__ == "__main__":
    main()
