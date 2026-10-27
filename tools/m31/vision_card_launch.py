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
    for command, flag in (("up", "--no-switch"), ("down", "--if-owned")):
        help_text = subprocess.run(["aictl", command, "--help"], check=True,
                                   capture_output=True, text=True).stdout
        if flag not in help_text:
            parser.error("aictl lacks atomic ownership guard " + flag +
                         "; install card_aictl_guard.patch before launching jobs")
    job = args.job.resolve(strict=True)
    invalidated = job.parent / 'INVALIDATED.json'
    if invalidated.exists():
        parser.error(f'каталог помечен {invalidated}; этот кандидат отклонён, подготовьте новый job')
    backup = Path('/etc/aictl') / ('config.json.bak-m31-vision-' + job.stem)
    if backup.exists() or backup.is_symlink():
        parser.error(f'бэкап {backup} уже существует: нужен уникальный basename job JSON, одного нового имени unit недостаточно')
    spec = json.loads(job.read_text())
    if not isinstance(spec.get("measurement", True), bool):
        parser.error("measurement must be boolean")
    if job.with_suffix(".result.json").exists():
        parser.error("job already has a result; prepare a fresh job")
    source = Path(__file__).with_name("vision_card_session.sh")
    helpers = {}
    for name in ("vision_card_worker.py", "vision_card_preset.py"):
        helper_source = source.with_name(name)
        helper_target = job.with_suffix("." + name.removeprefix("vision_card_"))
        helper_data = helper_source.read_bytes()
        with helper_target.open("xb") as stream:
            stream.write(helper_data)
        helper_target.chmod(0o444)
        helpers[name] = {"snapshot": str(helper_target),
                         "sha256": hashlib.sha256(helper_data).hexdigest()}
    frozen = job.with_suffix(".runner.sh")
    data = source.read_bytes()
    with frozen.open("xb") as stream:
        stream.write(data)
    frozen.chmod(0o444)
    with job.with_suffix(".launcher.json").open("x") as stream:
        json.dump({"source": str(source), "snapshot": str(frozen), "job": str(job),
                   "sha256": hashlib.sha256(data).hexdigest(),
                   "helpers": helpers,
                   "measurement": spec.get("measurement", True)}, stream, indent=2)
        stream.write("\n")
    subprocess.run(["sudo", "systemd-run", "--unit=" + args.unit, "-p", "User=nekto",
                    "-p", "KillMode=mixed", "-p", "TimeoutStopSec=600",
                    "/bin/bash", str(frozen), str(job)], check=True)


if __name__ == "__main__":
    main()
