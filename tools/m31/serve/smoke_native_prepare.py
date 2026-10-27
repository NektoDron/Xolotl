#!/usr/bin/env python3
"""Freeze an isolated common-shim GPU smoke job; never queue or launch it."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    args = parser.parse_args()
    destination = args.directory.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    tree = destination / "tree"
    tree.mkdir()
    source_files = list((ROOT / "serve").glob("*.py"))
    source_files += [ROOT / "tools/m31" / name for name in (
        "serve_m31.py", "profile_plan.py", "profiles/native-gpu.json", "server_session_job.py")]
    source_files += list((ROOT / "tools/m31/serve").glob("*.py"))
    frozen = []
    for source in source_files:
        target = tree / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        target.chmod(0o444)
        frozen.append({"source": str(source), **fingerprint(target)})
    acceptance = json.loads((args.acceptance / "speed/manifest.json").read_text())
    engine_source = Path(acceptance["engine"]["path"])
    tokenizer_source = Path(acceptance["tokenizer"]["path"])
    for source, target_name, expected in (
            (engine_source, "m31_serve-frozen", acceptance["engine"]["sha256"]),
            (tokenizer_source, "m31_tokenizer-frozen", acceptance["tokenizer"]["sha256"])):
        target = destination / target_name
        subprocess.run(["cp", "--reflink=auto", "--preserve=mode,timestamps", str(source), str(target)], check=True)
        target.chmod(0o555)
        actual = fingerprint(target)
        if actual["sha256"] != expected:
            raise RuntimeError(f"frozen executable hash mismatch: {source}")
        frozen.append({"source": str(source), **actual})
    config = {"runtime_profile": str(tree / "tools/m31/profiles/native-gpu.json"),
              "engine": str(destination / "m31_serve-frozen"),
              "model": "/mnt/ai/models/MiniMax-M3.1-Flash-GGUF/UD-IQ3_S/MiniMax-M3.1-Flash-UD-IQ3_S-00001-of-00005.gguf",
              "tokenizer_argv": [str(destination / "m31_tokenizer-frozen"),
                                  "/mnt/ai/models/MiniMax-M3.1-Flash-GGUF/UD-IQ3_S/MiniMax-M3.1-Flash-UD-IQ3_S-00001-of-00005.gguf"],
              "model_name": destination.name}
    config_path = destination / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    config_path.chmod(0o444)
    scripts = tree / "tools/m31/serve"
    audit = destination / "engine-audit.jsonl"
    session = {"server": ["/usr/bin/env", "M31_COMMON_SMOKE_TREE=" + str(tree),
                           "M31_COMMON_SMOKE_AUDIT=" + str(audit), "/usr/bin/python3",
                           str(scripts / "smoke_native_server.py"), "--config", str(config_path), "--port", "8181"],
               "health": "http://127.0.0.1:8181/health", "ready_timeout": 900,
               "clients": [{"name": "common_native_smoke", "argv": ["/usr/bin/python3",
                   str(scripts / "smoke_native_client.py"), "--tree", str(tree), "--config", str(config_path),
                   "--audit", str(audit), "--output", str(destination / "smoke-result.json"), "--port", "8181"],
                   "timeout": 600}]}
    session_path = destination / "common-native-smoke-http-session.json"
    session_path.write_text(json.dumps(session, indent=2) + "\n")
    session_path.chmod(0o444)
    job = {"preset": "m31-xolotl", "measurement": False, "command_timeout_seconds": 1800,
           "commands": [["/usr/bin/python3", str(tree / "tools/m31/server_session_job.py"), str(session_path)]]}
    job_path = destination / (destination.name + "-card-job.json")
    job_path.write_text(json.dumps(job, indent=2) + "\n")
    job_path.chmod(0o444)
    # Both --help import/CLI checking and full main() with all constructors mocked
    # run on CPU. They cannot allocate a GPU or launch m31_serve.
    help_check = subprocess.run([sys.executable, str(scripts / "serve_m31.py"), "--help"],
                                text=True, capture_output=True, check=True)
    (destination / "cli-help.txt").write_text(help_check.stdout)
    subprocess.run([sys.executable, str(scripts / "smoke_native_preflight.py"), "--tree", str(tree),
                    "--config", str(config_path), "--output", str(destination / "cli-preflight.json")], check=True)
    preflight = json.loads((destination / "cli-preflight.json").read_text())
    assert preflight["environment"] == acceptance["environment"]
    assert preflight["environment_count"] == 48
    manifest = {"format": "m31-common-native-smoke-plan-v1", "prepared_only": True,
                "source_files": frozen, "acceptance_manifest": fingerprint(args.acceptance / "speed/manifest.json"),
                "config": fingerprint(config_path), "job": fingerprint(job_path),
                "cli_preflight": fingerprint(destination / "cli-preflight.json"),
                "engine_environment_equal_to_acceptance": True, "environment_count": 48,
                "limits": ["Not launched by prepare; queue after quality and CUPTI with coordinator approval.",
                           "Native common-shim HTTP smoke only; no GSM/tools quality or performance claim.",
                           "Accepted standalone adapter/core defaults remain unchanged."],
                "launch_argv": ["/usr/bin/python3", str(ROOT / "tools/m31/card_launch.py"),
                                str(job_path), "--unit", "codex-m31-" + destination.name]}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"directory": str(destination), "job": str(job_path), "prepared_only": True}, indent=2))


if __name__ == "__main__":
    main()
