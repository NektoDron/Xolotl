#!/usr/bin/env bash
set -euo pipefail
exec >/mnt/ai/logs/m31-port/port7-lookup-build-v1.log 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
exec python3 /mnt/ai/Xolotl-m31/tools/m31/port7_lookup_build.py
