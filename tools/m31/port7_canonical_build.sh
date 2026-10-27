#!/usr/bin/env bash
# Coordinator source-freeze ACK required. Invoke with systemd-run User=nekto.
set -euo pipefail
ROOT=/mnt/ai/Xolotl-m31
LOG=/mnt/ai/logs/m31-port/port7-canonical-${INVOCATION_ID:?systemd invocation required}.log
exec > >(tee -a "$LOG") 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
exec /usr/bin/python3 -u "$ROOT/tools/m31/port7_canonical.py" build "$@"
