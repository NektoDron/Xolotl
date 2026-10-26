#!/usr/bin/env bash
# Invoke through sudo systemd-run -p User=nekto. Worker launches the actual models from aictl.
set -euo pipefail
M31_ROOT=/mnt/ai/Xolotl-m31
M31_JOB=$(realpath "$1")
M31_STEM=$(basename "$M31_JOB" .json)
M31_PRESET=$(python3 - "$M31_JOB" <<'PY'
import json, sys
name = json.load(open(sys.argv[1])).get('preset', 'm31-xolotl-check')
if name not in ('m31-xolotl-check', 'm31-xolotl'):
    raise ValueError('unexpected temporary preset')
print(name)
PY
)
exec >>"/mnt/ai/logs/m31-port/${M31_STEM}.session.log" 2>&1
exec 9>/mnt/ai/logs/.xolotl-card.lock
flock -w 86400 9
exec 8>/mnt/ai/logs/m31-port/.build.lock
flock -w 86400 8
M31_BACKUP="/etc/aictl/config.json.bak-m31-port-${M31_STEM}"
M31_STOP=0
M31_INSTALLED=0
M31_UP_STARTED=0
trap 'M31_STOP=1' INT TERM
cleanup() {
    # up's wait has completed before reaching EXIT, including a received TERM/INT.
    if [ "$M31_UP_STARTED" = 1 ]; then sudo aictl down; fi
    if [ "$M31_INSTALLED" = 1 ]; then
        sudo python3 "$M31_ROOT/tools/m31/card_preset.py" rollback "$M31_BACKUP" "$M31_PRESET"
    fi
}
trap cleanup EXIT
python3 - <<'PY'
import json
from pathlib import Path
state = json.loads(Path('/var/lib/aictl/state.json').read_text())
if state.get('active'):
    raise RuntimeError('another model is active; refusing to touch it: ' + state['active'])
PY
sudo python3 "$M31_ROOT/tools/m31/card_preset.py" put "$M31_BACKUP" "$M31_JOB"
M31_INSTALLED=1
M31_UP_STARTED=1
sudo aictl up "$M31_PRESET" &
M31_UP_PID=$!
M31_UP_RC=0
# Interrupted wait is retried until the actual up process has returned.
while true; do
    wait "$M31_UP_PID" && M31_UP_RC=0 || M31_UP_RC=$?
    if ! kill -0 "$M31_UP_PID" 2>/dev/null; then break; fi
done
if [ "$M31_UP_RC" != 0 ]; then exit "$M31_UP_RC"; fi
M31_RESULT="${M31_JOB%.json}.result.json"
while [ ! -f "$M31_RESULT" ]; do
    if [ "$M31_STOP" = 1 ]; then exit 143; fi
    if ! systemctl is-active -q codex-m31-port-card-worker; then exit 1; fi
    sleep 5 & wait $! || true
done
python3 - "$M31_RESULT" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))
print(json.dumps(result, ensure_ascii=False, indent=2))
sys.exit(0 if result['returncode'] == 0 else 1)
PY
