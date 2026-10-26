#!/usr/bin/env bash
set -euo pipefail
exec >>/mnt/ai/logs/m31-port/port5-prefill-events-export-v1.log 2>&1
/usr/local/cuda-13.1/bin/nsys export --type=sqlite --output /mnt/ai/logs/m31-port/port5-gpu-v3-prefetch-events-v1.sqlite /mnt/ai/logs/m31-port/port5-gpu-v3-prefetch-events-v1.nsys-rep
python3 /mnt/ai/Xolotl-m31/tools/m31/port5_prefill_overlap_events.py --database /mnt/ai/logs/m31-port/port5-gpu-v3-prefetch-events-v1.sqlite --out /mnt/ai/logs/m31-port/port5-prefill-overlap-independent-v1.json
