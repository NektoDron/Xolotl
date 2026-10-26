#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/ai/Xolotl-m31
exec >>/mnt/ai/logs/m31-port/port5-vocab-final-check-v2.log 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -s -w 86400 9
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export M31_TEST_TOKENIZER="$ROOT/build-m31-oracle/m31_tokenizer"
export M31_TEST_MODEL=/mnt/ai/models/MiniMax-M3.1-Flash-GGUF/UD-IQ3_S/MiniMax-M3.1-Flash-UD-IQ3_S-00001-of-00005.gguf
python3 "$ROOT/tools/m31/test_server.py" RealTokenizerTests -v
