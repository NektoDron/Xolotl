#!/usr/bin/env bash
# Own helper build and link-only engine frontend syntax check; no vendor rebuild.
set -euo pipefail
root=/mnt/ai/Xolotl-m31
exec >>/mnt/ai/logs/m31-port/server-build-eval.log 2>&1
exec 9>/mnt/ai/logs/m31-port/.build.lock
flock -w 86400 9
cmake -S "$root/tools/m31/oracle" -B "$root/build-server-eval" \
 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 -DCUDAToolkit_ROOT=/usr/local/cuda \
 -DM31_LLAMA_SOURCE=/mnt/ai/llama.cpp-m31pr-1026 \
 -DM31_LLAMA_BUILD=/mnt/ai/llama.cpp-m31pr-1026/build
cmake --build "$root/build-server-eval" --target m31_tokenizer -j24
/usr/bin/g++-14 -std=c++20 -fsyntax-only \
 -I"$root/tools/m31" -I"$root/include" \
 -I"$root/build-m31-cpu/_deps/strata_llamacpp-src/ggml/include" \
 -I"$root/build-m31-cpu/_deps/strata_llamacpp-src/vendor/nlohmann" \
 "$root/tools/m31/m31_serve.cpp"
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export M31_TEST_TOKENIZER="$root/build-server-eval/m31_tokenizer"
export M31_TEST_MODEL=/mnt/ai/models/MiniMax-M3.1-Flash-GGUF/UD-IQ3_S/MiniMax-M3.1-Flash-UD-IQ3_S-00001-of-00005.gguf
python3 "$root/tools/m31/test_server.py"
