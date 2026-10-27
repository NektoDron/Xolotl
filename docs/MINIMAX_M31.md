# MiniMax-M3.1-Flash

This experimental port runs MiniMax-M3.1-Flash GGUF weights with one GPU
and system RAM. The ordinary CUDA path uses native ggml GPU arithmetic.
The implementation is in `tools/m31`; the common Strata server integration
is still planned.

The model has 60 layers: three dense layers and 57 routed MoE layers,
128 experts with four selected per token, and a shared expert. All layers
use the MSA indexer. Hidden width is 6144. The port keeps the original
IQ3_XXS, IQ4_XS and NVFP4 expert weights and their external scales.

Dense weights and F16 main/index KV caches live on the GPU. Expert weights
are repacked into a RAM arena, with an individual expert cache on the GPU.
Layer prefetch overlaps registered-host-memory DMA with computation.
Native GPU prefill keeps the routed FFN intermediates on the GPU and can
keep the residual stream there between layers. Small-batch generation
retains the CPU expert path and the GPU dense/decode chain.

The optimized settings are recorded in the portable
[native GPU profile](../tools/m31/profiles/README.md). The profile planner
prints the complete environment and engine arguments for review; it does
not start a process. The speed figures below require this profile, not
only the build commands.

## Build

Use a C++20 toolchain and CMake (the CUDA sources use C++17). The root
build fetches its pinned ggml; the optional oracle is built separately.
The tested CUDA machine used
GCC 14, CUDA 13.1 and an RTX 5090 (`sm_120a`). CPU and CUDA are the tested
backends for this port; HIP and SYCL have not been validated.

CPU tools:

```sh
cmake -S . -B build-m31-cpu -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_M31=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_GGML_CUDA=OFF \
  -DSTRATA_DENSE=OFF -DSTRATA_BUILD_TESTS=OFF
cmake --build build-m31-cpu --target m31_inspect m31_generate m31_serve -j 24
```

Native CUDA tools:

```sh
cmake -S . -B build-m31-cuda -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_M31=ON -DSTRATA_ENABLE_CUDA=ON -DSTRATA_GGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120a \
  -DSTRATA_M31_REGISTERED_ARENA_CUDA=ON \
  -DSTRATA_M31_CPU_QUANTS_CUDA=OFF -DSTRATA_M31_FAST_FFN_CUDA=OFF \
  -DSTRATA_DENSE=OFF -DSTRATA_BUILD_TESTS=OFF
cmake --build build-m31-cuda --target m31_inspect m31_generate m31_serve -j 24
```

To also compile the optional CPU-compatible verification and custom GPU
FFN implementations, set both `STRATA_M31_CPU_QUANTS_CUDA=ON` and
`STRATA_M31_FAST_FFN_CUDA=ON`. They are independent build options; native
GPU execution does not require them. A host with a newer glibc may need
its CUDA/toolchain compatibility configuration in addition to these
portable CMake options.

`m31_generate` accepts token IDs and writes logits for comparison.
`m31_serve` is the engine subprocess used by `tools/m31/serve_m31.py`.
The current HTTP adapter uses a separately built llama.cpp helper for
tokenization, chat templates and structured parsing. This dependency is
still present; the model layers run in the port's engine.

On the shared aiserver, acquire GPU 0 through `aictl` and the
`tools/m31/card_launch.py` runner. The CUDA tools require that managed
session. Do not run them beside another GPU workload or benchmark during
a build. See [the tool README](../tools/m31/README.md) for the runner and
request examples.

## Numerical contract

The fast-path oracle is GPU-offload llama.cpp PR
[#31192](https://github.com/ggml-org/llama.cpp/pull/31192), pinned at
`6b95a9908d2c7b242817a447708aec38e937e2c0`. The oracle receipt records
the actual binary and placement. Acceptance requires top-1 agreement at least
99%, mean KL at most `1e-3` and maximum per-position KL at most `0.05` on
gate20, plus GSM8K100 at least 95/100 and tools23 at least 19/23.
Bitwise CPU agreement is an optional regression check, not the fast-path
acceptance criterion. Set `M31_GPU_EXACT=1` in a build containing the
verification kernels to select that separate path.

Keep oracle placement, KV types, chunk schedule, MMQ precision and fusion
settings in the receipt. A short-context gate does not establish long
context accuracy. The active tested profile disables CUDA fusion with
`GGML_CUDA_DISABLE_FUSION=1`; unrestricted fusion is not accepted.

## Measured speed and numerical checks

**Canonical V8 passed the speed, numerical and quality checks on 2026-10-27.**
One RTX 5090 on the aiserver,
EPYC host, context capacity 40000, F16 main/index KV, 170 GiB RAM-arena
budget, 7 GiB GPU expert-cache budget, chunk ceiling 32768 and 80 CPU
threads. Values are medians of two runs:

| Requested depth | Prefill, tokens/s | Generation, tokens/s |
|---|---:|---:|
| 4K | 1390.85 | 27.445 |
| 16K | 1535.00 | 27.015 |
| 32K | 1246.65 | 25.305 |

The two 32K generation results were 24.61 and 26.00 tokens/s. Its median
meets the 25 tokens/s floor; one individual run was below it. Two rounds
do not establish statistical significance. The older llama.cpp baseline
reported 433 tokens/s prefill at 32K with Q8_0 KV, versus this port's F16
KV; this is a configuration comparison, not identical KV settings.

This used native stock GPU FFN (`M31_GPU_FAST_FFN=0`, `M31_GPU_EXACT=0`),
registered arena DMA, upload depth 8 with one copy stream, GPU prefill
chain, adaptive chunks and fusion disabled. The MSA indexer used the
baseline untiled/F32 causal-mask configuration. Reported peak GPU memory
was 30483 MiB. V8 includes optional debug kernels in its binary; they
were disabled at runtime.

The full local receipt is
`/mnt/ai/logs/m31-port/port6r4-final-v8-acceptance-v1/speed/`:
`manifest.json` records the environment and frozen binary, `report.json`
and `speed.tsv` record the measurements. The frozen engine SHA256 is
`39a86da6bd1a601adbfe02ee8556278be5c7c6f87a2ca3b0cc3ea5d47a778472`.
The same canonical V8 engine build passed all 20 gate prompts and 28,723
logit rows against the GPU oracle: top-1 agreement 100%, KL and maximum
absolute error 0. V8 also matched 256 selected positions each after a
direct 10240-token batch and at 16K and 32K, including 32 teacher-forced
decode positions per case. These are selected-position checks, not all
long-context logits. The long oracle used case-sized physical KV capacity
to avoid its worst-case reservation OOM; active keys, actual chunks and
RoPE parameters were checked against native context 40000. See the
[receipt audit](m31/port6-v8-receipt-audit.md).

GSM8K100 scored **97/100** and tools23 **19/23** (17/20 initial tool
cases and 2/3 follow-ups). The original evaluation client and dataset
were preserved: temperature 1.0, top-p 0.95, at most 6000 output tokens,
and the model's default adaptive thinking. Tools were scored without
executing their calls. The independent audit confirmed all 123 unique
case IDs, boolean scores, unchanged sampling and the same engine,
adapter and environment as the speed run. It reported no length stops
or marker leaks. This one stochastic quality run does not establish a
significant improvement over the older baseline's 96/100 and 19/23.

Quality receipts are under the same acceptance directory's `quality/`;
the independent result is
`/mnt/ai/logs/m31-port/port6-quality-independent-final-v1.json`.

## Current limits

The exercised capacity is 40000 tokens with F16 KV. Larger contexts,
including 128K, have not been accepted. This port does not yet use the
common `Ds4MoeTier`/disk-parking path. It serves one sequence at a time,
resets KV between requests and has no engine prefix cache.

Vision preprocessing can produce audited mixed token/embedding batches,
but the engine does not yet consume them. Engine vision, video, common
server integration and prefix reuse remain separate work. See the
[server integration plan](m31/port6-server-integration-plan.md) and
[native GPU audit](m31/port6-native-ggml-audit.md).
