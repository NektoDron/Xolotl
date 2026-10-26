# M31 experimental server

This is an opt-in adapter for the independent `m31_engine`. It does not change the
other Xolotl servers. It is part of M2 work, not an accepted speed/quality result.

`m31_serve` owns a single native `Engine` and consumes JSON lines on stdin. It
emits `ready`, `prefill`, `token`, `done` (or `error`) records on stdout; diagnostics
remain on stderr. A request supplies token IDs and sampling settings. With
`interactive:true`, each emitted token requires `{"continue":true|false}` before
the next forward pass. This allows HTTP disconnects and string stops to cancel
without killing/reloading the model. Each new request resets the KV state; prefix
reuse, concurrency and speculative decoding are not connected.

`m31_tokenizer` is a **separate process** linked against the audited llama.cpp PR
#31192 libraries. It loads vocabulary only, sets `CUDA_VISIBLE_DEVICES` empty and
never runs model inference. It supplies NFC tokenization, token bytes, EOS/model
suppression IDs, the GGUF chat template, `common_chat_templates_apply` (including
the reference enable_thinking handling), and the same MiniMax PEG reasoning and
nested XML tool parser as the reference. Xolotl and llama ggml are not linked in
one executable. No handwritten substitute template is used.

`serve_m31.py` uses only Python's standard library and binds loopback. Routes:

- `GET /health`, `/healthz`, `/v1/models`;
- `POST /tokenize`, `/detokenize` for `~/om_bench.py`;
- `POST /v1/chat/completions` and `/v1/completions`, with nonstream or SSE;
- `POST /completion` for basic raw-prompt compatibility.

Chat responses split `reasoning_content`, `content`, and `tool_calls`; tool calls
are emitted together at completion in SSE. UTF-8 pieces are decoded incrementally,
partial stops are withheld, parser errors are returned explicitly, and service
markers are interpreted by the reference parser. Tool argument strings are
JSON. No tool is executed.

Sampling supports greedy `temperature:0`, temperature/top-k/top-p/min-p,
`seed`, `ignore_eos`, `max_tokens`, string `stop`, and model suppression tokens.
The default chain is top-k40, top-p.95, min-p.05, then temperature1.0. The C++ RNG is
not bit-identical to llama.cpp. Nonzero repeat/frequency/presence penalties,
logit bias, response schemas, grammar constraints and forced tool choice are
rejected. Automatic tools use the reference prompt/parser and its real lazy
GBNF sampler in the isolated vocab helper. The worker proposes a sampled token;
the helper checks it and accepts it only after validation. If rejected, the helper
filters the whole vocabulary and returns allowed IDs; the worker resamples the
original logits with the same sampling chain on that support. It does not use a
greedy fallback. Plain responses under auto remain permitted. The PR reasoning
budget sampler suppresses grammar within thinking blocks and replays the end
sequence when thinking finishes. Each chat request rebuilds the grammar state.
`tool_choice:auto|none` is supported. Images/video are explicitly rejected.

Responses include `usage` and llama-compatible `timings` with prompt/prediction
counts, milliseconds and rates. Timers come from the engine and include per-token
controller overhead; HTTP/model loading is outside them. The first sample belongs to prefill; generation timing measures the remaining
`predicted_n - 1` steps and excludes final token transport, like the reference. Prefill
computes only the last needed logits row instead of dumping every vocabulary row.

Build integration (coordinator):

1. Add `m31_serve` linked to `m31_engine`, include the fetched vendor's
   `vendor/nlohmann`, define `M31_HAVE_CUDA` when CUDA is linked.
2. Add `m31_tokenizer` to the separate oracle project. Include audited PR `common`,
   `vendor`, `vendor/nlohmann`, `include`, `ggml/include`. Link, in order,
   `libllama-common.a`, `libllama-common-base.a`, then the same llama/ggml/CUDA
   dependencies as `m31_oracle`, and `vendor/hash/libvendor-hash.a`.
3. `python3 tools/m31/test_server.py` checks protocol cancellation, engine errors,
   UTF-8/string-stop boundaries, invalid sampling parameters and vision refusal.
   These tests do not establish real token/template/inference parity.

Example launcher arguments, **only inside an aictl-owned card session under the
shared card lock and cleanup trap**:

```sh
python3 tools/m31/serve_m31.py \
  --engine /mnt/ai/Xolotl-m31/build-m31-cuda/m31_serve \
  --tokenizer /mnt/ai/Xolotl-m31/build-m31-oracle/m31_tokenizer \
  --model /mnt/ai/models/MiniMax-M3.1-Flash-GGUF/UD-IQ3_S/MiniMax-M3.1-Flash-UD-IQ3_S-00001-of-00005.gguf \
  --backend cuda --ctx 4096 --chunk 128 --threads 64 --port 8180 \
  --arena-gib 170 --gpu-experts-gib 2 --reuse-allocations --persistent-cpu-pool
```

This command does not authorize direct GPU use. The adapter defaults to port8180; the batch worker health service owns8179.
The coordinator owns aictl,
flock, preset backup/rollback and systemd units. Increasing context/batch/cache
requires the corresponding VRAM gate. The accepted M1 setting still requires
`GGML_CUDA_DISABLE_FUSION=1`; backend optimizations need their own logits gate.

Validation on aiserver, 2026-10-26:

- Coordinator built CPU/CUDA `m31_serve` and separate `m31_tokenizer` successfully.
- `codex-m31-port3-tokenizer-tests-v2`, User=nekto, hidden CUDA: **11/11** tests,
  0.427 seconds; `/mnt/ai/logs/m31-port/tokenizer-tests-v2.log`.
  Includes real GGUF NFC equality/Unicode roundtrip, control-token parsing modes,
  MiniMax think/content and disabled-reasoning parsing, streaming prefixes, nested
  XML object/array tools and the next tool-response chat template.
- The first real helper gate exposed an omitted PEG arena in the convenience
  parser-params constructor; the wrapper now explicitly installs it and uses
  token-aware input/prefix, matching `common_chat_session`. The failed v1 log is
  retained; v2 passes after rebuild.
- These checks do not establish real HTTP generation quality or server speed.
  The coordinator performs GPU correctness and performance gates separately.


Port4 CPU checks: private `build-server-eval/m31_tokenizer` builds against audited
PR libraries; C++ server syntax checks pass. `server-grammar-tests.log` records
**15/15** protocol and real vocab-only tests in 0.603s: lazy triggering, text-only
auto replies, trigger inside reasoning, Unicode nested object/array arguments,
two calls, rejection of an unknown function name with allowed token IDs, reset,
and cancellation during a proposal. A real inference request using this new
handshake still needs an HTTP gate; tools23/GSM8K100 are not yet measured.
`--cpu-decode-cache` is forwarded to the opt-in engine flag; its correctness and
speed are checked separately by the CPU agent.

Live native tool gate, port4: `server-tools-http-v1/http-tool-smoke.json`,
**38/38** checks, systemd/card session rc0. Auto replies391 without a call;
nonstream/SSE and the request after a thinking-block disconnect return
get_weather(city=Казань,unit=celsius). Native sampling reports11grammar checks
for the plain response and49per tool response. These examples required no
rejections; full tools23 quality is still pending. This gate used the separate
build-cpu-decode/m31_serve with both CPU/dense decode caches enabled.

Native timings now expose `sampling_ms`/`sampling_calls`, `grammar_wait_ms` and
`interactive_wait_ms`/`interactive_waits`. They cover the complete request,
including the first sample and final token ACK; existing prompt/predicted timing
boundaries stay unchanged. With adapter `--profile`, one `M31 request timings`
JSON record is written to stderr after native completion, so om_bench's compact
TSV can be paired with the full breakdown. The Application profile flag defaults
false, including in protocol tests.
Greedy temperature0 uses a single finite-checked scan without a vocabulary-sized
pair vector. Its tie, exclusion, grammar-support and error semantics match the
previous max_element path;1416 checks include128real code128logit rows.

The adapter now parses nonstream responses only at completion. Streaming keeps
per-token stop decisions and sends the native ACK before PEG parsing, overlapping
parse work with the next forward pass. A PEG error drains/cancels at the next
boundary; no pending native record remains for the following request. Protocol
and real vocab comparisons pass21/21 checks, including final stream/nonstream
message equivalence, stops, parse failures and disconnect cleanup. Live HTTP
verification of this optimization follows the in-progress tools23 run.

Top-level `reasoning_effort` matches the audited PR server: a nonempty string
wins over the corresponding template kwarg, and `none` disables thinking.
Reference speed uses `reasoning_effort:low`; reference tools quality leaves it
unspecified. These are different settings from `enable_thinking:false`.

Port4 original tools23 fixtures completed **20/23** (reference19/23), with no
length truncations or marker leaks. Full raw choices and sampling metadata are
in `/mnt/ai/logs/m31-port/server-tools23-v3/`. This is one stochastic run, not a
claim of a significant quality improvement. The evaluation never executed a tool.
It used the loaded server from before the ACK/nonstream optimization; that new
Python path has22/22 CPU/vocab checks, including every truncated token prefix of
a reasoning+XML-tool response, and is undergoing a separate live HTTP gate.

The new ACK/nonstream adapter also passed its live native HTTP gate:
`server-ack-http-v1/http-smoke.json` **109/109** and
`server-ack-http-v1/http-tool-smoke.json` **38/38**, unitrc0 and full card cleanup.
A strict4K speed comparison with reference reasoning_effort=low is separate from
these correctness checks and remains the coordinator's next measurement.
