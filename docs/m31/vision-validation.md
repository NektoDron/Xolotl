# Проверки зрения MiniMax M3.1

27.10.2026. Этот документ описывает инструменты проверки. Факт GPU-приёмки,
скорость и конечные ответы записываются в `/mnt/ai/logs/m31-vision/REPORT.md`;
наличие скрипта само по себе не означает, что модель прошла проверку.

## Независимый LLM oracle

`tools/m31/vision_oracle.cpp` отдельно линкуется с зафиксированным llama.cpp
PR #31192 (`6b95a9908d2c7b242817a447708aec38e937e2c0`). Он читает собранный
`m31-vision-batch-v1`, получает текстовые строки из настоящего
`token_embd.weight` модели, подставляет F32 vision rows и вызывает один
`llama_decode` с embedding batch всего промпта. Main decoder positions
линейны: `0..N-1`; в MiniMax decoder нет переноса 2D RoPE vision encoder.
Порядок кадров, текстовые границы и timestamps уже входят в подготовленный
mtmd stream; oracle не придумывает их повторно.

CUDA placement: dense/KV на GPU0, expert weights на CPU,
`op_offload=true`, F16 KV, flash attention, fusion выключен. Ограниченный
выход head сохраняет вычисление последнего FFN для всех строк через
`llama_set_embeddings_layer_inp(n_layer,true)`; только head обрезается до
последних 256 строк. Это даёт проверку logits visual rows и завершающих
текстовых строк, но не всех 3863 строк видео.

Сборка под shared `.build.lock`, `-j24`, в отдельном systemd unit:

```sh
cmake -S tools/m31/vision_oracle_build -B build-vision-oracle \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DM31_LLAMA_SOURCE=/mnt/ai/llama.cpp-m31pr-1026 \
  -DM31_LLAMA_BUILD=/mnt/ai/llama.cpp-m31pr-1026/build
cmake --build build-vision-oracle --target m31_vision_oracle m31_oracle -j24
```

`vision_numeric_suite.py` запускается только внутри собственного aictl
worker с `M31_CARD_SESSION=1`, `CUDA_VISIBLE_DEVICES=0`. Он последовательно
выполняет native/oracle/compare для ref image515, video3863 и текстового
code128. Native использует принятый профиль `native-gpu.json`; ctx4096,
chunk равен фактической длине multimodal stream. Для code128 сохраняются
все 128 logits rows. Это correctness job, его wall time не является TTFT.

```sh
python3 tools/m31/vision_card_launch.py \
  /mnt/ai/logs/m31-vision/numeric-job-v1.json \
  --unit codex-m31-vision-numeric-v1
```

`vision_compare_logits.py` требует PR CUDA GPU-op-offload oracle,
native CUDA candidate, один фактический chunk на весь prompt, одинаковые
token/-1 arrays, positions и head geometry. Затем сравнивает top1 и mean KL
по исходным F32 logits: top1 ≥99%, mean KL ≤1e-3, отдельно записывает maxabs
и побайтовое равенство. Хеши input files включаются в JSON. Этот gate
проверяет LLM при одинаковых готовых embeddings; encoder parity требует
отдельного single/batch или CPU/GPU сравнения encoder output.

Для реального GPU encoder пути suite принимает `--encoder-gpu-summary`
с `vision_gpu_encoder_check.py/summary.json`. Он требует успешную
single/batch parity, берёт `single_ipc` каждого image/video case и соседний
`batch/manifest.json` с backend=cuda. `--skip-text` пропускает повторный
code128, `--ctx` задаёт контекст (по умолчанию4096). Хеши всех пяти входных
файлов снимаются до native и сверяются после oracle/compare; изменение
входа отвергает приёмку.

## HTTP и содержательные проверки

```sh
python3 tools/m31/vision_validation.py prepare \
  /mnt/ai/logs/m31-vision/validation-fixtures-v1
python3 tools/m31/vision_validation.py http \
  /mnt/ai/logs/m31-vision/validation-fixtures-v1/cases.json \
  --endpoint http://127.0.0.1:8181 --model m31-vision \
  --output /mnt/ai/logs/m31-vision/http-v1.json
```

prepare создаёт детерминированные картинки без инференса: надпись
`M31-7429`, зелёный квадрат и три оранжевых круга. Шесть HTTP cases:
ref PNG (красный круг, цифра 1), ref video (красный круг → зелёный квадрат
→ синий треугольник), OCR, две картинки в одном сообщении, картинка в истории
диалога перед assistant ack и последующим вопросом, подсчёт трёх кругов. Для видео используется file URL
из разрешённого каталога сервера; остальные картинки идут data/base64.

http сохраняет весь SSE stream, видимый ответ, reasoning, TTFT первого
текстового delta и отдельно первого видимого content delta. Отсчёт до POST
включает HTTP, media load, encoder и LLM prefill. Thinking выключен;
температура 0. Ref TTFT image5.238/video36.171с записывается рядом, вместе
с отношением ref/current. Условие «порядок цветов» проверяется отдельно от
их присутствия; факты проверяются в видимом ответе. Ответы сохраняются для
ручной проверки формы, положения и временных границ. Текстовый tools gate
— существующий `server_tool_smoke.py` либо `serve/smoke_native_client.py`.

CPU-проверка самого validation client:

```sh
python3 tools/m31/test_vision_validation.py
```

Она проверяет отклонение неверного порядка видео, границы цифры и реальный
локальный SSE транспорт с разделёнными reasoning/content deltas. GPU и
model process в ней не создаются.
