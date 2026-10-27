# MiniMax-M3.1: TTFT разобран, оптимизации opt-in

27.10.2026. Функциональный commit342bcc74, отчёт
`/mnt/ai/logs/m31-vision/REPORT-ttft.final.md`.
Текст515 TTFT3.072с, PNGwarm3.139с: примерно2.97с занимает самprefill,
143GB обязательных экспертных весов идутпоPCIe. Общийunseen<1сне достигнут.
Historyprefetch убрал20.33GB лишнегоDMA: PNGwarm2.727с/newquestion2.734с,
безKVreuse. Prewarm перенёс1.47сworkerinitвstartup:firstPNG4.765→3.067с.
ExactwholepromptKVcache отдельно: PNG0.096–0.103с/video0.270–0.275с;
новыйpromptmiss,cache_promptfalseбypass, памятьhostLRU<=1GiB.
Gate9012allrowlogitsпобитно+3GPUcheckpointcontinuationPASS, HTTPvision6/tools3PASS.
CPU/CUDApost-mergebuildPASS, ELFunchanged; своиunitsзакончены,presetснят,
GPU0/GPU1майнят,duty8084healthy,prod/port7неизменены,localcommitsбезpush.
Профили `tools/m31/profiles/vision-ttft*.json`, encoderexample prewarmtrue.

---

# MiniMax-M3.1: vision принят

27.10.2026. Ветка `aiserver/m31-vision`, функциональный код `23ccb294`.
Итог `/mnt/ai/logs/m31-vision/REPORT.final.md`, приёмка `acceptance.final.json`.
Картинки/видео идут через mtmd CPU/CUDA и native `forward_mixed`: весь stream
одним v8 prefill, main positions linear (не M-RoPE). PNG515/video3863/code128:
все4506 logits rows побитно GPU oracle. HTTP vision6/6, tools3/3 PASS.
Cold TTFT PNG4.36048с/video4.04333с против ref5.238/36.171; warm3.06169/3.20453с.
RTX5090 GPU0: mainctx40000/cache7 + encoder peak28382MiB, auto→CUDA.
CPU fallback6×16, cache backend+GGMLenv; CPU/GPU encoder outputs различаются,
поэтому паритет заявлен отдельно по backend. Stock PR CUDA PNG побитен.

Опционально: standalone `tools/m31/serve_m31.py --vision-config FILE`, common
bridge config vision; схема `docs/m31/vision-http.md`. Frozen binaries/receipts
`/mnt/ai/logs/m31-vision/final-build/`, все исходники/коммиты локальные, push нет.
Merge актуального `aiserver/m31-port` faeb1332 Already up to date; CPU/CUDA
пересборка и107unittest (10optional skips)+NumPy/tiny прошли.
Свой preset m31-vision-test снят, свои GPU/build units закончены; GPU1/prod не
менялись, duty8084 healthy. GPU0 затем забрала соседняя сессия, её не останавливать.

---

# M4: CPU encoder adapter и общий пакет кадров

Port4, 26.10.2026. Это подготовка M4; native vision inference ещё не подключён.

`tools/m31/vision_adapter.cpp` — отдельный процесс на неизменённых библиотеках
llama.cpp PR `6b95a9908d2c7b242817a447708aec38e937e2c0`. Собран в
`build-vision-prepare/m31_vision_adapter`, link-only CMake в `vision_build/`.
Использует только vocabulary основной модели, CPU mmproj (`use_gpu=false`,
`warmup=false`, `CUDA_VISIBLE_DEVICES=''`), настоящий chat template и mtmd
image/video preprocessing. `llama_context` не создаётся, LLM не исполняется.
Две библиотеки ggml разных проектов не смешиваются в одном процессе.

IPC содержит исходный prompt, текстовые token IDs, абсолютные позиции,
порядок изображений/кадров, serialized mtmd metadata и optional F32 encoder rows.
`vision_batch.py assemble` валидирует непрерывность позиций и сохраняет единый
mixed text/vision row stream: `text_ids.i32`, `visual_rows.i32`, `positions.i32`,
`visual.f32`. Весь набор кадров готов для одного LLM префилла; фактического
выполнения этого батча в Xolotl пока нет. `text_ids=-1` обозначает visual row,
это не допустимый вход обычного token-only `Engine.forward()`.

CPU preprocessing проверен на реальных файлах ref (`0.png`, `shapes-12s.mp4`):

| Вход | Полный prompt | Visual rows | Кадры | Resize | Patch / merged grid |
|---|---:|---:|---:|---|---|
| PNG |515|299|1|644×364|46×26 /23×13|
| MP4,1fps |3863|3588|12|644×364 каждый|46×26 /23×13|

Количество токенов и **все абсолютные начала visual spans** совпали с saved
mtmd server logs референса. PNG span начинается с210. Видео начинается с
223,532,833,1134,1435,1736,2045,2346,2647,2948,3249,3558.
Позиции учитывают текстовые границы изображений и timestamps `[0m00.00s]`,
`[0m05.00s]`, `[0m10.00s]`. Для равных prompt counts существенно передать
`chat_template_kwargs.reasoning_effort="low"`, как в ref request.
Без этого системный template был короче на9 токенов; неудачная v2 сохранена.

Артефакты: `/mnt/ai/logs/m31-port/vision-preprocess-v3/`, unit
`codex-m31-port4-vision-preprocess-v3` rc0. Сборки v1/v2 rc0, лог
`vision-build.log`. В первом preprocessing-v1 обнаружен vocab-only контракт:
`llama_model_n_embd_inp()` возвращает0, поэтому6144 теперь читается из GGUF
metadata `minimax-m3.embedding_length`, без загрузки основных весов.

Шесть CPU IPC tests прошли: сохранение абсолютных позиций/порядка, отказ при
overlap/reordering/bad serialized grid/nonfinite или коротких embeddings,
обнаружение изменённого F32 значения. `vision_batch.py compare` сравнивает
реальные encoder files по frame ID/grid/shape, maxabs и relative L2.

Повторная подготовка:

```bash
python3 tools/m31/vision_prepare_jobs.py /mnt/ai/logs/m31-port/vision-next
sudo systemd-run --unit=codex-m31-vision-next -p User=nekto \
  /bin/bash /mnt/ai/Xolotl-m31/tools/m31/vision_preprocess.sh \
  /mnt/ai/logs/m31-port/vision-next
python3 tools/m31/vision_verify_reference.py /mnt/ai/logs/m31-port/vision-next
```

`vision_prepare_jobs.py --encode` включает настоящий CPU encoder через
`mtmd_batch_add_chunk/mtmd_batch_encode`; единый пакет кадров собирается после
кодирования независимо от поддержанного mtmd encoder batch size. Не запускать
encode в окно скорости. На первом этапе encode не запускался; его последующая
проверка описана ниже. Сравнение pixel values не заявляется: публичный API
позволяет выгрузить preprocessing geometry, но не сами normalized pixel buffers.

После сверки CPU encoder отдельно реализовать native
`Engine.forward_embeddings`/mixed rows, проверить main logits против mtmd,
и только затем подключать HTTP vision и измерять TTFT. Текущие server/engine
файлы не менялись, GPU/пресеты/aictl не затрагивались.

## Второй этап: настоящие F32 encoder rows

Добавлен `encoder_mode="single"`: прямой `mtmd_encode_chunk()` +
`mtmd_get_output_embd()`. Исходный `encoder_mode="batch"` использует
`mtmd_batch_add_chunk()` + `mtmd_batch_encode()` + per-chunk output pointer.
Оба пути используют один audited PR backend; это проверка adapter/IPC,
а не независимая проверка native encoder Xolotl.

Первый PNG CPU16: encoder1.301с, полный процесс2.60с, maxRSS1.92GiB.
Получены1837056 конечных F32 values, shape `[299,6144]`; диапазон
[-35.26296,793.1085]. Повтор через single API: encoder1.299с.
Это диагностическое время encoder, не M2 speedbenchmark и не полный TTFT.

Видео через batch API: encoder14.646с,12×299=3588 visual rows.
Готов единый mixed-row пакет3863 позиций с F32 `[3588,6144]`
(`video-batch-batch/visual.f32`,88178688байт). Настоящий mtmd encoder PR
**не поддерживает multi-image batching MiniMax**: даже с budget4096
`mtmd_batch_add_chunk()` разбил12 кадров на12 encoder calls. Общий LLM
пакет всех кадров при этом собран один раз. Не выдавать это за ускорение
или единый encoder forward всех12 изображений.

Рабочие артефакты `/mnt/ai/logs/m31-port/vision-encode-v1/`.
Каждый encode выполнен отдельным CPU процессом под shared `.build.lock`
через `systemd-run User=nekto`, GPU скрыта, use_gpu=false,16 потоков.
`vision_validate_encode.sh` последовательно выполняет independent API paths
и пишет image/video compare reports со строгими atol0/rtol0.

Проверка завершена: unit `codex-m31-port4-vision-encode-compare-v1` rc0.
PNG single/batch и все12 видео-кадров single/batch **побитно совпали**:
maxabs0, relativeL2=0 по каждому кадру. Видео single encode14.679с,
maxRSS обоих видеопутей≈1.95GiB. Все F32 arrays имеют ожидаемый размер и
конечные значения; assembled visual files тоже побитно совпали.
`image-compare.json`, `video-compare.json`, `summary.json` сохраняют точные
времена, размеры, SHA256 файлов и пределы проверки. Прежние text token IDs,
frame IDs, geometry и абсолютные позиции также совпали с preprocessing-v3.

Для продолжения native mixed-input path готов пакет
`vision-encode-v1/video-batch-batch/manifest.json`:3863 последовательные
позиции,3588 visual rows,275 текстовых token IDs. Actual encoder parity
проверена между двумя PR API путями; собственного native encoder ещё нет,
native main logits и HTTP vision не проверялись. GPU не занималась.
