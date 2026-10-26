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
encode в окно скорости. В этой контрольной точке encode не запускался,
raw F32 embeddings и их parity ещё не измерены; сравнение pixel values тоже
не заявляется: проверены resize geometry/token spans.

Следующее: CPU encode PNG (и повтор с другим encoder batching для проверки
детерминизма), затем12 кадров. После этого отдельно реализовать native
`Engine.forward_embeddings`/mixed rows, проверить main logits против mtmd,
и только затем подключать HTTP vision и измерять TTFT. Текущие server/engine
файлы не менялись, GPU/пресеты/aictl не затрагивались.
