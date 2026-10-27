# M31: смешанный вход движка

27.10.2026. `Engine::forward_mixed(tokens, visual_rows, visual_embeddings,
visual_count, n, error)` использует обычный послойный prefill, включая v8
`M31_GPU_PREFILL_CHAIN`. Для текстовой строки `visual_rows[i]=-1`, token ID
обычный; для визуальной `tokens[i]=-1`, индекс выбирает строку F32 `[6144]`
из общего буфера. Замена выполняется перед единственным upload начального
hidden state; attention, эксперты и output head выполняет Xolotl.
`forward(tokens)` вызывает тот же путь без визуального буфера.

Позиции проверены по PR `6b95a9908d2c7b242817a447708aec38e937e2c0`:
`src/llama-model.cpp` относит `MINIMAX_M3` к `LLAMA_ROPE_TYPE_NEOX`;
`tools/mtmd/mtmd.cpp:595` выбирает `MTMD_POS_TYPE_NORMAL`, а
`mtmd_image_tokens_get_decoder_pos()` присваивает `pos_0+i`;
`mtmd_image_tokens_get_n_pos()` возвращает число visual tokens.
`mtmd-helper.cpp:139–196` использует одну позиционную ось и
`set_position_normal`. Поэтому main decoder сохраняет `position()+i`,
обычные causal/MSA block coordinates и partial NeoX RoPE. Пространственный
RoPE энкодера не переносится в main decoder. Разметку границ изображений,
текст timestamps и порядок кадров создаёт mtmd; native IPC не меняет их.

`m31_mixed.hpp` загружает `m31-vision-batch-v1` и до изменения KV проверяет
receipt формата/PR, размеры F32/I32, словарь, полное покрытие chunks,
непрерывные линейные позиции, упорядоченные visual rows и конечность F32.
JSONL сервер принимает `mixed_manifest` как локальный request-owned путь.
Исходное `tokens` сохраняется для Python grammar controller, native prefill
использует полный смешанный массив из manifest. Общая последовательность
делится только по штатному общему `prefill_chunk()`, независимо от кадров.
При достаточном chunk/VRAM все кадры проходят одним forward. Фактическую
разбивку correctness инструмент записывает в `actual_chunks`.

`m31_generate --mixed-manifest /abs/manifest.json --out NEW_DIR` принимает
смешанный вход вместо `--ids-file`/`--cases`; формат выходных logits прежний,
`m31-logits-v1`. Для одинаковой арифметики output head его ширина/число строк
должны совпасть с oracle (`--head-tail` и `--rows`). Текстовый oracle отклоняет
mixed-вход: для него существует отдельный vision oracle с F32 embeddings.

Нативный CPU preflight проверен на реальном video manifest (3863 полных
строки / 3588 визуальных) и 14 synthetic valid/rejection cases:
`/mnt/ai/logs/m31-vision/native-mixed-preflight.json`. Это проверка контракта
и ошибок, не main-logits parity и не замер скорости. Итоговые GPU/parity/TTFT
числа публикуются координатором в `/mnt/ai/logs/m31-vision/REPORT.md`.

Прямой CPU тест `Engine` на tiny F32 fixture также прошёл:
`/mnt/ai/logs/m31-vision/engine-preflight-v1.log`, unit
`codex-m31-vision-engine-preflight-v1`, rc0. Две внутренние строки заменены
их настоящими F32 token embeddings и переданы как visual rows: все logits
побитно равны обычному text forward на том же chunk. Отказ на ошибочном
visual map и NaN сохраняет position, прошлые logits и возможность повторить
валидный append без reset. Этот тест не заменяет vision GPU oracle.
