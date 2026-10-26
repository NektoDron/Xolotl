# M3/M4: подготовка по исходникам

26.10.2026. Это проект следующих этапов, **M3 и M4 не реализованы и не
проверены запуском**. Работы над ними продолжаются только после M2 и решения
о его достаточном выигрыше. Ниже проверено чтением локальных исходников
PR #31192 SHA `6b95a9908d2c7b242817a447708aec38e937e2c0`, HF snapshot из
`/mnt/ai/logs/m31-port/refs/manifest.json` и заголовков `refs/aux-gguf.json`.

## M3: контракт DSpark

Файл `dspark-MiniMax-M3.1-Flash-Q8_0.gguf`: architecture=dflash, 62 тензора,
payload2081138692 байт. Он использует обычные GQA и плотные FFN, а не
DeepSeek-V4.1 HC/MLA/MoE. Прямо подключать `Ds41DSpark` нельзя.

| Параметр | Значение |
|---|---:|
| Draft layers / hidden / FF | 5 / 6144 / 12288 |
| Q heads / KV heads / head dim | 64 / 4 / 128 |
| Partial NeoX RoPE dims / base | 64 / 10000000 |
| Window / block / noise ID | 4096 / 7 / 200058 |
| Targets GGUF | [4,18,32,46,60] |
| Targets HF | [3,17,31,45,59] |
| Markov rank / confidence input | 256 / 6400 |
| Activation / sample_from_anchor | swigluoai / true |

`common/speculative.cpp::common_speculative_impl_draft_dflash` запрашивает
именно **вход** целевого слоя через `llama_set_embeddings_layer_inp`, затем
склеивает пять residual rows в порядке метаданных для каждой позиции.
`src/models/minimax-m3.cpp` сохраняет `t_layer_inp[il]=inpL` до attention,
а `t_layer_inp[60]` — последний residual до финальной RMSNorm.
Таким образом для native engine правильные taps:

| Вход GGUF слоя | Native residual после слоя (нумерация с нуля) |
|---:|---|
| 4 | `l_out-3` |
| 18 | `l_out-17` |
| 32 | `l_out-31` |
| 46 | `l_out-45` |
| 60 | `l_out-59` до `result_norm` |

Этот сдвиг +1 объясняет отличие HF от GGUF. Использовать `l_out-4` или
нормированные `attn_norm`/`result_norm` было бы ошибкой. Форма входа encoder:
`[5*6144=30720, n]`, layer-major внутри каждого token row; `fc.weight`
имеет GGML shape `[30720,6144]`, далее `enc.output_norm.weight` RMSNorm.

В `src/models/dflash.cpp::graph<false>` есть два разных режима:

1. Embedding batch: склеенные target features проходят fc+encoder norm,
   затем каждый из пяти draft layers проецирует **этот общий encoder output**
   в свои K/V, делает per-head K RMSNorm и partial RoPE; пишет draft cache
   на исходные позиции. Это injection, а не полный forward пяти draft layers.
2. Token batch: `[последний принятый token, noise, ...]` на следующих позициях.
   Embeddings и output head берутся у основной модели, если отсутствуют в draft.
   Пять dense GQA+OAI слоёв видят injected context и текущий noise block;
   draft attention по умолчанию noncausal (флаг causal берётся из metadata).
   Window4096 применяется с исходной позиционной семантикой, не как MSA main.

В реальном sidecar нет собственного token embedding/head: шарить существующие
тензоры предпочтительнее второй загрузки. `blk.*.attn_q/k/v/output.weight`
и `ffn_gate/up/down.weight` Q8_0, нормы F32. OAI alpha1.702/clamp7/up+1,
Gemma +1 уже записан конвертером в нормы.

`build_dspark_markov_head` обрабатывает slots слева направо:
`bias=W2*W1[previous_token]`, logits_slot=base_logits_slot+bias;
previous_token следующего slot — **greedy argmax** текущего slot внутри графа.
`sample_from_anchor=true` начинает предсказания с slot0, поэтому block7 даёт
до7 draft tokens, а не6. Не переносить V4.1 block5 и его bonus-slot сдвиг.
Confidence этого dflash варианта:
`sigmoid(conf_proj*[final RMS-normalized draft hidden;W1[previous_token]]+bias)`.
`conf_proj.weight=[6400,1]`, bias[1]. У V4.1 в комментарии `ds41_dspark.hpp`
используется pre-norm collapsed hidden и confidence logit: это другой контракт.

Интеграция потребует отдельного `M31Draft`, а у main Engine — выборочных
residual taps всех обработанных rows и rollback/rewind до принятого префикса.
Нынешний append-only `forward/reset/position/logits` не позволяет откатить
непринятый хвост verify. Нужно также инвалидировать last_n/hidden/logits и
не подавать rejected target features в draft cache. KV/index rows хвоста
должны быть недоступны attention/MSA после отката; проверить переходы128/256
и случай отказа после первого draft token. Отдельные injected/noise cache
updates должны совпадать с oracle, включая принятый/bonus token и позиции.

Из V4.1 полезны контроль accepted-prefix, проверки verify logits независимым
prefill (`tools/v41/eval/dspark_tap_kl.py`, `vdump_cmp.py`), хранение окна только
принятых features и speculative sampling в `ds41_generate.cpp`. Семантику
транка, HC/MLA kernels, confidence input и block offsets переносить нельзя.

Приёмка M3: сравнить пять taps, encoder projection и injected K/V с oracle;
затем base+Markov logits, confidence и draft IDs на одинаковых blocks.
Greedy итог должен совпадать с ordinary greedy, а verify logits — с независимым
prefill того же потока. Для stochastic sampling сохранять distribution через
корректный accept/reject/residual sampler; confidence разрешает укоротить
draft, но не заменяет verification. Измерять acceptance и wall time совместно.

Референс при p_min=.75 принял86–90%, но дал TG≈22 и PP375 вместо обычных
23.33/425 на4K. Это аргумент сначала ускорить verify; acceptance сама по себе
не является критерием выигрыша. Полный M3 benchmark повторяет M2 глубины.

## M4: изображение и видео

`mmproj-BF16.gguf`:524 тензора,1735264256 байт payload; architecture=clip,
projector_type=minimax_m3. Vision32 layers, hidden1280, FF5120, heads16,
patch14, spatial_merge2, output6144. HF image mean/std и dynamic-res processor
лежат в `refs/hf-preprocessor_config.json`; nominal672×672 не означает
фиксированное разрешение всех inputs. Resize кратен28, лимиты/округление и
нормализация должны совпадать с выбранным mtmd-референсом.

`tools/mtmd/models/minimax-m3.cpp` описывает encoder:

- Сумма двух patch convolution kernels (`patch_embeddings_0/1`) с patch stride14.
  Для одиночной картинки это соответствует повторённому temporal patch input;
  t-axis сейчас оставлена без вращения в локальном mtmd graph.
- Spatial merge переупорядочивает patches так, чтобы2×2 соседи были подряд;
  `clip.cpp` генерирует h/w positions в порядке bh,bw,mh,mw.
-32 слоя ViT с LayerNorm/GELU_ERF и 2D partial NeoX RoPE. Только h/w части
  головы вращаются со смещениями axd/2*axd, где
  `axd=2*((2*(head_dim/2)/3)/2)` в целочисленной арифметике.
- Per-patch MLP `mm.1 → GELU_ERF → mm.2`, затем concatenate2×2 patches и
  merger MLP `mm.merger.fc1 → GELU_ERF → mm.merger.fc2` до6144.

Для первого пути лучше отдельный mtmd encoder adapter, связанный с тем же
зафиксированным PR, и свой main Engine `forward_embeddings([6144,n], positions)`.
Нельзя выдавать запуск LLM в llama.cpp за native Xolotl vision: mtmd только
строит embeddings, attention/experts должны выполняться нашим Engine.
Отдельные библиотеки/процессы избегают смешения двух несовместимых ggml ABI.
При IPC передавать F32 embeddings, frame order, позиции, grid и token spans.
Это проект интерфейса; adapter и forward_embeddings пока отсутствуют.

Границы vision — `]<]start of image[>[` и `]<]end of image[>[` по mtmd.cpp;
HF placeholder IDs image200025/video200026 — не embedding обычного слова.
Текстовые границы токенизируются общим NFC tokenizer, а vision span заменяется
encoder rows. Не подавать сотни повторённых image IDs вместо actual embeddings.
У main текстового M3.1 partial NeoX RoPE; 2D RoPE vision encoder не следует
механически переносить на LLM positions. Точное interleaving/text positions
сверять с mtmd chunk metadata и oracle batch, не с предположением о Qwen MRoPE.

HF video processor имеет temporal_patch_size=2: pad нечётного числа кадров
повтором последнего, merge temporal pairs и prune соответствующих placeholders
и timestamps. **Локальный mtmd ref идёт другим поддержанным путём:12 отдельных
frames при1fps, по299 visual tokens**, без joint temporal encoder для пары.
Первый M4 video должен воспроизводить именно этот reference, с теми же кадрами
и timestamps. HF joint temporal path — отдельное расширение, его нельзя считать
проверенным по parity12 независимых картинок.

Фикстуры уже есть в `/mnt/ai/logs/m31-ref/media/`: `0.png` и
`shapes-12s.mp4`, ffprobe.json. Серверный ref на картинку:299 visual tokens,
515 prompt tokens,5.120с prompt/5.238с TTFT. Видео:12frames/3588visual,
3863prompt,35.846с prompt/36.171с TTFT. Фактические переходы4с и8с,
ответ llama.cpp называл5–9с и10с; это не эталон точной временной локализации.

Сначала сверить resized pixels, patch order, encoder output shape/values,
text/embedding spans и main logits одной PNG. Затем12 фиксированных кадров
в том же порядке. После parity можно собрать embeddings всех кадров в общий
layer-batched LLM prefill: главное ожидаемое преимущество по ref log — избегать
12 отдельных прогонов CPU-expert streaming, а не ускорять≈25мс encoder/frame.
Сравнить описание формы/цвета/положения, порядок стадий и отдельно границы
времени. TTFT включает HTTP/preprocessing/tokenization; prompt_ms их не заменяет.

## M2: наблюдения по текущему MSA path

Это read-only анализ исходного correctness engine, **без профиля wall time**.
При обычном routed layer создаются три transient Graph: attention/router/shared,
CPU MoE, GPU residual add. Каждый allocate/compute включает synchronization,
а destructor очищает CUDA graph cache до освобождения buffers. На decode это
даёт174 таких Graph для57 routed слоёв, плюс3 dense и head.

Residual/shared/norm читаются на CPU; residual/shared затем возвращаются
на GPU для двух add и сразу читаются обратно. При сохранении F32 арифметики
`(moe+shared)+residual` можно проверить opt-in CPU addition и убрать этот
roundtrip; требуется raw-logits gate, а не утверждение об эквивалентности.

positions, causal mask, local bias и cell→block map одинаковы у60 слоёв
одного chunk, но каждый layer заново строит host vectors и uploads. Можно
precompute once per chunk; общий device input должен жить до завершения всех
графов. No-tap путь не должен сохранять debug outputs как graph outputs.

Безопасный первый шаг — reusable graph allocator/CPU thread pool и общие
inputs, сохраняя ops, shapes, row ordering и F16/F32 conversions. Настоящий
cache persistent graph должен учитывать layer, n, padded nk, top, tap mode и
start-dependent KV view offsets; RoPE positions и masks обязаны обновляться.
Просто изменить host `pos` без обновления graph views некорректно. Удерживать
все возможные nk per layer без eviction тоже нельзя из-за VRAM.

Для CUDA MSA нельзя прямо подставить V4.1 lightning indexer: у M3.1 per-GQA
group dot+causal mask **до block max-pool**, local-block force, top16 blocks128,
а V4.1 использует другую агрегацию голов и compressed KV. Переиспользуется
инфраструктура `test_ds41_indexer.cpp` для capture/replay и короткий visibility
input, но новое fused dot/pool/top-k ядро нуждается в собственном replay gate:
F16 cache→F32 до score, точное tie/order и masked future rows. Decode FA gather
пока score-sorted; переход к хронологическому порядку меняет сумму attention.
Main/index min pad256 выбирать как в oracle: смена на128 уже ломала parity
из-за другого FA tiling. Fusion до локализации прежней ошибки остаётся disabled.
