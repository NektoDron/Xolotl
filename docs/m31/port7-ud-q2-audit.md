# Port7: UD-Q2_K_XL — локальный аудит совместимости

Обновление14:37UTC27.10.2026: доступны **все4 GGUF headers,1188 tensors**. Прочитаны только метаданные, без tensor payload, загрузки движка, GPU и скачивания. Новый immutable JSON — `/mnt/ai/logs/m31-port/port7-ud-q2-header-audit-v2.json`; старый v1 сохраняет частичный снимок836 tensors. Reader `tools/m31/port7_udq2_audit.py` не изменялся после source freeze: v2 создан вызовом тех же read-only функций с output в logs. На свежем read-only снимке15:11UTC все4 файла имеют ожидаемые logical extents и *.aria2 отсутствуют (`port7-ud-q2-download-status-v3.json`). Payload checksum, runtime load и качество не проверялись; новая модель в этой волне не запускалась.

## Полные заявленные объёмы и типы

Shard1 — metadata-only GGUF,0 tensors/8,243,332bytes; shard2 —436 tensors/49,084,401,440bytes; shard3 —400/49,125,534,656bytes. Ожидаемый extent shard4 из352 headers —45,691,889,152bytes, его payload на header-снимке14:37 был частичен, последующий status15:11 имеет полный extent без sidecar. Declared tensor payload всего **143,901,749,760bytes=134.018948GiB**, включая routed experts **133,857,017,856bytes=124.664063GiB**. Метаданные/выравнивание файлов, KV, scratch, file cache и GPU allocations сюда не входят.

Архитектура совпадает: minimax-m3,60 layers,3 dense lead, embedding6144, expert FF3072,128 experts/4 selected,1 shared, MSA128/4heads/top16, context1M. Индексы слоёв GGUF zero-based. Ни одного GGML Q2_K tensor нет: название UD-Q2_K_XL не задаёт фактический expert type.

| Tensors | Формат и block geometry | Число | ОбъёмGiB | Слои/назначение |
|---|---|---:|---:|---|
| Routed gate/up | **IQ2_XS**,256elements/74bytes |112 |72.843750 |gate/up по56 tensors, layers3–58 |
| Routed IQ3 | IQ3_XXS,256/98 |56 |48.234375 |gate/up layer59; down3–51,53–57 |
| Routed IQ4 | IQ4_XS,256/136 |3 |3.585938 |down52,58,59 |
| Dense/shared/attention/embedding | Q5_K,256/176 |237 |6.233322 |Q/O0–58, gate/up dense/shared0–58, token_embd |
| Dense/shared/attention | Q6_K,256/210 |179 |1.254501 |K/V0–58, большинство down; protected Q/O/GU59 |
| Output head | Q4_K,256/144 |1 |0.643936 |output.weight |
| Protected down/KV | Q8_0,32/34 |5 |0.174316 |dense down1–2; K/V/shared-down59 |
| Norms/indexer/router/bias | F32,1/4 |595 |1.048810 |0–59 |

Полный actual same-name metadata diff со старым UD-IQ3_S — `port7-ud-q2-vs-iq3-header-full-v2.json`. Expert payload163,754,016,768→133,857,017,856bytes: **−18.2573%**, без экстраполяции невидимых слоёв. Gate/up112 tensors IQ3→IQ2; down12 tensors IQ4→IQ3; down52/58/59 NVFP4→IQ4, поэтому3 старых external scale tensors исчезли (1191→1188 tensors). Вся математика нового кванта требует своих oracle/quality гейтов; результаты старого V8 не переносятся.

Стандартный routed слой с IQ2 gate/up и IQ3 down содержит2.162109GiB raw weights/128experts; layers52/58 с IQ4 down —2.496094GiB, protected59 —2.917969GiB. CPU arena layouts считаются по фактическим role bytes/alignment. GPU fixed-slot max-span и LCM теперь используют типы74/98/136bytes; cache slot count, padding и минимальный бюджет надо заново подтвердить init receipt. Поддержка type-generic streaming не означает одинаковое число slots со старым квантованием.

## Что уже поддерживает код

| Путь | Статус по source, без runtime proof |
|---|---|
| GGUF reader/geometry | IQ2_XS256/74 уже есть в `include/strata/artifact/gguf_reader.hpp:133`; geometry проверяет block/shape/in_bounds, не требует IQ3 |
| CPU stock routed graph | Есть `ggml_vec_dot_iq2_xs_q8_K`, activation typeQ8_K и его from_float. Сам IQ2 weight from_float=null — это нормально, веса не переквантуются |
| GPU stock/native prefill | Vendor CUDA MMQ, MMVQ и GET_ROWS имеют IQ2_XS dispatch; Q4_K/Q5_K/Q6_K также штатные |
| ExpertArena/registered H2D/LRU/layer banks | Расчёт ggml_row_size, offsets и stride LCM общий. IQ2 не запрещён. Layer-bank jobs копируют role spans, не требуют одинакового byte layout со старой моделью |
| Ordinary n1 CPU fallback | Engine direct_supported=false при IQ2, поэтому cpu_direct/cpu_fused не создаются; `routed_cpu_into()` уже переходит в обычный cached RoutedGraph/stock ggml CPU |
| CPU direct/fused/batched custom wrapper | Сейчас allowlist только IQ3_XXS/IQ4_XS/NVFP4. В Engine.verify IQ2 будет отвергнут CpuBatchedVerify; автоматического stock fallback внутри этого API пока нет |
| Exact cached GPU/CPU-compatible CUDA | Только IQ3/IQ4/NVFP4. IQ2 в GpuDeviceDots не реализован; простое расширение allowlist недопустимо |
| Experimental GpuFastFfn | Gate/up разрешены IQ3/IQ4, IQ2 не реализован; явный M31_GPU_FAST_FFN=1 будет отвергнут |
| DSpark batch-dense exact API | Текущий hardened marker разрешает только F32/Q6_K/Q8_0. Новые Q5_K projections и Q4_K head не подходят; `M31_DSPARK_BATCH_DENSE=1` завершится явной ошибкой |

Source anchors: `cpu_direct_decode.hpp:19`, `cpu_batched_verify.hpp:295`, `m31_engine.cpp` direct_supported/routed_cpu_into, `m31_gpu_experts.cpp` init/stride/exact lists, `gpu_cpu_quants.cu` supported guards, `gpu_fast_ffn.cu:263`. Stock support подтверждён в vendor `ggml-cpu.c`, `arch/x86/quants.c`, `ggml-cuda/mmq.cu`, `mmvq.cu`, `getrows.cu`. Patched exact marker guard — `build-port7-dense/mmv-api-v3/vendor/ggml/src/ggml-cuda/ggml-cuda.cu`; helper требует успешный marker в `m31_dspark_batched_dense.hpp:7`. Наличие source dispatch не заменяет runtime/load/quality proof финального ELF.

## Минимальная совместимость без нового fast kernel

Самый короткий путь — после завершения всех4 файлов отдельный diagnostic профиль **ordinary n1 + native GPU prefill**. Существующий CPU RoutedGraph уже даёт stock fallback. Отключить cached decode/DSpark GPU cache, exact GPU, experimental fast FFN и DSpark; не переносить старый rank-seed receipt/model identity. Arena остаётся полной, её бюджет и GPU slots рассчитываются заново. Никакая новая реализация IQ2 арифметики для этого пути по source не нужна. Первые проверки: complete headers/in_bounds → load → short ordinary greedy против GPU-offload oracle того же Q2 → PP/long/quality отдельно. В текущей волне новую модель не запускали; отдельный runtime/quality preflight остаётся будущей задачей.

Если нужен именно Engine.verify/M3, ограниченный следующий opt-in совместимости можно сделать без ручной IQ2 математики:

1. Отдельный флаг вроде `M31_CPU_STOCK_EXPERTS=1` разрешает **только IQ2_XS дополнительно** при наличии vec_dot и activation.from_float, подходящем block geometry и int-sized dimensions. Общий helper использовать и в Engine direct_supported, и в CpuDirectDecode traits/supported. Неподдержанные прочие types по-прежнему отклонять.
2. CpuFusedDecode/CpuBatchedVerify уже вызывают stock vec_dot(nrc1), Q8_K from_float и прежний F32 OAI/rank sum. Pair helper для IQ2 возвращаетfalse и выполняет независимые stock dots; group1 тоже stock. Gate/up/down смешанных типов остаются раздельными traits. Не требовать weight.from_float — у IQ2 он закономерно null.
3. Для такого Q2 M3 профиля поставить `M31_DSPARK_BATCH_DENSE=0` и все GPU cached/exact flags0: сохранить per-query ordinary dense n1, CPU FFN layer-major со stock traits. Текущий DSpark CPU allowlist надо расширить этим opt-in; иначе verify failfast. При необходимости ещё более консервативный fallback — serial forward(1) verification wrapper, но он должен аккуратно собрать capture rows до rewind; это отдельный API test, не простая silent ветка.
4. До включения нужны bounded real IQ2 weights component: stock nrc1 против wrappers, K1/2/4/8, разные routes, mixed IQ2 gate/up+IQ3 down, all-miss mask, original rank order и reset/rewind. Затем модельный short/long gate на **этом** Q2. Старый V8 default/ELF не менять.

Этот opt-in не применён live в рамках аудита. Для plain compatibility он не обязателен; он нужен, чтобы сохранить custom wrapper/M3 возможности. Производительность IQ2 stock dot не измерялась.

## Доработки для быстрых вариантов

Приоритет после минимального stock запуска: IQ2_XS pair/unpack-once CPU helper, если профиль покажет пользу; CPU-exact cached GPU IQ2 kernel со своим lane/FMA bitproof; gate/up experimental GPU IQ2 dequant/IMMA, если он потребуется вместо уже имеющегося native MMQ. Новый IQ3 lane IMMA помогает IQ3 down, но не заменяет IQ2 gate/up kernel. Все требуют явного opt-in и quality гейта.

Для DSpark batch-dense нужны расширение hardened exact-MMV dispatch на Q5_K/Q4_K и K1–4 bitproof против штатного n1. Штатные MMV kernels для этих типов уже существуют, так что это может быть dispatch/eligibility доработка вместо новой математической реализации. Сохранить opNONE/WEIGHTS-buffer/shape safeguards, не разрешать MSA score tensors. Перепроверить role bundles, projection concat и planner/VRAM на новой смеси types. До этих работ Q2 не считать принятым port7 fast profile.
