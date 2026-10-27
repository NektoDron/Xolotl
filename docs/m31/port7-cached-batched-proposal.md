# Port7: GPU cached subset в слоевом verifier

Компоненты proposal прошли проверку. По разрешению root после freeze V5 helper
и thin wrapper применены в live `gpu_cached_exact.*` и `m31_gpu_experts.*`.
Engine verifier меняет отдельный владелец; проверка финальной интеграции ещё
не выполнена. Исходный patch и source hashes сохранены в
`/mnt/ai/logs/m31-port/port7-cached-batch-proposal-v1/`.

Идея: при K≤8 CPU miss FFN занимает больше времени на слой, поэтому старые
CPU-compatible GPU dots для cached hits и два join проще скрыть этим окном.
Новые CUDA kernels не нужны. Native cached math уже отвергнута по KL; здесь
каждый dot сохраняет исходную n1 quant/dot/OAI арифметику.

## API

`GpuCachedExact(embd, ff, maximum_used=4, maximum_tokens=8)` выделяет маленькие
фиксированные буферы. Старый конструктор с тремя аргументами и `begin()`
сохраняются и делегируют n1 варианту.

`begin_batched(types, pools, slot_stride, x, n, assignments, weights, scales)`:

- `x`: полный token-major F32 `[n,embd]`, копируется до возврата.
- `assignments`: только GPU hits, `{token,rank,slot}` в строгом token/rank порядке.
  Одинаковый slot разрешён для разных token; внутри одного token дубликаты
  отвергаются. Это физические слоты из проверенного cache lookup.
- `weights/scales`: по одному значению на hit assignment.
- `finish()`: weighted F32 `[hits,embd]` в том же порядке assignments.

Worker загружает x один раз, считает gate/up для всех hit assignments на одном
stream, выполняет первый D2H join, CPU scales+libm OAI, затем down для всех hits
и второй D2H join. Per-assignment вызовы `GpuDeviceDots` остаются n1 и
`original_tiled=false`, включая повторяющийся эксперт на8 токенах: цель —
ordinary n1 arithmetic, а не tiled math по histogram8.

При n8/used4 максимальные buffers: GPU2169720B, pinned2162688B, вместо
277368/270336B у n1 helper. Отдельный per-token join отсутствует.

CPU API согласован с агентом routed CPU:

`CpuBatchedVerify::forward_masked(weights/scales, x, full_ids, full_weights,
                                cpu_mask, output_cpu_terms)`

Маска `[n,used]`, `1=CPU miss`, `0=GPU hit`. На выходе полные weighted terms
`[n,used,embd]`, в GPU slots —0. CPU не должен заранее суммировать misses.
После finish GPU terms с assignment-map и CPU terms складываются по исходному
rank0→rank1→rank2→rank3 отдельно для каждого token/row. GPU partial sum + CPU
partial sum запрещена из-за изменения F32 reassociation.

## Будущая интеграция

Отдельный cached-only verifier API должен брать union hit keys под lease до
finish/drain настоящего private stream, включая ошибки CPU callback. Никакого
weight admission на misses: они остаются на CPU. Полный lookup/validation до
submission, all-miss возвращает обычному CPU verifier, all-hit не вызывает CPU
graph. Маска строится по original IDs, а GPU assignments используют их слоты.

Нельзя вызывать обычный streaming `forward_into(n>1)` ради этого пути: он
снимает decode pins и загружает miss weights. Нужен именно cached-only batch
API с отдельным opt-in, native prefill сохраняет существующий streaming путь.
GPU cache290/prefill unpin контракт уже проверен отдельным lifecycle proof.
Planner должен учитывать retained worker buffers суммой с verifier/dense owners.

## Реальный потолок трафика

На двух held-out autoregressive кейсах290 slots дают19,08% route hits.
Но после CPU union-reuse кеш убирает только14,83% уникальных весов при K4 и
12,78% при K8: популярные hot эксперты уже амортизированы batch.
Остаточный CPU трафик относительно обычного n1 —0,61881 при K4+cache и0,48531
при K8+cache. Для162 slots дополнительная экономия union —9,74% и8,15%.

Это traffic potential, не speedup. Helper всё ещё делает несколько точных
kernels на каждый hit assignment; при all-hit и коротком CPU окне он может
оказаться медленнее CPU. Primer внутри генерации входит в TG.
Источник: `port7-actual-decode-rank-v1/cached-union-windows.json`.

## Проверка proposal

Standalone component готовит K1/2/4/8 × all-hit/half-hit/one-hit-per-token на
реальных GGUF слоях3/10/52 и L3 activation rows. CPU n1 FFN reference исполняется
параллельно GPU worker. Caller-owned x/weights/scales мутируются сразу после
begin, чтобы проверить копирование/lifetimes. Проверяются weighted term bits,
старый n1 API, сортировка/уникальность assignments и верхняя граница K.
Это39 случаев/~2M F32 и6 invalid-input probes. Пока component не прошёл,
proposal не считается готовым к интеграции или model/quality acceptance.

## Проверка компонентов и текущая обёртка

Standalone helper: 39 кейсов, 2 009 088 weighted F32, 6 проверок неправильных
assignments; K=1/2/4/8 и IQ3/IQ4/NVFP4 — побитное совпадение с исходным n1 CPU.
Joint GPU helper + настоящий `CpuBatchedVerify::forward_masked`: 39 кейсов,
3 391 488 full-rank weighted F32, побитное совпадение, 78 joins. Результаты:
`port7-cached-batch-component-v1/component-result.json` и
`port7-cached-batch-joint-component-v1/component-result.json` в каталоге логов.

`GpuExperts::forward_cached_batched` включается `M31_DSPARK_GPU_CACHED=1`.
Отдельный worker ограничен 8 токенами и 32 hits. Вызывает decode-only seed
один раз, затем не переносит и не вытесняет веса. Полный wall включает primer,
submit, CPU callback, join и merge. При all-miss возвращает false без callback
и изменения output; при all-hit CPU callback не вызывает. RAII lease дожидается
настоящего private worker и очищает защиту слотов при исключении.

Direct wrapper component на малом GGUF с неизменёнными блоками исходных
квантованных весов собран: `build-port7-cached-wrapper/v1`, ELF SHA256
`93d05a015bfb2e02c450b56ddb6cb806ff490489e7dfa29e003539140892b64c`. CPU-only
и CUDA без optional CPU-quants syntax PASS. Runtime job
`port7-cached-wrapper-component-v1` прошёл: 16 кейсов K1/2/4/8, 11 520
выходных F32 побитно. Проверены all-hit/partial/all-miss, две ошибки callback
с восстановлением, 20 calls/92 tokens/368 routes/184 hits, 14 успешных
forwards и 4 all-miss. Worker сделал 16 jobs/32 joins; seed сработал один раз,
после него отсутствовали weight DMA/evictions. Это
не проверка качества модели и не измерение скорости.

Receipt: `/mnt/ai/logs/m31-port/port7-cached-wrapper-component-v1/component-result.json`.
Engine integration/modelgate/greedy для нового batched режима ещё ожидаются.

Read-only review Engine hook (после изменения владельцем DSpark): `input` и
`hidden` разделены; callback получает original-rank mask; all-miss fallback
сохраняет прежний `CpuBatchedVerify::forward`; shared/residual прибавляются
после full-rank merge. `cpu_ms` охватывает весь hybrid call и fallback,
`hybrid_ms` показывает только вызов GPU wrapper, `cpu_ffn_ms` — настоящий CPU
FFN callback/fallback. Проверка n≤8 выполняется до cache call. Это обзор исходника,
а не runtime model acceptance. Cached agent Engine не редактировал.
