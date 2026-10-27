# Port7: F32 shadow для canonical F16 index-cache

Только дизайн и graph-helper proposal: `tools/m31/index_f32_shadow_proposal.hpp`. Engine, planner, CMake не менялись; сборок, CPU/GPU checks, sysctl и THP-изменений не выполнялось. Числа/источники также в `/mnt/ai/logs/m31-port/port7-index-f32-shadow-design-v1.json`.

Сейчас ordinary n1 attention каждого слоя сохраняет indexer ik в canonical F16 cache, затем CAST всего `[128,nk]` в F32 перед index GEMM. F32 shadow может хранить ровно этот cast постоянно. Сам F16 cache остаётся источником истины; FP32 ik ДО F16 rounding использовать нельзя.

## Точное обновление штатными ops

После существующего `written_f16 = ggml_set_rows(...F16index,ik,positions)`:

```
rounded = ggml_get_rows(ctx, written_f16, positions); // F16 -> F32, одна реальная stored row
updated = ggml_set_rows(ctx, F32shadow, rounded, positions);
visible = ggml_view_2d(ctx, updated, 128, nk, updated->nb[1], 0);
```

Index GEMM получает `visible`, сохраняя shape, strides, F32 значения и requested precision. Зависимость от RETURNED F16 SET_ROWS упорядочивает rounding→GET_ROWS→shadow write→GEMM. Нельзя подменить `written_f16` leaf alias или `Graph::borrow(updated)`: borrow отрежет dependency. Leaf alias владельца shadow до SET_ROWS допустим. Mutable shadow buffer не маркируется WEIGHTS.

Bootstrap после prefill — один `ggml_cpy(F16cache,F32shadow)` на слой, до первого использования shadow. Копируются все физические строки выделенного shadow span, включая padded/stale tail, чтобы invariant был побитовым physical CAST, а не только равенством видимого префикса. CUDA CPY поддерживает F16→F32. Можно выделить один contiguous owner для N слоёв и привязать per-layer2D views: это один большой allocation вместо60.

## Трафик и память

Для60слоёв, dim128:

| Span | Gross F32 shadow | На слой | Убираемый F16 read + F32 write/token | Только bandwidth bound при1792GB/s |
|---:|---:|---:|---:|---:|
| 32K | 0.9375GiB | 16MiB | 1.40625GiB | 0.843ms |
| 40192 | 1.1499GiB | 19.625MiB | 1.72485GiB | 1.034ms |
| 64K | 1.875GiB | 32MiB | 2.8125GiB | 1.685ms |

GEMM по-прежнему читает полный F32 index. Убирается только conversion traffic; update одной строки добавляет≈105KiB/token по60слоям. При stock ops один fullCAST/kernel заменяется двумя маленькими GET_ROWS+SET_ROWS, то есть ещё≈60 kernels/token. Поэтому при4K (удаляется≈0.1ms идеального bandwidth traffic) overhead может съесть пользу. При32/64K идея перспективнее, но эти нижние оценки времени трафика не являются измеренным ускорением.

**Gross не равен net VRAM.** Stable ordinary decode уже владеет60 отдельными Graph/gallocr; fullCAST F32 output может доминировать scratch каждого. После его удаления часть gross shadow заменит существующий scratch. При64K полныйCAST32MiB/слой больше sorted K/V gather scratch. Но точный net budget требует allocator SUM old/new: lifetime-overlap и другие outputs могут сохранить high-water. При short activeNk и большом configured context выделение full configured span тратит память, почти ничего не заменяя.

## Текущий headroom и admission

Старый статический admission table для cache7GiB: ctx40000/cap40192 имеет10.44GB до context scratch;64K —6.94GB, после условного2GiB reserve —4.79GB. Это header-derived planning, НЕ фактический64K runtime proof.

Фактический frozen V8 после32K prefill ещё держал dense_workspace3.658GB, expert scratch1.510GB, hidden0.503GB; backend_free≈2.486GB ДО decode. Просто добавить1.15GiB shadow поверх этого и старых dense graphs опасно. Сначала нужно сохранить prompt head/raw sample и освободить PP scratch/captures, затем admit shadow и строить новый n1 graph. V6 при3324prompt/ctx40000 после PP показывал free7.013GB, steady K8 verifier6.632GB; ownership отличается от ordinary. Эти числа не заменяют новый memory receipt.

Начальное admission должно быть консервативным по gross, не по предполагаемой экономии scratch. Выделять N≤60 детерминированных слоёв, укладывающихся в explicit shadow budget и actual free minus reserve. Остальные слои сохраняют старый CAST. Все слои одинакового index shape имеют одинаковую gross стоимость; первая версия может брать первыеN. Не следует автоматически вытеснять expert cache или наугад проверять cudaMalloc OOM. Отдельно учесть lookup maps, KV, GPU expert pool, retained dense/verify graphs, sidecar/caller residents и драйверный запас. Planner считает actual shadow owner ровно один раз суммой.

Span policy — отдельный выбор. Полный configured span стабилен до reset, но перерасходует память при коротком prompt. Shadow с active paddedNk дешевле; grow на смене nk требует нового buffer/копии physical span и пересборки captures. Такая смена уже меняет обычную dense graph geometry, но отсутствие дополнительной стоимости ещё не доказано. Geometric growth уменьшает число allocations ценой extra reserve. Не нужен shadow30GiB для1M context на этой карте: budget fallback должен оставить неподдерживаемые слои/режимы на штатном CAST.

## Lifecycle

- Allocate/initialize только после PP transition; на prefill постоянной памяти shadow нет. Первый decode оплачивает bootstrap/allocations, так что короткие ответы могут проиграть. Нужна окупаемость по реальным128/256-token запросам.
- Graph/capture consumers уничтожаются до shadow owner; сначала drain backend/shared work. Перестройка на другой shadow span не может оставить старые указатели CUDA graphs.
- Rewind меняет logical visibility, не physical F16 bytes: shadow сохраняет такие же stale rows. Следующий branch перезаписывает и F16, и F32 одну и ту же позицию.
- Reset/cold PP/любой multi-token ordinary forward инвалидирует/освобождает shadow ДО новых canonical writes; перед cold PP planning память освобождается, иначе PP chunk урежется. Exact prefix-hit без нового PP может сохранить owner, если physical invariant доказан.
- Ошибка после F16 write, но до shadow write может нарушить invariant. Engine остаётся poisoned; reset обязан сбросить/rebootstrap shadow, а не разрешать его reuse.
- Начальный scope — ordinary n1. На входе Engine.verify shadow нужно invalidate/drop либо отдельно реализовать и доказать multi-query write ordering. Speculative KV bytes после rewind нельзя оставлять неотражёнными в ordinary shadow.

## Предлагаемый компонентный proof

1. Bootstrap и все physical F32 bytes против baseline backend fullCAST F16 (включая padded/stale data), отдельно CPU/CUDA.
2. SET_ROWS обновления позиций0,last,block boundary, repeated/rewound slot; F32 около F16 rounding midpoints, signed zero, subnormals, overflow. Expected берётся от фактически записанного F16, не исходного ik.
3. После каждой операции сравнить fullCAST и shadow bytes; для finite fixtures сравнить F32 index score и top_k, сохранив GEMM dtype/shape/precision. Одного совпадения cache bytes недостаточно для подтверждения неизменного backend dispatch.
4. Reset/new PP/bootstrap, отбрасываемая future branch, failure/recovery, mixed shadow/stock layer subset; teardown captures до владельцев.
5. Снять transition cost, steady component CUDA events, actual free/allocator SUM, затем ordinary gate20/long и реальный TG. До этого default не менять.

Вывод: это разумный bounded opt-in для длинного activeNk, с потенциально небольшим net VRAM благодаря замене scratch. Не доказано, что резерв окупится; при коротком контексте возможен отрицательный результат. Сейчас рекомендация — сохранить дизайн как reserve, сначала component/allocator evidence.
