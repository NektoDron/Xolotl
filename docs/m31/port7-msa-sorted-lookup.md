# Port7: exact n1 sorted-index lookup

Подготовлены `tools/m31/msa_sorted_lookup.hpp` и `port7_msa_sorted_lookup_check.cpp`. Компонентные CPU/GPU proof выполнены отдельно. Затем добавлен Engine opt-in, описанный ниже; frozen candidates не менялись.

Идея выразима существующим публичным API. В vendor6050b467 `ggml_get_rows` принимает F32 source `[block,blocks,heads,1]` и I32 picks `[top,heads,1,1]`, возвращает contiguous F32 `[block,top,heads,1]`. CPU `ops.cpp` и CUDA `getrows.cu` явно адресуют source через `i01*nb01+i11*nb02+i12*nb03`, indices через собственные strides. Поэтому исходный strided top_k view работает без compact copy; разные head planes выбираются корректно.

Владелец один раз создаёт два immutable F32 tensor и один backend buffer:

- token[ lane,block_id,head ] = block_id*block+lane;
- row[ lane,block_id,head ] = (block_id*block+lane)*kv_heads+head.

Helper получает тот же score-sorted idx, делает два get_rows и cast I32 после flatten. Выходной порядок lane→selected block→head совпадает с текущими tj/tr. Сами block selection, sorted ties/padded picks, K/V/mask gather, FA и attention arithmetic не меняются.

Для M3 index_heads=kv_heads=4 и padded capacity≤2^20. Token≤2^20−1, row≤2^22−1: все промежуточные целые точно представимы F32. И текущие idx*block+lane, затем *4+head остаются в этом точном диапазоне; замена не меняет округление. Helper требует равные index/KV head count и capacity*heads≤2^24. Не пытается обобщать несовместимую head geometry.

При1M capacity два source занимают32MiB; при64K —2MiB. Это один Engine-owned resident owner, а не60 copies. Planner должен учитывать его buffer size суммой ровно один раз. Все graph/capture borrowers обязаны умереть до maps; optional source aliases поддерживают существующий Graph::borrow. Использование full padded capacity позволяет переиспользовать maps при смене текущего nk; idx остаётся ограничен текущими видимыми blocks.

Proof source сравнивает прежний graph pipeline, новый lookup и независимую integer closed form. Случаи: capacities256/4K/16K/32K/64K/1M, active blocks равны либо меньше map capacity, top1/top16, descending/per-head/permuted/repeated picks; indices имеют лишние7элементов stride между head planes. Проверяются I32 outputs целиком, включая крайнюю row4,194,303. CPU и CUDA — отдельные режимы; CUDA только через `M31_CARD_SESSION=1`/aictl runner. Это component proof source, а не full attention/model numerical proof и не замер ускорения.

Ожидаемый эффект — убрать scale/repeat/arange/add pipeline и часть маленьких kernels. Фактическая экономия зависит от CUDA fusion и должна быть измерена после component/model gates. До этого никаких TPS claims.

## Выполненный компонентный proof

CPU и CUDA:96cases/1 433 600I32 values каждый, побитно old/new/closed form; strided96/head-distinct48. Actual graph nodes16→7. CUDA event us old→new:4K9.117→4.344;16K8.522→4.307;32K8.529→4.310;64K8.542→4.315. Maps/idx resident,5warmups/200repeats; это только expansion. Простое суммирование delta по60слоям≈0.25–0.29ms/token — оценка, не измеренный TG gain. Весь1M footprint32MiB прошёл component proof. Receipts: `/mnt/ai/logs/m31-port/port7-lookup-cpu-proof-v1/receipt.json` и `/mnt/ai/logs/m31-port/port7-lookup-gpu-proof-v1/component-result.json`.

## Engine integration

`M31_MSA_SORTED_LOOKUP=1`, default0. Один Engine owner создаётся до fixed-resident free-VRAM snapshot; planner учитывает его actual allocation ровно один раз. Graph borrowers получают leaf aliases через Graph::borrow. Destroy/reset порядке: все graph/capture/workspace owners освобождаются до maps, затем state/backend; ordinary reset и rewind maps сохраняют. n1 attention получает прежний idx (в том числе preselected BATCH_MSA), меняет лишь idx→tok/row expansion; sorted gather/FA и batch-prefill прежние.

Current combined Engine CPU/CUDA syntax PASS: `build-port7-dense/lookup-engine-syntax-v1/receipt.json`, Engine SHA `1ea8b919759a2865ef244cfe7b0f60398ba67e4c706a84e45f0c251c5f4d0207`. Header/inc snapshot совпал с live на момент ACK.

Короткая реальная OFF/ON проверка immutable V11 выполнена27октября14:25UTC:
`/mnt/ai/logs/m31-port/port7-p7v11-rms-lookup-gate-v1/lookup-compare.json`.
Отдельный LOOKUP1 arm при RMS0/cache0 совпал с тем же V11 OFF побайтно по
raw logits и IDs:128teacher-forced tail rows fixed256, top1=100%, KL/maxabs0.
Engine log подтверждает один map owner capacity40192/resident1286144 bytes.
Это короткая диагностика; long-tail/reset, gate20/quality и реальный TG gain
пока не подтверждены этой проверкой. Frozen P7-v8 не менялся.
