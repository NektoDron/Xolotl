# Port7: динамический размер GPU expert pool — только проект

Runtime код не менялся, GPU/сборки не запускались. Анализатор
`tools/m31/port7_cached_growth_design.py` читает существующие greedy routes,
размеры исходных весов и V5 backend memory log. Результат:
`/mnt/ai/logs/m31-port/port7-cached-growth-design-v1.json`.

Вариант имеет смысл как резерв после V6: при context40000 увеличивать pool с
290 до400 слотов после освобождения prefill scratch и загрузки DSpark. Перед
следующим холодным префиллом нужно выгрузить drafter и вернуть290 слотов.
475 слотов не проходят даже резерв1GiB по имеющемуся warm memory minimum.
При64K400 слотов не допускает даже KV-only экстраполяция без резерва.

## Величина возможного выигрыша

Один slot — 25 909 632 байта, включая роли/выравнивание. Числа replay относятся
к двум held-out реальным greedy последовательностям, не к отвергнутым draft
траекториям M3. Assignment hit rate нельзя считать экономией чтения CPU union.

| Slots | Pool, GiB | Дополнительно к290 | Held hit assignments | GPU cacheable CPU union, K4 | То же K8 |
|---:|---:|---:|---:|---:|---:|
|290|6,9978|0|19,0841%|14,8295%|12,7773%|
|400|9,6521|2 850 059 520 B|23,3941%|18,6294%|16,2344%|
|475|11,4619|4 793 281 920 B|25,6613%|20,8870%|18,3898%|

Относительно290 CPU union traffic при K4 уменьшается ещё на4,46% для400 и
7,11% для475. При K8 — на3,96% и6,43%. Верхняя граница ускорения только
чтения routed CPU: K4 — 1,0467×/1,0766×; K8 — 1,0413×/1,0687×.
Это не прогноз end-to-end TG: GPU hits требуют дополнительных exact dots,
CPU OAI, joins и primer; работа GPU может перестать скрываться CPU misses.

Полная замена pool теряет все прежние веса. Raw prefix seed занимает
6 533 627 904 B для290, 9 003 220 992 B для400 и10 686 136 320 B для475.
Это больше наблюдавшихся5,989GB primer290: там часть hot весов уже оставалась
после префилла. При57,6GB/s нижняя оценка полного нового seed400≈156ms,
seed475≈186ms без allocator/launch overhead. Переключение нужно делать ДО
первого `prime_decode_seed()`, иначе оплата seed290 и seed400 суммируется.
Все resize/seed затраты входят в TG wall; вычитать их нельзя.

## Бюджет памяти

Используется `ggml_backend_dev_memory` в точке перехода, а не номинальная
VRAM/NVML total. В V5 shared_inputs sweep при context40000 backend warm decode
minimum free=5 853 741 056 B; максимум retained verifier workspace=366 516 736 B.
В V5 ещё нет нового batchworker: дополнительно вычтены2 169 720 B.

| Context | Slots | Проектируемое free после grow | Резерв2GiB |
|---:|---:|---:|:---|
|40000|290|5 851 571 336 B|да|
|40000|400|3 001 511 816 B|да|
|40000|475|1 058 289 416 B|нет, также меньше1GiB|
|65536|290|2 348 016 776 B|да, небольшой остаток|
|65536|400|−502 042 744 B|нет|

64K строка вычитает только дополнительный target F16 KV/index
3 503 554 560 B. DSpark ring зависит от WINDOW4096, не от target context,
но будущие graph/backend buffers всё равно требуют проверки. V5 измерен около
позиции3,4K: это не доказательство64K или32K-position memory admission.
С резервом2GiB арифметический максимум432 slots при40K и297 при64K; с1GiB —
474/339. Это верхние пределы до будущих владельцев, не разрешённые профили.

При первой реализации лучше400slots+reserve2GiB при40K, без автопоиска
максимума. Проверить реальные workspace для N1..8/нескольких geometry прежде
чем делать boundary allocation. До первой проверки нужно зарезервировать ещё
не созданные verifier graphs/draft graph shapes или сначала прогреть безопасный
round, не запуская прежний primer290. При каждом запросе повторяется реальный
memory query; заранее выписанный receipt не подтверждает наличие VRAM.

## Владельцы и связи из исходников

| Owner в GpuExperts::Impl | Зависимость от размера/адреса pool | Что требуется при замене |
|:---|:---|:---|
|`pool`, Stats resident/pool/budget/slot_count|одна device allocation|дренировать всех; free old; allocate new; обновить реальные get_size/base|
|`slot_metadata`, `pool_proxy`, `slot_proxies`|GGML tensors с указателями в старый pool|создать новый Context/proxies; старые уничтожить до free pool|
|`entries`, `free_slots`, LRU clock|эксперт→slot index|очистить; все новые slots пусты; не переносить старые slot IDs|
|`seeded_keys`, `decode_seeded_keys`, `decode_seed_order`, armed|pins и prefix verified rank|очистить pins; установить новый prefix только после новой проверки; seed после ready metadata|
|`cached_decode_graphs[layer]`|native n1 persistent graph содержит `weight_view`, shape `slot_count`, scales[slot_count]|clear captures пока old buffers живы; уничтожить все graph/context/allocators; пересчитать decode scratch|
|prefill `allocator`|retained graph allocation и backend captures|`release_prefill_workspace()` уже завершён; очистить captures и пересоздать пустой allocator|
|`uploads`|transfer backends/events/ring; размер только slot_stride, но завершённые frontier/readers fences|finish/synchronize; безопаснее пересоздать event/ring owner с исходными depth/stream configs; не откатывать depth к2|
|`background`, `pending_prefetch`|private future содержит borrowed `Job.destination` из slot_proxies|collect/wait или drain, снять pending; удалить/recreate owner до metadata free|
|`cached_exact_worker`, `cached_batch_worker`|private stream, последний job хранит borrowed pool pointers|drain оба фактических worker; после drain GPU work нет; следующий begin заново получает new pools, buffers можно сохранить|
|`active_read_keys`|защита веса от eviction во время настоящего private reader|resize только между serialized calls; require empty + drain workers; одного ggml synchronize недостаточно|
|`Layer types/sources/offsets/raw/allocation/scales`|модель/slot_stride, scales экспертные, не слотовые|сохранить; geometry и stride не меняются|
|`ExpertArena`, borrowed/owned `registered_arena`|153GB host registration, не device slot count|сохранить целиком, не reset GpuExperts/arena/registration|
|`layer_banks`, `fast_ffn`, `cpu_quants`|независимые stream/metadata контракты|первый вариант запретить: resize только native LRU, banks0/exact0/fast0|

`weight_view()` задаёт ne[2]=slot_count, nb[2]=slot_stride, view_src=pool_proxy.
Просто заменить p_->pool без удаления graph views создаст stale pointers и
неправильные bounds/scales. `GpuDeviceDots` собственного постоянного weight
cache не имеет: веса передаются каждому `multiply`; last-job pool pointers
в helper безопасны только после завершённого drain и перед следующим begin.

## Предлагаемые API, пока без применения

Private `replace_slot_pool(target_slots)` — единственный путь перестройки
storage+metadata. Public boundary APIs разделены по режиму:

- `grow_decode_pool(model, trusted_receipt_path, trusted_receipt_sha256,
  reserve_bytes, future_workspace_bytes)` проверяет модель/receipt/rank и
  бюджет ДО teardown; native LRU only; возвращает фактические slots/bytes.
- `restore_prefill_pool()` возвращает сохранённый исходный pool290/nominal7GiB,
  убирает decode-only pins/order; холодный pool допустим. Не меняет target KV,
  router/MSA state или Engine position. Потерянные resident weights не повод
  пересчитывать токены.

План транзакции: проверить всё, что можно, и подготовить CPU metadata; запретить
новые consumers; завершить actual private workers, prefetch future и upload
streams; clear CUDA captures; уничтожить graph views и slot metadata; free old
pool; allocate target (без одновременного пика двух pools); bind new metadata;
пересоздать очищенные transfer/event owners; опубликовать cold cache; выполнить
новый seed; снять переходный флаг. При ошибке allocation/seed — не публиковать
частичные entries. По умолчанию terminal `ready=false`; если нужен graceful
fallback, отдельно доказать allocate-original290→cold metadata rollback, не
обещать сохранение прежних resident весов. OOM может требовать очистки backend
error; эта ветка нуждается в отдельном реальном тесте.

Кумулятивные counters сохраняются, добавить resize count/seconds/reseed bytes,
peak resident=max(old peak,new size); текущие resident/occupied/slot_count
должны соответствовать новому pool. Old `seed_hot()` не подходит: он требует
forwards==0 и cold init. Для grow нужен отдельный проверенный prefix path.

## Trusted receipt для grow

Текущий sealed receipt v1 связывает rank SHA/identity/current model headers+stat,
конкретные `slots` и `hot`. Для400/475 выписать НОВЫЕ decode-only preflight
seals existing `port7_decode_seed_preflight.py --slots400 --hot400` и475/475.
Старый290 seal переиспользовать нельзя. Rank identity/SHA сохраняются; только
явно проверенный prefix и бюджет меняются. Применение обычного seed остаётся
ограничено `slots-experts`, его правило не ослабляется.

Нельзя передать grow400 через существующий `M31_GPU_DECODE_SEED_HOT=400` на
Engine init с7GiB: init честно отклонит receipt.slots400 != actual290.
Нужен отдельный future-grow profile/receipt tuple или явные boundary API args;
не менять env посреди runtime и не выдавать290 seal за400. Initial ordinary и
deferred seeds можно оставить0 до grow. JSON `mode=decode-only` не записан
отдельным полем в seal v1: именно явный decode-grow API/flag задаёт режим,
а текущий v1 receipt проверяет slots/hot и идентичность модели/rank. При новом
универсальном sealed v2 разумно добавить enum purpose и initial/target pool,
но существующие immutable v1 seals не переписывать.

## Возврат к холодному префиллу и критерии проверки

Hook принадлежит Engine/DSpark owner: `begin_prefill()` уже выгружает drafter,
а session finish очищает его и external resident credit. После drain/release
перед любым `prefill_chunk[_at]` восстановить pool290. Только уменьшить pool,
оставив sidecar resident, недостаточно, чтобы гарантировать прежний v8 chunk.
Engine initial free snapshot получен при resident290; после восстановления
того же pool он снова применим, если прочие independent owners учтены или
освобождены. Пока pool grown, планировщик холодного PP не должен использовать
старый snapshot как будто290 всё ещё текущий pool.

Минимальный runtime proof до modelgate: synthetic290→400→seed400→N1/N8 cached
all-hit/partial→restore290→n>1 full128distinct prefill; повторить2цикла; callback
failure/drain; allocation-failure rollback или terminal poison; нет повторной
153GB host registration; размер pool/events/metadata подтверждён; old views
не вызываются. Затем same-ELF grow0/1 gate+greedy с реальным hot rank и замер
полного wall, включая каждый resize/seed. Отдельно32K/64K admission и
повторный cold om PP подтверждают сохранение v8 chunk/TG/качества.

Дополнительные детали реализации: `clear_captures(backend)` затрагивает общий
CUDA graph cache backend, включая возможные captures drafter/dense. Владельцы
их tensor buffers остаются живы; следующий compute может повторно захватывать
CUDA graph, это тоже wall overhead. `decode_scratch_bytes` после удаления
native cached graphs лучше пересчитать как сумму сохранённых exact workers,
а не оставлять прежний high-water allocator count. При отказе reserve-check
до free old pool можно просто продолжить прежний290 cache; после начала
teardown нужен доказанный rollback или terminal failure. Автоматически
урезать400 receipt до случайного меньшего slot_count нельзя: это другой
sealed slots/hot контракт.
