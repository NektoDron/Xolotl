# Port5: компактная загрузка и большие слоевые батчи

Текущее принятое направление: **whole-query TOP_K_MASK + in-place score ADD**,
компактные role uploads через pinned ring2, отдельный persistent packing pool,
next-layer prefetch и adaptive chunk из основного generate/serve пути. V4 прошёл
code128 и selected large3000 strict-oracle gates побитно. В реальном om_bench
32K при context40000 allocator выдержал **3×8192+7680+762/763 prompt rows
без OOM**, GPU peak30653MiB. Planner действительно уменьшил четвёртый chunk;
это runtime подтверждение адаптации, не оценочный proposal и не четыре полных8K.

**Query tiling не принято:** query512 провалило numeric gate; canonical engine
явно запрещает `M31_MSA_QUERY_TILE>0` и направляет к `M31_MSA_TOPK_MASK=1`.
Frozen v4 benchmark ELF может предшествовать этому guard, его env query0.

Фактическое H2D/expert-kernel overlap в V4 —176.816ms по CUPTI,627large copies,
339kernels с суммарным CUPTI device time3.620753с, пересечение4.8834%.
Это часть работы, не измеренный speedup. Два полных круга om_bench4K/16K/32K
закончены: TG27.51/26.385/25.44t/s, PP215.05/231.3/237.9t/s. Генерация выше
reference примерно на16–18%, префилл ниже llama.cpp425/433; цель кратного
ускорения PP не достигнута. M2 частичный, M3 не запускался. Итоговый full20
`port5-gpu-v4-final-gate20-v1` завершён rc0:20cases,28723полных rows,
top1=1,mean/maxKL=0, scope `M1-complete-logits`. Отчёт:
`/mnt/ai/logs/m31-port/port5-gpu-v4-final-gate20-v1/compare.json`.

Ниже сохранён ход работы и область доказательств каждой промежуточной версии.

27.10.2026. `PrefillUploads::enqueue_spans` загружает только три backend-readable
role extent, исходные bytes + нулевой хвост до backend allocation size. Промежутки
между ролями и trailing alignment fixed slot через PCIe не передаются. Порядок,
формы `[used,n]`, веса и router сохраняются. Источник можно переиспользовать сразу
после enqueue: им владеет прежнее pinned ring2 до completion event. begin/finish
защищают VRAM-слоты от перезаписи под потребителями предыдущего батча.

Интеграцию `M31_GPU_PREFILL_COMPACT=1` выполнил gpu_cache. Прежний `enqueue`
не меняет семантику: полный slot с нулевыми padding bytes. В compact path все
читаемые CUDA ядром padding bytes по-прежнему нулевые, включая повторное
использование слота слоями с разной geometry. Необъявленные gaps не являются
весами, consumers не должны их читать.

Для реального GGUF сумма полного routed stream: 152.508 GiB raw против
176.054 GiB fixed slots. Теоретическое сокращение transfer bytes —13.37%
(не измеренный прирост скорости). Поэтому основной шаг — amortization stream:
CLI и GpuExperts допускают max chunk8192 вместо2048. Один запрос около4K
помещается в один layer batch, вместо двух-трёх полных загрузок всех экспертов.
CUDA exact tile поддерживает n8192. Planner теперь допускает8192, сохраняет
pad256/block alignment и проверяет MSA int32 indexing bound.

Большой chunk не означает гарантированное размещение: MSA scratch растёт с
`n*nk*index_heads`; при32K context безопаснее начинать с4096, проверяя allocator.
Full8192 vocabulary logits требуют ещё около6.5GiB, поэтому correctness helper
bench_integration добавил одинаковый bounded head-tail для native/oracle.
В HTTP сервере нужен только последний logit row. Изменение batch width может
менять dispatch/FP rounding, поэтому нужен same-chunk oracle gate.

## Проверки

- `codex-m31-port5-prefill-build-v1`:rc0. Link-only1C++TU, существующие ggml libs,
  gcc14/CUDA13.1, свой `build-port5-prefill`. CPU compact byte/event probe:
  51 slots/208896 output bytes побитно, varying role extents и immediate source
  reuse; untouched gaps остаются прежними, backend padding обнуляется.
- `codex-m31-port5-prefill-events-v1`:rc0, GPU0 через card_launch/aictl/flock.
  Те же51slots побитно, pinned=true/asynchronous=true,49ring reuses. Загрузка
  demonstration35043bytes против208896slotbytes; это synthetic layout, его
  проценты не переносятся на реальный GGUF. down/rollback выполнены.
- `codex-m31-port5-prefill-build-v2`:rc0, CPU probe повторён;10 planner checks,
  включая8192 proposal, long-context shrink и отказ8193. GPUv1 содержал прежние
  7 planner checks, CUDA byte API тот же.
- `codex-m31-port5-prefill-build-v3`:rc0. Host-wall pack/staging-wait/submit
  counters проверены с profile0/1: при0 все нулевые, при1 pack/submit положительны.
  Clock читается только при `M31_GPU_EXPERT_PROFILE=1`; DMA device time не измеряется.
- `codex-m31-port5-prefill-build-v4`:rc0. CPU byte/planner проверки повторены;
  checkpoint51 disjoint groups даёт прежние побитные outputs. Новый background
  prefetch:34slots/17batches через private persistent backend+ring, одновременно
  main backend читает другой slot. Partial enqueue failure корректно drain/join,
  последующий start восстанавливает ring. GPU background proof ещё не выполнялся.

Это byte/lifetime/planner приёмка; full-model speed/code128/gate20 находятся у
координатора и владельцев интеграции. Никаких M2 speedup чисел здесь не заявлено.

## Реальное overlap upload и MoE compute

`PrefillUploads::checkpoint()` публикует готовые uploads и сохраняет открытый
batch: compute текущей группы может идти одновременно с upload следующей в
другие slots. Возвращаемый event borrowed; CUDA bridge должен enqueue wait до
повторной record event в следующем checkpoint/finish.

`prefill_background.hpp` добавляет `BackgroundPrefillUploads`: собственный
backend и pinned ring2 создаются при первом worker и сохраняются между слоями.
Descriptor job содержит destination и3role spans. `start(jobs,compact)` запускает
один finite worker, `wait()` возвращает дельты counters и join; failure сначала
дренирует DMA и затем пробрасывается. Destructor всегда join до освобождения
ring/backend. Source arena, pool и metadata обязаны пережить worker.

С gpu_cache согласована интеграция next-layer prefetch без изменения математики:
защищаем128slots текущего слоя и seed pins, до120 остальных slots заранее отдаем
следующему слою, грузим на worker во время compute текущего. Main публикует
cache entries только после successful wait, остальные нужные эксперты грузятся
в обычном load следующего слоя. На n~2048+ почти все128expert активны; это не
меняет routes и не требует subset arithmetic. Измеренный speed/overlap proof
этой интеграции пока не получен.

Отдельный возможный вариант требует disjoint slot groups и subset
worklists при неизменном полном histogram. Нельзя просто заменить исходное
`n/used` маленьким батчем: это меняет CPU-compatible tiled threshold>=8.

Следующий прямой путь: разбить distinct experts на группы, заранее квантовать
полный layer input один раз, сохранить full histogram, upload следующей группы
в независимые slots на transfer stream и compute предыдущей на compute stream.
Gate/up/down scatter должен оставлять исходные route позиции; суммирование
только после всех групп в прежнем top-k порядке. Exact kernel API сейчас
возвращает host vectors и синхронизирует каждый multiply, поэтому отдельный
новый ring метод без изменения compute API такого overlap не создаёт.

Event proof подготовлен в `port5_prefill_overlap_events.py`: читает Nsight SQLite
readonly, использует H2D enum из самой базы, ищет большие H2D и exact expert
kernels на разных streams одного GPU/context. Сохраняет конкретные интервалы,
stream IDs и union пересечения. Это доказательство device copy/compute overlap,
не speed benchmark. На момент написания real-model trace ещё не выполнялся.

Независимый source review next-layer cache интеграции: текущие selected keys и
seed pins защищены; зарезервированный slot исключается из free/LRU сразу;
все cache entries публикуются только после successful background wait; failure
перед metadata reset сначала join/drain; destructor уничтожает background до
pool/arena metadata. Read-only arena sources и main compute destinations не
пересекаются с prefetch slots. Blockers не найдены. Отдельный review dense chain
нашёл async readback exception lifetime hazard; владелец добавил DenseDrain.

## Поздняя интеграция: параллельный CPU pack и runtime chunk

Live code128 profile обнаружил основной bottleneck:7.029с CPU packing на115.29GB
(16.4GB/s), при exact CUDA kernels около0.441с. Поэтому ring получил optional
synchronous `Copier(const CopySpan*,size_t)`: один callback всех3roles, bounded
fixed descriptor array, все CPU writes join до DMA. `CpuPackSpans` от cpu_executor
использует отдельный persistent4/8-worker pool и1MiB stripes; математика не
меняется. Background ring получает тот же copier. Main pack и background pack
в current plan не выполняются одновременно; quant/OAI executor отдельный.
CpuPackSpans25MB3roles/unalignedguards/reuse/empty/overlap CPU gate принят1/4/8
потоков. Измерение реальной pinned copy bandwidth ещё требуется.

Для32K principal path согласована адаптация: Engine выбирает admissible chunk
через `plan_prefill`, frontend использует результат перед forward и записывает
actual chunks. Бюджет — snapshot free VRAM после weights/KV/pool init; retained
scratch не вычитается дважды, дополнительный reserve покрывает late allocations.
Большие MSA matrices ограничивают chunk по мере роста nk. Владельцы engine/frontend
заканчивают реализацию; этот абзац фиксирует согласованный контракт, не pass M2.

Query-tiling planner source готов: `attention_query_tile=0` сохраняет прежнюю
формулу; opt-in использует tiled score/mask/pooled buffers, full F32 causal
`[nk,n]`, full projections/residual/concat и full MoE union. GPU graph stages
оцениваются по максимуму attention/FFN/routed; CPU host activation vectors в
VRAM не добавляются. `f16_mask` учитывает native F16 gather/add. Initial snapshot
не вычитает retained gallocr scratch повторно. Это envelope, не измеренный peak.
Engine/frontend callers реализованы владельцами, allocator gate обязателен.
Новые CPU планировочные cases подготовлены, сборка отложена до закрытия speed
window. Независимый review query views/left-deep concat/global causal positions
и CUDA private-stream lifetime не нашёл blockers; query-tail1 может выбрать
другой FA kernel, требуется выборочная FP-проверка.

## Реальный GPU overlap: v3 trace, numeric gate НЕ принят

`codex-m31-port5-prefill-events-export-v1`:rc0. Изготовленный ранее large3000
correctness `.nsys-rep` экспортирован в SQLite CPU-only, нового GPU запуска нет.
Независимый `port5_prefill_overlap_events.py` требует одинаковые process/device/
context, разные streams, exact expert kernel и H2D>=1MiB. Из339expert kernels
и21126largeH2D61copy реально пересеклась с kernels, union18.701ms. Примеры:
background H2D stream18 на7.2/10.0MB и private exact kernel stream14, GPU0,
context1. Данные: `port5-prefill-overlap-independent-v1.json`.

False-positive ограничения: main routed uploads завершены до execute; quant H2D
и expert kernels на одном private stream и исключаются same-stream фильтром;
decode hidden inputs меньше1MiB; trace содержит один isolated native процесс.
По порядку engine данные пересечения относятся next-layer prefetch. Это реальное,
но пока малое device overlap, не подтверждение speedup. Вся v3 math configuration
провалила large-logit gate; её M2 корректность или качество этим НЕ приняты.

CUPTI runtime/device durations (`port5-prefill-runtime-api-analysis-v1.json`):
private H2D4.432GiB API3.185с/device0.591с, private D2H30.977GiB API5.052с/
device1.482с; background H2D74.795GiB API0.061с/device3.062с. Pageable Async
может блокировать host до окончания kernels/copy; точная причина driver stalls
не доказана. Следующий bounded pinned-buffer вариант готовит exact_kernel.

Standalone fusion diagnostic в shared CPU-v5 session: original Q6_K block0
`gate/up+OAI` с terminal GLU дал побитные38Mvalues normal/fenced наn128/2968.
То есть standalone shader bug НЕ воспроизведён. Norm+MUL побитен, тройной
norm+MUL+RoPE отличается до~1e-6 из-за другой reduction/FP формы. Model-context
output fencing принят отдельно; точная причина большой fusion регрессии остаётся
открытой. Подготовленный отдельный Nsight tiny job не запускался.

## Whole-query TOP_K_MASK: source и planner CPU gate

Query512 дала large-logit регрессию:CPU query512/gather0 с fusion0/1 имеет
одинаковое расхождение(.578top1/KL2.357), query0/gather0 и query0/gather1
побитны свежему full-last-layer oracle. Поэтому query512 не включается в
принятый путь и не используется для финального speed без исправления.

Новый альтернативный путь dense/exact владельцев использует уже встроенный
`GGML_OP_TOP_K_MASK`: whole-query GEMM/FA geometry сохраняется; index score ADD
выполняется in-place, затем формируется сразу конечный F16 mask из causal,
TOP_K indices и block map. Planner fields `top_k_mask`/`inplace_score` default0,
whole8K score stage4GiB, конечный mask2GiB (с temporary F16 causal~0.5GiB),
эти стадии используют память по очереди. Плюс full causal и projection storage,
резервы late routed scratch. Максимум стадий не суммируется.

`codex-m31-port5-prefill-build-v6`:rc0.17planner checks, включая actual M3.1
4head geometry,8192rows/32768keys/context40000,10GiBfree и2GiBreserve;
whole8192 proposal принят, отсутствие in-place снижает chunk. Повторены51byte
outputs/checkpoints и34background slots с partial failure/recovery. Это CPU
planning gate, реальный GPU allocator/math/full4K16K32K остаётся у координатора.
Независимый read-only source review прямого mask path не нашёл blockers:
permuted TOP_K strides, global causal broadcast, exact block map, future -Inf
и lifetime score→pool→mask соответствуют старому semantic output.

## V4: лёгкие gates, затем полный benchmark и full20

`codex-m31-port5-gpu-candidate-v4-gates`:rc0. TOP_K_MASK/score-inplace/query0
настоящий code128 прошёл128rows, large3000/head-tail64 —64rows к свежему strict
CPU oracle побитно. Позднее итоговый full20 завершён rc0:20cases/28723rows,
top1=1,mean/maxKL=0 (`M1-complete-logits`); итоговые benchmark числа приведены
в начале документа. Runtime planner реально
вызывается из generate/serve перед forward; JSON записывает actual chunks.
Default query tile0 сохраняет полную форму index GEMM/FA, MoE histogram полный.
Query512 остаётся **неподдержанным экспериментом с известной numeric регрессией**,
в принятую конфигурацию не входит.

PINNED/private stream/cache7/PACK8/prefetch128 candidate имеет независимый
CUPTI proof:339expert kernels,22641H2D>=1MiB,627H2D пересеклись с expert kernels
на разных streams одного process/GPU/context. Union176.816ms:
`port5-prefill-overlap-independent-v4.json`, совпадает с независимым анализатором
владельца cache (`port5-gpu-v4-prefetch-event-proof-v1.json`). Background stream18
передал160175554560bytes; kernel stream14. После speed window из готового SQLite
посчитана точная сумма339CUPTI expert intervals:3620752809ns, она совпала с их
union (один stream). Пересечение176815570ns составляет4.883392% активного
expert-kernel device time. Итоговый JSON с denominator:
`port5-prefill-overlap-independent-v4-final.json`. Это доказательство механизма, не speedup и
не полное перекрытие. Наблюдавшийся рост18.7→176.8ms не приписывается одному
PINNED: между v3 и v4 изменились несколько настроек и v3 numeric gate не принят.

Ложноположительные совпадения ограничены: обычный main routed upload закончен
до execute; private activation H2D находится на том же stream14 и исключён;
малые decode hidden transfers меньше1MiB; trace содержит isolated native процесс.
Полные om_bench4K/16K/32K координатор выполнил отдельно без Nsight, итоговый
full20 gate также пройден. Итог M2 частичный: генерация быстрее reference,
цель ускорения префилла не достигнута. Числа component/profile не заменяют
эти измерения.
