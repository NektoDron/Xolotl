# Port5: воспроизводимые окна скорости и ограниченный head

## Итог направления: все глубины измерены

Полное изолированное окно `m31-port5-gpu-v4-full-v1` завершилось rc0:
warmup и два запроса на каждой глубине, без OOM. TG медианы4K/16K/32K —
**27.51/26.385/25.44 ток/с**, ref23.33/22.63/21.92 (+17.9/+16.6/+16.1%).
PP медианы — **215.05/231.3/237.9 ток/с**; ref4K425/32K433.
Префилл остаётся примерно вдвое медленнее ref, цель кратного ускорения
не достигнута. Это частичный результат M2; M3 и GSM100 не запускались.

Frozen serve825ed0d7, native generatee1b9fb43; receipt
`build-port5-cache/candidate-v4/build-receipt.json`. F16/ctx40000/max8192,
TOPK_MASK1/querytile0/adaptive1, cache7/290slots/seed20/prefetch128,
PINNED1/privateStream1/PACK8/HOST32, CPU fused80/concat/chain/native gather,
CACHED_DECODE0. Planner уменьшал четвёртый32K batch до7680:
8192/8192/8192/7680/762 и .../763. GPU peak30653MiB, sumVmRSS320.355GiB.
Оба16K запроса использовали8192/8192/{258,254};4K — один4349/4348.

Полный итог направления и машинные данные:
`/mnt/ai/logs/m31-port/port5-gpu-v4-full-report-v1.md`, одноимённый `.json`,
компактный `port5-gpu-v4-full-summary-v1.json`. Там raw строки, ratios,
request profiles, actual chunks, память, flags/ELF, ограничения атрибуции.
Combined kernel_D2H wait не выдаётся за чистое время PCIe или kernels.
Unit/client/server rc0, normal EOF, aictl down/rollback/возврат майнера выполнены.

Координатор после окна запустил единственный финальный full20 на том же ELF
против существующего strict CPU oracle c2048/decode32/allrows. Его результат
фиксирует общий итог root. Новых speed/quality/GPU экспериментов после окна
направление bench не запускало. Ниже сохранена история решений и отбраковки.

Референс прежний: TG4K/16K/32K
23.33/22.63/21.92, PP4K/32K425/433, Q8_0/64K/ubatch2048/3GPUexpert layers.

`tools/m31/port5_bench_prepare.py` готовит новый каталог и временный
`m31-xolotl` card job; сам карту не занимает и конфиг не меняет. Копирует
проверенный ELF, Python adapter и session runner; сохраняет SHA256, source diff,
параметры KV/context/chunk/threads, явные env, client/corpus fingerprints.
Текущий workspace source может отличаться от prebuilt ELF: для связи с исходником
есть `--build-receipt`. Копирование замороженного ELF не является новой сборкой.

По умолчанию speed-only: неизменённый `/home/nekto/om_bench.py`, exact
`OM_OFF={"reasoning_effort":"low"}`, warmup и два4K/16K/32K. Нет повторного
109-check smoke или quality внутри окна. `--depths warmup,4K` готовит короткое
диагностическое плечо; readiness `/health` не является проверкой качества.
Запуск только после объявления координатором изолированного окна через
сохранённый `launch_argv`/`card_launch.py`, который держит card/build locks,
aictl backup/up/down/rollback. Текущий KV F16 отличается от reference Q8_0.

Performance baseline включает CPU/dense graph caches, persistent CPU pool,
CPU residual, host read reuse и shared overlap. GPU budget>0 дополнительно
включает exact CPU quants, pinned async/compact uploads, route sorting и
32-thread host quant/OAI executor; hot20 проходит свежий sealed rank preflight.
Эти параметры нельзя считать быстрыми до реального измерения конкретного ELF.
Любой параметр можно переопределить повторным `--env NAME=VALUE`.

`port5_bench_report.py DIRECTORY --out NEW.json` собирает исходные TSV,
медианы/отношения к ref, per-request native timings, prefill/decode stage sums
и means, peakGPU0 и сумму VmRSS потомков. Shared/GPU/CPU интервалы могут
перекрываться; их суммы не объявляются разложением wall time.
Проверка на сохранённом port4 CPU окне воспроизвела22.155TG/183.3PP,
неполные16K/32K отмечены явно.

## Завершение сервера и профилирование

Раньше session runner посылал SIGTERM всей группе процессов, а adapter
`RPC.close()` сразу terminate native engine. Его destructor не исполнялся,
итоговые cache counters терялись.

Теперь runner сначала сигналит только Python adapter; тот закрывает stdin
helper и ждёт нормального EOF/выхода. RPC fallback:15с→TERM/5с→KILL;
runner fallback:45с→groupTERM/10с→groupKILL. Inference/timing boundaries
не менялись. CPU protocol+real vocab22/22PASS, отдельный helper на EOF
вышелrc0 и напечатал destructor marker; повторныйclose прошёл.
Новые session results записывают server returncode и способ shutdown.

## Bounded correctness head для больших chunk

Новые opt-in `--head-tail64 --rows last:64` (имя и значение отдельными argv)
на native/oracle вычисляют head только над последнимиmin(64,n) строками
каждого chunk. Оба используют одинаковую contiguous ширину, поэтому не
сравнивают одиночный MMV с большим MMQ. `--decode-tail32` на3000tokens
даёт64проверяемые позиции:32prefill+32decode, без vocab×8192 output.
Default `--rows all`, head-tail0 сохраняет прежнюю математику/полный head.

Bounded mode требует tapsnone и rowslast/last:N. Dump отвергает запросы
строк вне вычисляемого хвоста до первого forward; manifest сохраняет
`head_tail`/`row_selection`, comparer отвергает разные headwidth и явно
считает такие проверки diagnostic. `--require-suite` требует head_tail0
и все строки каждого кейса. Comparator5/5PASS с отдельным mismatch test.

Oracle adapter relink из неизменённых audited PR6b95a990 archives:
`build-port5-bench-oracle/m31_oracle`, unit
`codex-m31-port5-bench-oracle-build-v2`rc0. Header/tool caps8192;
oracle math/backend/offload settings не менялись. Нативная проверка
нового bounded режима на GPU ещё предстоит; это не новый M1/gate20.

`codex-m31-port5-oracle-tail64-v1` завершёнrc0: strictCPU reference
code3000/d32 дляchunk128 и8192, обаhead64/last64. References:
`/mnt/ai/logs/m31-port/port5-oracle-tail64-v1/oracle-c128` и `oracle-c8192`.
Проверены64positions2936..2999, не все3000строк. Ни одного reference
не выдают за проверку соответствующего нового native пути без сравнения.

## Первое изолированное окно v3 — только диагностика

**После большого correctness gate этот кандидат отклонён для M2.**
Fusion ON на большом префилле дал одинаково неверные logits в CPU/GPU
expert пути; nofusion прошёл. Code128 не покрывал эту форму. Ни числа
v3 ниже, ни подготовленные CPUv4/GPUv1 fusion ON планы не являются
принятой скоростной конфигурацией. Их каталоги содержат `INVALIDATED.json`;
исходные timings, manifests и frozen ELF сохранены для диагностики.

`m31-port5-dense-v3-short-v1`rc0, warmup+один4K через неизменённый
om_bench/exacteffortlow. FrozenELF5120c544, CPUgraph64/F16/ctx8192/chunk2048,
chain1/asyncReads1/safeFusion1/fusion0, direct0/arena0/cache0,
hostreads1/shared1. Source receipt `port5-dense-source-v3.json`.
Параллельных сборок и тяжёлых CPU задач не было; coordinator объявил окно.

TG4K23.40 противref23.33 (+0.30%), PP4K182.6 против425; prompt4347/gen256.
Это один прогон и малый TG эффект, устойчивый выигрыш им не доказан.
Native16K/32K этим окном не измерены. PeakGPU013225MiB,
суммаVmRSS164.29GiB. Session/server/clientsrc0, helpers завершились нормально
черезEOF; aictldown/targetedrollback/возвратмайнера завершились.

На255decode шагах среднее main dense13.821ms, CPU MoE26.446ms,
sharedenqueue.519/wait.075/read.0018ms, head.714ms. В async режиме dense
submit.919ms/read12.571ms: read включает ожидание реальной GPU работы,
поэтому низкий submit нельзя выдавать за дешёвый полный dense.
Полный отчёт `/mnt/ai/logs/m31-port/port5-dense-v3-short-report-v1.json`.
Префилл остаётся следующим приоритетом; M2 не объявлен принятым.

Подготовлено следующее GPU4K/chunk8192 окно на согласованной сборке ELF
673ab07a: batched exact kernels, persistent GPU results, parallel reduce,
compact ring, host32 и sealed seed20/cache6GiB/arena170GiB.
`port5-gpu-c8192-short-v1` включает blind next prefetch1;
`port5-gpu-c8192-short-p0-v1` сохраняет prefetch0. Они ещё не измерены.
Агент cache проверил code128p0/p1 побитно; его diagnostic wall и upload bytes
показали, что слепая загрузка следующего слоя может ухудшать трафик/время.
Этот результат не является om_bench speed, поэтому выбор финального режима
остаётся за изолированным окном.

Ранее подготовленный fallback `port5-dense-v3-full-v1` также отменён после
fusion gate. Обязательные4K/16K/32K будут измеряться на исправленном пути.

## Bounded oracle: полная математика последнего слоя

У PR выбор64 выходных строк переносил crop перед FFN59. Это меняло
число назначений каждому эксперту и выбор CPU vecdot/tiled accumulation.
Native вычислял весь последний слой, затем выбирал64строки для head.
Ранние bounded references поэтому не были эквивалентны native по геометрии
последнего FFN, хотя укладывались в численные допуски.

Исправление использует существующий API audited PR:
`llama_set_embeddings_layer_inp(ctx, n_layer, true)`. Флаг `tap_last` сохраняет
весь последний FFN и переносит crop после него. Head остаётся64строки,
никакие веса или библиотеки PR не меняются. Новые manifests содержат
`full_last_layer_before_crop:true` и реальную последовательность `actual_chunks`.

`codex-m31-port5-oracle-full-last-tail64-v2`rc0. Новые references:
`/mnt/ai/logs/m31-port/port5-oracle-full-last-tail64-v2/oracle-c128` и
`oracle-c8192`; frozen oracle19d67898. Старые pruned references сохранены.
Existing nofusion CPU large и corrected-fence v5 large против новойc8192
reference совпали **побитно на64строках**, KL/maxabs0. Ранее наблюдавшиеся
maxabs.0025/KL2e-9 объяснялись crop, а не новой ошибкой native.

Comparator нормализует контракт: native всегда full; oracle head-tail0
full; legacy oracle head-tail>0 без флага cropped. Несовпадающие контракты
теперь отвергаются. Для bounded также проверяется реальная chunk-геометрия;
для legacy manifests она восстанавливается из chunk/decode_tail.
Девять comparator tests прошли, включая отказ для старого cropped oracle
против полного native. Full M1 по-прежнему требует все строки и head-tail0.

## Адаптивный runtime префилла

Frontend generate/serve перед каждым prefill block вызывает
`Engine::prefill_chunk(requested,error)`. Decode1 и decode-tail границы
сохраняются. Opt-in `M31_PREFILL_ADAPTIVE=1` использует planner dense/stream
для ограничения batch по VRAM и int32 MSA; default0 сохраняет заданный chunk.
Native/oracle dump фиксирует фактические чанки и проверяет полное покрытие
входа. Benchmark report сохраняет prefill chunks каждого запроса: requested8192
нельзя выдавать за фактический8192, если planner выбрал4096/2048.

Следующий CPU4K план — `port5-cpu-v5-short-v1`, frozen81b9342a,
corrected fusion fence, concat/fusedCPU, native gather и adaptive planner.
Он подготовлен и ещё не измерен; финальный выбор после его gates/окна.

## Принятое короткое окно GPUv2 → CPUv5

`m31-port5-gpu-v2-p1-cpu-v5-window-v1` завершёнrc0. Перед запуском
координатор закрыл все сборки/тяжёлые CPU задачи. На GPUv2 большие и короткие
gates прошли, CPUv5 large64 совпал побитно с новым full-last oracle.

Общие параметры decode: F16,64CPU threads, original quant weights,
corrected fusion fence/chain/concat/host reads/shared overlap,
CPU fused experts, native F16 gather. GPUv2 использует WMMA exact integer
kernels, reuse buffers/parallel reduce, compact async uploads, PACK4/HOST32,
ranking seed20/cache6GiB/arena170GiB и next-layer prefetch rank64.
`CACHED_DECODE=0`, min32: decode остаётся CPU fused. CPUv5 — arena/cache0.
Нагрузка: неизменённый om_bench, exact effortlow, warmup+один4K/256tokens.

| Принятый путь | TG4K | PP4K | Реальные prefill chunks | PeakGPU0 |
|---|---:|---:|---|---:|
| GPUv2P1 |26.09|214.3|4345|20263MiB|
| CPUv5 |26.15|185.1|2048/2048/250|12985MiB|
| llama.cpp ref |23.33|425|ubatch2048|30.37GiB ref snapshot|

TG выше ref на11.8/12.1%, но PP составляет50.4/43.6% ref. По одному
запросу на arm; это ещё не полные4K/16K/32K и не M2 acceptance.
GPUv2 sumRSS317.72GiB, CPUv5 sumRSS164.33GiB: per-process VmRSS с mappings,
не уникальная физическая память. GPU v2 actual prefill20.27с, CPUv523.48с.

Среднее по255 decode шагам, GPU/CPU arm: main dense12.429/12.389ms,
CPU fused MoE23.625/23.662ms, shared enqueue.576/.555ms,
wait.088/.089ms/read≈.002ms, head≈.72ms. Async dense read включает ожидание
GPU compute. Не складывать пересекающиеся интервалы как независимые затраты.

Оба helper завершились черезEOF сrc0; destructor cache/upload counters
сохранены. Одна aictl reservation держала оба arm последовательно, затем
down/targetedrollback/возвратмайнера завершились. Reports:
`port5-gpu-v2-short-p1-report-v1.json`, `port5-cpu-v5-short-report-v2.json`.
Telemetry ranges включают startup: у GPU arm долгий CPU arena init искажает
медиану GPU clocks; она не выдаётся за частоту GPU во время вычисления.

Подготовлен обязательный full fallback `port5-cpu-v5-full-v1`:
ctx40000/chunk2048, warmup+два4K/16K/32K. Он ещё не запускался.

## Full32K план v3 временно отклонён

`port5-gpu-v3-full-v1` и соответствующий prepared full20 plan не запускались.
V3 code128 прошёл побитно, но real3000/d32/head64/query512 дал top1.578125,
meanKL2.3565/maxabs19.75. Оба плана помечены `INVALIDATED.json`; идёт
локализация query tile/F16 mask. Принятые короткие v2/CPUv5 цифры сохраняются,
их нельзя переносить на v3. Full fallback CPUv5 остаётся подготовленным.

Full20 план использует существующий strict CPU oracle
`oracle-integrated-suite20-c2048-d32-v1`,20/all28723positions, head-tail0,
ctx4096/chunk2048/decode32. Подготовка читала только малые JSON manifests,
без повторного чтения/хеширования гигабайтов. Comparator full20 теперь
отвергает GPU-op-offload oracle и несовпадающие actual chunk boundaries до
чтения сырых logits. Одиннадцать tests прошли, включая оба negative cases.
Единственный финальный full20 будет запущен после выбора последней принятой
скоростной конфигурации и её полного4K/16K/32K окна.
