# Port7: routed CPU decode

Работа поверх принятого port6 V8; accepted ELF не меняется. Владелец этого
направления — routed_cpu. Цель — сократить около21,5ms routed FFN на токен.

## Исходные факты

EPYC9654:96 физических ядер/192 SMT, один NUMA node,12 CCD/L3 доменов,
384MiB совокупного L3. Remote NUMA placement на этой загрузке отсутствует.
V8 использует80 CPU threads и persistent ggml pool, `M31_CPU_FUSED=1`.
Default ggml poll=50 означает до6553600 spin iterations между заданиями;
CPU pool не обязан засыпать во время каждого GPU перехода.

`PrefillThpMapping` уже имеет opt-in `M31_ARENA_THP=1`,2MiB alignment,
MADV_HUGEPAGE и измерение AnonHugePages. Системный режим THP=madvise,
defrag=madvise. На старте port7 free Normal high-order blocks всего около2GiB,
а guard для152,5GiB арены требует около190,6GiB: безопасный режим откатится
к прежнему allocator. FORCE не включался: рискует длительным direct compaction.

## Новый opt-in CPU kernel

`M31_CPU_FUSED_PAIRED_QUANT=0` (default/absent) сохраняет старое тело V8.
`=1`: каждый worker считает gate/up одной строки и сразу swigluoai,
без глобального барьера между GEMV и активацией. CPU vec_dot traits,
scale multiplications, scalar expf и порядок взвешенной суммы сохранены.
`=2`: дополнительно квантует независимые блоки down-активации тем же CPU
from_float. Это применяется только при целой размерности и достаточном
числе блоков для всех workers; иначе используется безопасный путь `=1`.
Число глобальных барьеров соответственно5→4→3. Никакой GPU math нет.

## Проверка и замер

`port7_cpu_sweep.cpp` читает исходный GGUF один раз,5 представительных
слоёв3/10/52/58/59 со всеми128 экспертами. На каждом проходе меняет4 IDs;
рабочий набор больше L3, а не повторяет одни4 горячих эксперта. Весовая
полоса — точные bytes используемых gate/up/down делить на wall time FFN.
Сравнивает mode1/2 с V8 fused mode0 побитно на20 expert-наборах для
каждой конфигурации threads; это компонентная проверка, не gate20 модели.
Сборка `tools/m31/port7_cpu_build.sh` использует отдельный build-port7-cpu,
existing canonical CPU ggml archives; CUDA не вызывается.

Результатов и рекомендации по включению пока нет. Скорость должна
измеряться в exclusive CPU окне, отдельно от сборок/конвертаций/GPU модели.

## Первый компонентный sweep

`port7-cpu-sweep-v1/receipt.json`:rc0,5,94s,24 конфигурации,
480 bitcomparisons (320 для новых modes1/2,160 baseline selfchecks).
CPU80 unpinned:233,23/232,24/232,96GB/s для mode0/1/2; статистически
обоснованного выигрыша нет. CPU48/64/96 baseline:163,65/210,68/220,83GB/s.
Poll0 хуже:114,35GB/s. Строгая affinity80 также не дала выигрыша.
Первые modes в новом процессе имеют mmap minor-fault warm bias; готовится
R2 с явным проходом по всем128 экспертам до таймера и обратным порядком.
Новый CPU mode пока не рекомендован для итогового профиля.

## Exact CPU batched verification

`cpu_batched_verify.hpp`: отдельный small-batch компонент для layer-major
DSpark verifier. Никаких изменений n1 пути. Внешняя группировка по экспертам,
затем строка weight, внутри token assignments: bytes весов могут обслуживать
несколько токенов из CPU cache. Все quant/from_float/vec_dot(nrc1)/F32 scales/
scalar OAI и последовательная сумма rank сохраняют accepted CPU arithmetic.

`port7_cpu_batch_check.cpp` сравнивает на реальных GGUF слоях3/10/52/58/59:
K1/2/4/8, used1/2/4, same/partial/disjoint routes.180 batch checks,
675 token rows. Затем9 timing combinations K2/4/8 × routing pattern,
оба плеча с теми же ID, rotating experts и полным warmup до таймера.
Это компонентный тест CPU FFN; GPU dense scheduler и полная генерация
DSpark требуют отдельной численной проверки. Сборка и запуск pending.

Компонент exact batch прошёл: `port7-cpu-batch-check-v1/receipt.json`,rc0,
15,26s,180 batch/675token rows,9timing arms тоже bitwise. Headerd073421a…,
ELF68cc3425…. Same-expert K2/4/8 speedup1,134/1,181/1,182;
partial0,982/0,841/1,030; disjoint1,040/1,066/1,112. Это не full model TPS.
При K8 идеальный reuse bytes8× дал всего1,18×: CPU dequant/dot arithmetic
остаётся существенным. Такой результат не подменяется теорией bandwidth.

Готовится actual-route replay из8prompts×127autoregressive n1 positions,
с теми же5representative layers. Payload5080rows сохраняет реальные IDs
и F32 route weights; активации synthetic одинаковые для обоих плечей.
Source trace SHA905fa29f…, payload3d407ce7…. Замер будет содержать exact
serial/grouped times, assignment и unique weight bytes, union experts,
побитовое сравнение каждой пары — не только вычисленную GB/s.

Actual-route replay завершён: `port7-cpu-actual-check-v1/receipt.json`,rc0,
15,52s,14800token-row pairs bitwise. K2 serial2,123s/grouped2,397s (0,886×),
K4 2,011/2,782s (0,723×), K8 1,845/3,154s (0,585×). Logical requested/unique
weight bytes:489,309/445,246GB;481,542/353,445GB;466,008/251,801GB.
Unique expert groups:18321/14522/10327 соответственно. Это честный
отрицательный результат: reuse bytes не превратился в ускорение.

Вероятный источник — распределение равного числа expert rows по workers
при разном количестве token assignments в группе. Строка hot expert с8
назначениями стоит в8раз дороже singleton, а текущий range учитывал обе
одинаково. В uniform same-expert тесте этой неравномерности нет. Следующий
вариант — weighted row partition по assignments×rows с сохранением
единственного владельца каждой строки. Причина требует проверки новым
таймингом, не объявляется уже доказанной.

## Следующий рычаг: общий unpack для пары RHS

Vendor x86 IQ3_XXS/IQ4_XS/NVFP4 все имеют nrows=1 и assert(nrc==1).
Готового nrc2 нет. Дополнительный opt-in pair2 должен копировать исходные
integer pipelines и FMA/hsum8 отдельно для каждого RHS; singleton остаётся
на stock vec_dot. IQ3: на256weights block повторяются64gridlookups и32sign
lookups; share decode по паре особенно полезен для gate/up. IQ4: nibble LUT
и scales unpack один раз, два независимых dot/accum. NVFP4: общий nibble
decode/4UE4M3 scales на64weights, но собственные Q8_0 half-scales и тот же
порядок двух FMA каждого RHS. Не заменять hsum8 на иной горизонтальный sum.

По всем57слоям actual routes парами можно обработать21,13/42,39/59,09%
назначений приK2/4/8; число unpack calls уменьшается на10,56/21,19/29,55%.
Raw `port7-cpu-real-pair-potential-v1.json`. Это операционная оценка,
не измеренный прирост. Код копируется в отдельный helper с MIT license
и указанием vendor commit6050b467; live vendor не меняется.

Weighted partition реализован в live header3b123ac0… (ещё pending proof).
Границы выражены в assignments×rows; ceil на обеих соседних границах
даёт каждому полному weight row ровно одного владельца.700 ownership
проверок покрывают multiplicity1..8, rows1..6144, pools1..96 и bound
max−min weighted work≤2max_multiplicity+1. Обвязка
`port7_cpu_weighted_{build,speed}.sh` собирает/запускает partition,
180batch675token-row proof и все14800actual-route rows. Frozen old header
остался в build-port7-cpu-batch и build-port7-cpu-actual.

`cpu_batched_dot.hpp` подготовлен как отдельный НЕВКЛЮЧЁННЫЙ pair2 helper.
IQ4_XS и IQ3_XXS получают один unpack на два независимых RHS. SIMD float
accum/FMA/hsum8 исходные; FP16 преобразуется тем же ggml CPU LUT, не иным
intrinsic. Runtime CPUID AVX2/FMA, unsupported ISA/type возвращает fallback.
MIT license и source commit6050b467 записаны в файле; live vendor не менялся.
Row proof запланирован на196608full-row pairs и786432single-block pairs,
1966080отдельных dot results из реальных12weight tensors128experts16rows,
8разных RHS patterns/scales. До его PASS helper не включается в CPU batch.

Weighted fix прошёл: `port7-cpu-weighted-check-v1/receipt.json`,rc0,
13,80s.700ownership cases/10079540rows,180batch675token-row component
checks и все14800actual-route rows bitwise. Actual K2/4/8 serial/grouped:
2,174/2,030s (1,071×),2,146/1,906s (1,126×),1,987/1,689s (1,176×).
Сырые времена и те же byte/unique counts сохранены; старый отрицательный
прогон не удалялся. Header3b123ac0… проверен и передан DSpark владельцу.

Metadataaudit всего GGUF: gate/up57eachIQ3; down42IQ3/12IQ4/3NVFP4.
По raw weight bytes доли примерно88,1/9,4/2,5%. Наш5layer fixture намеренно
содержит все3NVFP4downs для numeric coverage, а не пропорциональный model
performance sample. Поэтому результаты компонента не выдаются за model TPS.
Raw `port7-cpu-weight-type-budget-v1.json`.

Pair2 raw proof PASS: `port7-cpu-pair-proof-v1/receipt.json`,rc0,0,571s.
1966080dot results,maxabs0:IQ3 1802240,IQ4 163840; full-row pairs196608,
selected256-block pairs786432.8RHS patterns,128experts16rows,96inputcases.
Helper e32db70e… после этого включён в новый live CPU batch header304c128b…
по прямому ACK координатора. `M31_CPU_BATCH_PAIR=0` default weightedstock,
`=1` IQ4pair, `=2` IQ4+IQ3pair. Singleton/unsupportedtype/ISA — stock dots.
FFN modelcomponent ещё должен пройти675rows и actual14800rows во всех3modes;
до этого mode2 не используется в модели. Сборка frozen pair-batch rc0,
ELFsff087c6f…/429de2f8…. Proof receipt явно записывает FP compile flags,
AVX2/FMA target attribute, CPUID guard/runtime ISA, source/ELF/input identity.

Pair-integration full component proof PASS:
`port7-cpu-pair-batch-check-v1/receipt.json`,rc0,41,65s. Во всех modes0/1/2
по675component+14800actual-route rows =46425FFN-row comparisons bitwise.
Normalized serial/grouped speedups K2/4/8:

| CPU batch mode | K2 | K4 | K8 |
|---|---:|---:|---:|
|0 weightedstock|1,072|1,111|1,190|
|1 IQ4 pair|1,058|1,108|1,193|
|2 IQ3+IQ4 pair|1,079|1,143|1,216|

Mode2 добавил около2% наK8 к weightedstock в одном replay; CPU clocks/
cache jitter не дают объявлять это устойчивым приростом модели. Mode1
не показал отдельного выигрыша. Assemblyаудит frozen binaries:
IQ3pair970B/0vectorstack refs; IQ4pair1415B/24vectorstack refs.
Raw `port7-cpu-pair-assembly-v1.json`. Loop-unroll/register-pressure
оптимизация IQ4 остаётся возможным следующим экспериментом, не применена.

Live headers304c128b…/e32db70e… теперь прошли raw dots и FFN component;
координатор/DSpark владелец уведомлены. Default pairmode0. Source ABI и
выходные token-major vectors прежние. Новые math-kernels не участвуют
в обычном v8 n1 CpuFusedDecode; никакой финальный model TPS/quality gate
этими компонентными доказательствами не подменён. Все сборки/проверки
были systemd User=nekto под согласованными shared/exclusive build locks;
GPU этот агент не занимал, vendor/system config не менял, коммитов не делал.

## Подготовленный batch thread sweep

`tools/m31/port7_cpu_batch_threads.sh NEW-LOG-DIR` подготовлен, но не запускался. Он использует неизменный `build-port7-cpu-pair-batch` ELF с уже проверенным unmasked body и mode2, реальные frozen маршруты, синтетические одинаковые входы, полное prefault весов каждой из пяти fixture layers перед каждым arm. Сравниваются T80/64/96 на K2/4/8, фактическая работа/уникальные байты проверяются на равенство; весь процесс имеет общий timeout54с и exclusive build-lock. Нового Engine/arena153ГиБ нет. Старый snapshot header304c явно зафиксирован; это компонентное измерение unmasked CPUbatch, не скорость live GPU hybrid или модели. Результат n1best80 не переносится на batch. Запуск только после окна validation; baseline и candidate должны получить одинаковый idle CPU.

## Batch thread sweep: actual routes, mode2 (выполнен)

2026-10-27, чистое exclusive CPU окно validation, systemd `codex-m31-port7-cpu-batch-threads-v1`,46.483с. T80→64→96, одинаковые маршруты/веса/F32 входы, полный prefault каждой fixture layer перед arm. Все три K и все T прошли побитовое сравнение с независимыми per-token fused CPU вызовами:44 400 token-layer output rows,272 793 600 F32 values. Это пять выбранных слоёв с реальными IQ3/IQ4/NVFP4, actual greedy routes и синтетическими activations; не model TPS/не DRAM hardware counters. Рабочие окна и логические байты проверены на равенство между T.

| K | token-layer rows | unique experts | serial logical bytes | grouped unique logical bytes | T80 grouped s | T64 grouped s | T96 grouped s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 5040 | 18321 | 489308553216 | 445245603840 | 2.147044 | 2.296870 | 2.859534 |
| 4 | 4960 | 14522 | 481541750784 | 353444954112 | 1.979278 | 2.128857 | 2.567965 |
| 8 | 4800 | 10327 | 466008145920 | 251801468928 | 1.654601 | 1.858070 | 2.294449 |

T80 быстрее T64 на7.0/7.6/12.3% по времени K2/4/8; T96 медленнее T80 на33.2/29.7/38.7%. Поэтому рекомендуем80 threads для текущей CPUbatch M3 нагрузки. Это один чистый sweep с фиксированным порядком, без доверительного интервала; различие с96 велико, а финальный model TPS всё равно измеряется отдельно. При T80 grouped/fused serial speedups1.091/1.190/1.266×. Logical unique-bytes/grouped-time≈207.4/178.6/152.2ГБ/с: снижение здесь ожидаемо при амортизации весов и сохранении compute, и не является измеренной пропускной способностью DRAM.

Receipt: `/mnt/ai/logs/m31-port/port7-cpu-batch-threads-v1/receipt.json`. Прежний раздел «подготовленный sweep» сохраняет историю подготовки; приведённый здесь результат его завершает.
