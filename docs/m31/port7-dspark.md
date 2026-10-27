# Port7: DSpark, рабочий журнал

Итог исследования: точный opt-in DSpark реализован и проверен, но ускорение
не получено. На actual HTTP depth1 дал24,54/25,76/23,16т/с на4K/16K/32K;
depth3 медленнее. Принятые port6 v8 ELF/профиль не менялись. Экспериментальный
worker подключает M3 только с `--dspark-model`; по умолчанию sidecar не
загружается. Цель35–40т/с не достигнута; M3 по умолчанию не включать.

Ниже сохранён хронологический журнал. Фразы «pending» в ранних разделах
относятся к тому этапу; актуальный итог и ограничения находятся в конце.

## Архитектура

Реальный sidecar `dspark-MiniMax-M3.1-Flash-Q8_0.gguf`: DFlash, пять плотных
блоков; embedding6144, FFN12288, Q64/KV4/head128, partial NeoX RoPE64,
base10000000; окно4096; block7; Markov rank256; confidence head F32;
noise token200058; токенов200064. Веса исходные Q8_0, без переквантования.

GGUF `dflash.target_layers=[4,18,32,46,60]` — входы указанных слоёв, то есть
residual выходов нумерованных с нуля блоков `[3,17,31,45,59]` HF. Это
проверено по converter `conversion/minimax.py` → Qwen DFlash converter,
`src/models/minimax-m3.cpp` (`t_layer_inp`) и `common/speculative.cpp` PR31192.
Последний отвод берётся до `output_norm`, остальные — после routed FFN,
shared и residual additions; никакой подмены нормализованным hidden нет.

`m31_dspark.{hpp,cpp}`: отдельная ggml реализация encoder fc+norm, инъекции
K/V из подтверждённых target features, пяти noncausal блоков и Markov /
confidence heads. Заимствует неизменённый GPU output.weight target;
embedding читает из исходного target mmap. Noise K/V записываются в отдельные
scratch slots, подтверждённое окно4096 не перезаписывают. Параметры depth1..7
и p_min0..1 отсекают только предлагаемый префикс, не target распределение.

В Engine добавлен opt-in capture пяти отводов, persistent residual-only
capture graphs для decode (не transient cache clear на каждом токене),
getter features, заимствование output weight. По умолчанию список отводов
пуст, арифметика accepted path не добавляет capture kernels. Отдельный
`release_prefill_workspace` освобождает удерживаемую scratch память после
сохранения logits/features: draft загружается после prefill. Это пока
research transition, не серверный lifecycle для повторных запросов.

`m31_dspark_generate.cpp`: serial, parity, fixture и spec режимы. Parity
сначала генерирует greedy serial с сохранением каждого logits row, rewind
к prompt boundary, затем teacher-forced batches2/4/8; сравнивает top1,
maxabs, mean/max KL. Spec mode проверяет `[anchor,draft...]` target пачкой,
принимает только совпавший префикс, rewind отбрасывает хвост MSA/KV, в draft
ring добавляются только подтверждённые feature rows. Capture rows после
rewind недействительны. Выходные `.i32` позволяют независимое сравнение с
обычным greedy.

## Не закрытый риск

Пачка n>1 даже при CPU routed FFN не гарантирует совпадение с n1: MSA decode
собирает KV в score-sorted порядке, prefill использует physical-order маску;
projection/head MMV/MMQ и CPU grouped matmul тоже могут отличаться.
Поэтому качество draft и его confidence не заменяют обязательную проверку
batch-vs-serial на реальной модели. Если она расходится, потребуются n1 dense
subgraphs на каждый row и layer-major FFN batch с точной n1 арифметикой.

## Уже выполнено

`codex-m31-port7-dspark-syntax-v1`, rc0: CPU и CUDA syntax новых library и
driver. Source snapshot: `build-port7-dspark/syntax-v1`; лог:
`/mnt/ai/logs/m31-port/port7-dspark-syntax-v1.log`. Это не link, runtime,
математическая приёмка либо измерение скорости.

## Первые реальные результаты

Наивный n>1 verifier **отклонён**. Frozen combined V1, prefix3324 из
принятого long corpus и128 greedy rows:13/128 top1 отличаются от serial,
maxabs18.7452, meanKL0.238503, maxKL14.3565. Первый batch8 пересекал PAD256
границу3328. CPU routed threshold32 сохранял FFN наCPU, но это не устранило
различия dense/MSA/head. Диагностический unit завершён rc2, все результаты
сохранены в `port7-dspark-naive-parity-numeric-v2/native/summary.jsonl`.
Замер допускал concurrent сборки: его t/s не являются скоростной приёмкой.

Новый `m31_dspark_verify.inc`/`Engine::verify` строит layer-major проход:
каждому query оставлены n1 projection/shared/router/head matmuls, исходный
MSA sorted gather и отдельный per-query nk PAD256; только CPU routed FFN
читает общие expert rows для нескольких токенов. Его компонент
`CpuBatchedVerify` прошёл180 batches/675 token rows побитно на пяти реальных
IQ3/IQ4/NVFP4 слоях (K1/2/4/8, разная степень пересечения экспертов).
Полный модельный parity этого scheduler ещё в очереди; компонент не заменяет
его. Frozen V2 включает exact-shape driver и cached verify logits.

CPU/CUDA syntax нового scheduler/library/driver/probe прошёл rc0 в
`codex-m31-port7-dspark-exact-syntax-v1`. Getter обычных logits после verify
читает сохранённые n1 head rows; forward/reset/rewind/release их инвалидируют.
Live corner fix ограничивает capture max_rows; default4096 приK<=8 не менялся.

## End-to-end greedy и память

Actual DSpark V2 K7/p_min0, ctx40000, prefix одним batch3324:36 rounds,
92/246 draft tokens приняты (37.3984%),128 итоговых tokens. Первая raw
сверка с serial ctx4096/chunk2048 была **невалидной**: prefix раскладывался
2048+1276 вместо3324. Её117 различий не являются доказательством ошибки
rollback. Новый serial на том же frozenV2, ctx40000/maxchunk32768 и реальном
chunk[[0,3324,3328]] дал **128/128 одинаковых raw tokens**,
SHA414644292d608e255614fa935779e21beded07fb3e6f63db587a9383db452f3f.
Независимый strict comparator проверил prompt, actual chunks/nk, F16 KV,
все48 ENV/unsets, ELF и source receipt до scoring. Старый reference
отвергается до scoring статусом invalid_reference.

V3: weighted CPUbatch + opt-in persistent verify graphs. Те же teacher-forced
128 rows дали top1=100%, maxabs0; actual spec raw128 тоже совпали с serial.
В этом диагностическом прогоне spec вырос7.30→13.10т/с, ниже v8; это
сохранённый отрицательный baseline. Steady verify8: dense~99ms, CPU~145ms.
Профиль V3: fusion0/pair0/persistent1, cache7GiB, ctx40000. Замеры не om.

Точный API memory receipt V3 после освобождения prefill scratch:
free8733130752B при сохранённом expert pool7GiB. Sidecar занимает
2081138692B весов и44564480B ring; после load free6604521472B.
Сумма retained verify allocators приK8 всего345137664B (внутриграфовый
allocator переиспользует временные tensors между queries). Освобождать
expert pool / регистрацию152GiB арены не требовалось. NVML100ms:
peak28299427840B, minfree5891489792B, ошибок наблюдателя0; короткие пики
могут отсутствовать в samples. Exact API memory и NVML имеют разный scope.

## Следующий кандидат, пока без модельной приёмки

GPU component dense агента: forced MMV K4 на реальных F32/Q6_K/Q8_0
проекциях дал56/56 побитных случаев (2.79M F32). StockF32 K4 переключается
на MMF и отличается; поэтому нужен **per-node marker**, а не глобальное
изменение MSA score matmul с четырьмя RHS. K8 разбивается на4+4.

Новый opt-in `M31_DSPARK_BATCH_DENSE=1` группирует только QKV/index,
attention output, FFN gate/up/down, router и head. Norm/GLU/topk/MSA/FA
остаются n1 на каждом query. Default attention hooks=nullptr сохраняют
обычный путь. Cached агент проверил формы/offset/lifetimes без найденных
ошибок; CPU/CUDA syntax с реальным markerAPI header прошёл rc0. Компонент
не заменяет pending model parity.

Driver sweep готов: `--sweep 1:0,2:0,3:0,4:0,1:.5,...`; один target/draft
load, reset KV/ring между arms, отдельные raw outputs и structured
prefill receipts. DSpark owned resident bytes (weights/ring/graph scratch)
вычитаются из initial prefill budget при следующих arms; retained private
verify buffers учитываются суммой, ordinary forward освобождает их до
новой allocation. Нужна модельная сверка каждого принятого профиля.

V4 marked-dense модельный proof прошёл:128/128 top1, maxabs0. На том же
frozen ELF выполнен один процесс / один draft load для20 arms:
depth1/2/3/4/7 × p_min0/.5/.8/.95. Все20 raw128 результатов равны ordinary
serial, все actual prefix recipes=[[0,3324,3328]]. Лучший диагностический
результат23.10т/с (K1,p_min0), ниже serial26.53; K3/p0=19.52,K7/p0=14.14.
Это не итоговая скорость и не повод включать M3 по умолчанию. Отрицательные
arms сохранены целиком. Высокий p_min повышает acceptance, но меняет K и
провоцирует rebuild единственного persistent graph geometry.

Следующая отдельная оптимизация `M31_DSPARK_SHARED_INPUTS=1`: per-query
positions/mask/bias являются state-owned и заливаются один раз на verify,
все60 graph берут leaf aliases; hidden[D,K] одним H2D на слой и head.
В прежнем scheduler одинаковые masks/positions/bias заливались60 раз —
1440 избыточных H2D вызовов наK8. Owners освобождаются после всех graphs и
captures, shared buffers учитываются один раз. CPU/CUDA syntax с hardened
MMV API прошёл; независимые PCIe/cache и CPU reviewers не нашли owner/race
ошибок. Модельный proof этой новой оптимизации ещё нужен.

HTTP controller вынесен в `m31_dspark_session.hpp`: очередь содержит raw
verified rows и предложенные input IDs. Worker сохраняет обычный порядок
sample→emit→ACK→advance, не рисует sampler state наперёд. При совпадении
очередного выбранного token с cached input выдаётся соответствующий raw row;
при расхождении подтверждённые features добавляются в draft ring, target
откатывается к logical fed prefix и строится новая пачка. Finish обрезает
speculative tail к реальному числу fed tokens, освобождает private scratch и
выгружает draft перед холодным prefill. Draft init ленивый после первого
ACK: max_tokens0/1 и immediateEOS его не загружают.

Независимый причинный CPU fixture проверил2430 cases/13932 generated rows:
полное/нулевое/частичное принятие, suppression, confidence cut, cancel/max,
4096-window seed, ветку после stale-tail rollback и единственную ленивую
загрузку. Реальная Engine/Draft template instantiation прошла CPU/CUDA
syntax. Это проверка controller, не GPU/model/HTTP приёмка. Worker shim
готовит отдельный агент; stochastic/grammar остаются ordinary target fallback.

V5 shared-inputs модельный proof:128/128 top1,maxabs0. Следующий sweep
20×128 также завершился rc0, все raw outputs имеют SHA414644… и recipe
[[0,3324,3328]]. Лучший K1/p0=23.588т/с, K3/p0=19.890,K7/p0=14.304;
ordinary serialV5=26.889. Shared-copyfix уменьшил steady denseK8 ~89→80ms,
но M3 ещё медленнее ordinary. Пример steady phases: fill0.15ms,
upload1.23,compute69.7,read9.2,build0,reconfigure0. Профиль, одновременные
нагрузки и диагностический driver не позволяют считать эти числа om_bench.

V6 pending source содержит два независимых opt-in:
`M31_DSPARK_BATCH_OPS=1` группирует RMS/MUL,Q/K/index norm+RoPE, packedGLU,
router и headRMS в groups≤4; positionsI32[K] заливаются один раз на round.
MSA score/FA сохраняют n1 ordering. Actual production-helper component
прошёл444/444 cases (412helper-cases/2,913,600 values +32 отдельно
помеченных rawstockGLU), K1..4/позиции17,254,32766,65534,maxabs0.
Это компонентный proof при fusionOFF, не модельный gate V6.

`M31_DSPARK_GPU_CACHED=1` подключает exact GPU cached batch wrapper поверх
maskedCPU: original rank weights/sum сохраняются, allmiss идёт через прежний
CPUforward, allhit не запускает CPUgraph. Время cpu_ms охватывает seed,
submit,CPUmisses,GPUjoin иmerge; hybrid_ms/cpu_ffn_ms показывают части.
Wrapper actualGPU component:16cases K1/2/4/8,11,520 outputF32 bitwise,
184hits/368routes,all/partial/miss и восстановление после callbackexception.
CPU masked proof и GPUjoint proof выполнены владельцами отдельно. Flags0
сохраняют прежние операции. Isolated V6 syntax snapshot подготовлен; его
unit ожидает shared buildlock за чужим exclusive TTFT job. Модельные
parity/speed/HTTP/quality V6 ещё не выполнены.

V6 ops+cached integration isolated CPU/CUDA syntax прошёл rc0: engine,
library,driver,probe и реальная Engine/DSparkSession template instantiation.
Frozen snapshot `build-port7-dspark/v6-ops-cached-syntax-v1` содержит130
sourcefiles; compiler использовал hardened per-node marker APIv3. Root затем
заморозил combinedV6 (148 файлов) и собрал его rc0. Validation поставил
отдельную NEW matrix из трёх parity jobs: ops0/cache0 → ops1/cache0 →
ops1/cache1 c deferred hot290 и sealed actual-decode rank. Модельная matrix
ещё не завершена; индивидуальные sourcejobs повторно не запускаются.

Следующая независимая оптимизация после V6freeze:
`M31_DSPARK_SHARED_OVERLAP=1`. Отдельный persistent sharedGraph заимствует
main-graph packed norm после его allocation; main graph отдаёт CPU metadata,
затем shared FFN запускается async одновременно с CPU/hybrid routed. RAII
pending устанавливается ДО submission и гарантирует join до исключений,
reset/free; sharedGraph уничтожается до main norm-owner. Workspace считает
оба allocator, mode входит в geometry cache key. Default0 сохраняет прежний
путь и порядок readback. Source review dense агента прошёл; модельный gate
и speedpilot ещё нужны. Первый decode seed prime сам синхронизирует backend,
поэтому первый routed слой не даёт overlap; steady и seed wall различаются.

Следующий capture opt-in принадлежит prefix агенту. Controller и standalone
driver принимают валидный пустой feature slice для раннего chunk за пределами
окна; standalone выставляет окно prompt−4096 после reset при reuse1.
Verify перед своим legacy feature writer инвалидирует reuse metadata.
Fixture расширен skipped-window случаями, его новый запуск ещё ожидается.

V6 matrix завершилась rc0: ops0/cache0,ops1/cache0 иops1/cache1hot290 дали
128/128,maxabs0. Ordinary reference каждого arm получен внутри того же
процесса/ELF/profile и с тем же prefix recipe. Steady denseK8 при ops1
снизился~80→71.5ms (compute~70→61,read~9); общий numeric wall не считается
om acceptance. Cache1 включал реальный deferred seed290, exact main parity
сохранена. Actual M3 acceptance/speed в этом профиле ещё pending.
Подготовлен NEW one-process20arms sweep ops1/cache1hot290 с NVML и
same-ELF serial raw reference; запуском владеет validation/root.

Shared-overlap+capture-consumer isolated syntax прошёл CPU/CUDA rc0.
Controller fixture теперь4860cases/27864rows PASS, включая пустой ранний
capture chunk. Capture owner отдельно проверил565 component cases.
Scope этих проверок: компиляция/API/controller/host buffers; они не заменяют
модельный parity opt-in shared-overlap или final quality.

## P7-v8/v9: текущие проверки и границы выводов

Не путать P7-v8 candidate с принятым port6 v8. P7-v8 со independent VIEW и
PACKED прошёл3×parity128 maxabs0 и3 actualdepth1 raw128==ordinary. Numeric
скорости base24.227→VIEW23.926→VIEW+PACKED25.210т/с нельзя трактовать как
чистую причинную оценку: CPU wall заметно дрейфовал. P7-v8 long16K/32K
native+GPUoracle затем прошёл33rows,KL0 иbit-exact features с runtime
capacity/prefix guards (validation artifacts). Финальные quality suites ещё
не заменены этими локальными проверками.

P7-v9 source добавил отдельные defaultOFF MSA/LOOKUP/GROUP/capture-view.
MSA helper200 GPUcases: score/masked по67,481,600 F32, MAX/biased по527,200,
26,400 strided IDs — diff0. Scheduler сначала expands все n1 K/V/index
SET_ROWS, затем один F16→F32 CAST на Nk и группы sameNk≤4; attention получает
selectedIDX, skipwrite/skipselect, n1 sorted gather/FA прежние. GraphOPT1
запрещён для этой experimental ветки до отдельного proof. Lookup owner и
сам idx→tok/row helper принадлежат dense/CPU агентам, map lifetime/budget
учтены ими отдельно. Grouped cached helper прошёл warm/cold raw component
проверки у cached агента; основная CPU/GPU callback схема не изменилась.

Combined syntax CPU/CUDA Engine/Gpe/helper/actualSession прошёл; контроллер
9720cases/55728rows PASS, capture Host/View958cases PASS. В Engine init
исправлен только opt-in guard EXACT1/NATIVE0 при CPU_FUSED. Без compiled
exact helper GpuExperts::init по-прежнему отвергаетEXACT1, fallback нет.
P7-v9 Engine source SHA9baad5621bfe69c710c48ad513943c7d953bd0062efef7bb94447e5a476931d4.

Numeric causal chain base→MSA→LOOKUP→cached290→GROUPED: все5 parity128
maxabs0; actualdepth1 скорости23.278/24.594/23.048/24.471/24.414т/с в
нечистых окнах. Это ещё не35т/с и не основание включать M3 по умолчанию.
Capture-view этой standalone chain не упражняется: generate вызываетстарый
getter; lease API проверяется отдельно через реальный HTTP/session/rawprobe.
Standalone spec decode_ms также исключает sidecarinit+seedappend; HTTP TG
включает lazy prepare после первого ACK. Итоговую скорость считать по HTTP.

Следующий P7-v10 COALESCE_INPUTS прошёл helperGPU360cases/2,052,864F32,
26negative predicates (включая inplace producer dependency), source review и
Engine CPU/CUDA syntax. Он заменяет concat только adjacent full-width n1
views одного contiguous producer, count2..4. Guard требует VIEW1, mode входит
geometry, политики main/shared/head; flag0 старый AST. Modelcoalesce0/1
ещё pending. Purehelper header не пропускает inplace SCALE/CPY через
flattened view_src. Добавлены profile-only main/max/shared node counts для
измерения влияния на CUDA graph chunk512. GPU hidden finish описан отдельно
как future design, runtime не менялся.


## Закрытие волны DSpark: actual HTTP и чистая парная проверка

По решению координатора расширение M3 остановлено: следующий приоритет —
ordinary dense GPU. Gather-only FA не начинался; GPU hidden finish,
multiple-K persistent cache и F32 index shadow остаются предложениями.
Новых runtime правок после frozen P7-v10 в этой области нет.

Actual HTTP exploratory job
`/mnt/ai/logs/m31-port/port7-dspark-http-representative-v10-v1/`:
ровно6 запросов, actual om prompts4349/16640/33022, greedy256,
ignoreEOS, холодный prefix, profile MSA1/LOOKUP0/CACHE0/COALESCE0,
VIEW+PACKED+SHARED+OPS1, capture reuse/view1. Это по одному измерению на
конфигурацию, без повторений итогового om acceptance.

| Depth | 4K TG | 16K TG | 32K TG | 4K PP | 16K PP | 32K PP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 24,535 | 25,759 | 23,163 | 1276,5 | 1534,0 | 1245,9 |
| 3 | 22,084 | 21,719 | 20,965 | 1268,0 | 1529,1 | 1241,5 |

Скорости в токенах/с; TG=255 реально поданных токенов / predicted_ms.
Lazy sidecar init и seed append включены в TG: prepare150,6–166,1ms,
или1,31–1,56% TG. Начальная seed_ms отдельно не измерялась. Verify занимает
93,36–95,17% TG. Finish11,8–21,4ms показан отдельно и не входит в TG.
Нельзя обещать35т/с устранением одного sidecar startup.

`exploration-report.json` и дополнительный readonly
`phase-counter-audit.json` прошли проверки actual input IDs, PP recipes,
native generated IDs, sampling/ACK256, logical KV=P+255. Для каждого запроса
`verified=rounds+proposed`, `255=rounds+accepted`, `cached=accepted`.
На каждом контексте depth1/depth3 дали одинаковые256 native IDs. Это
cross-depth equality, отдельно обычный oracle для всех этих HTTP запросов
не запускался.

Чистый paired native RPC job
`/mnt/ai/logs/m31-port/port7-dspark-p7v10-coalesce-native-clean-v1/`:
один frozen P7-v10 worker ELF, prefix3324 одним chunk,128greedy,
4 команды ordinary→M3 сCOALESCE0, затем ordinary→M3 сCOALESCE1.
Lazy prepare включён в worker predicted_ms; это native RPC, не HTTP.

| COALESCE | Ordinary TG | DSpark depth1 TG | M3 к своему ordinary |
| --- | ---: | ---: | ---: |
| 0 | 25,348 | 23,035 | −9,12% |
| 1 | 24,667 | 22,807 | −7,54% |

Все128 raw IDs каждого M3 запроса совпали с его собственным ordinary
reference; у всех4 SHA256 `414644292d608e255614fa935779e21beded07fb3e6f63db587a9383db452f3f`.
M3 COALESCE1 к0 −0,99%, ordinary control дрейфовал −2,69%; воспроизводимый
выигрыш COALESCE не доказан. Результат отрицательный, feature остаётсяOFF.
Конфигурация depth1 приняла52/74 proposals за75 rounds,149 verify rows
на127 реально fed rows; prepare145–150ms, finish10,9–11,2ms отдельно.

## Default-off, ошибки и владение: заключительный source audit

Область аудита — замороженные DSpark изменения P7-v10 и их API; последующие
ordinary оптимизации других агентов этим заключением не покрываются.
`Config::draft_feature_layers` пуст по умолчанию, Graph view/coalesce flags
false, новые verifier ENV принимают только явное значение `1`. Без
`--dspark-model` worker не создаёт session/drafter и не включает feature
layers; библиотека линкуется, но не грузит веса и не создаёт GPU state.
`STRATA_M31_DSPARK=OFF` отключает research executables, не runtime linkage.
Capture Host/lease и sorted lookup также требуют отдельных ENV opt-in.

Forward без capture не добавляет feature graph/readbacks. Attention hooks
по умолчанию пустые; без selected_idx/kv_written выполняются прежние
SET_ROWS/MSA expressions. Verify вызывается только явно; все его
экспериментальные mode flags включены в geometry key, включая COALESCE.
Обычный forward освобождает retained verify scratch до новой allocation.
Узкое init-исключение EXACT1/NATIVE0 относится только к явно включённому
cached decode, без compiled exact helper init завершается ошибкой.
Это source-level проверка изоляции и локальные model proofs, а не новая
полная побитная приёмка default path относительно port6 v8.

Preflight verify отвергает неверные IDs/context/geometry/зависимости flags
до poison/KV mutation. После начала исполнения ошибка оставляет Engine
poisoned доreset. Shared async submission защищён RAII drain: join до
исключений и destruction. Main norm помечен OUTPUT при borrow в отдельный
shared graph. Shared consumer уничтожается раньше main producer; head и
layer graphs освобождаются до shared input buffers/KV/backend. Impl явно
сбрасывает verify/capture graph owners до backend_free. Workspace содержит
сумму main/shared/head/shared-input owners один раз.

Rewind сбрасывает head/hidden/logits/features/lease epoch, оставляет только
каузальный KV prefix. State geometry может жить дальше, masks/positions
обновляются на каждом verify. Controller хранит raw rows, выбирает token
один раз после предыдущего ACK, добавляет только consumed features;
reject/stop/EOS/cancel/max обрезают speculative tail к реально fed prefix.
Runtime ошибка worker сбрасывает Engine. Draft append при ошибке требует
reset; draft() пишет только scratch slots и не меняет подтверждённый ring.
Запрещено удерживать feature lease через следующий forward; release scratch
сохраняет host lease, reset/rewind/verify инвалидируют epoch.

CPU/CUDA syntax и actual controller9720cases/55728rows прошли; Host/View958
cases и сохранение seed append geometry прошли отдельно. Модельные local
proofs включают128-token parity/forced rejection, mixed K rollback и
16K/32K tails33rows сGPU-offload oracle KL0/featuresbitexact. Exception
probe для shared-submit с throw в route_observer собран, но GPU runtime
receipt в просмотренных артефактах отсутствует: его нельзя считать PASS.

Ограничения: итоговые GSM8K100/tools23/final gate20 на новом winner не
проводились; winner нет. M3 качество сохранено только в перечисленных
raw/logit checks. Не проверены HIP/SYCL и fusion1/batchops combination;
принятый профиль fusionOFF. При p_min переменныйK вызывает reconfigure и
CUDA recapture, bounded multiK cache не реализован. Драфт ограничен этим
конкретным GGUF/geometry; stochastic/grammar используют ordinary fallback.
64K M3 VRAM/quality не приняты; использовался targetctx40000.

## Полезный код для локального коммита

Runtime ядро этой работы (все opt-in): `m31_dspark.hpp/.cpp`,
`m31_dspark_verify.inc`, `m31_dspark_verify_state.hpp`,
`m31_dspark_batched_dense.hpp`, `m31_dspark_batched_graph.inc`,
`m31_dspark_batched_ops.hpp`, `m31_dspark_batched_msa.hpp`,
`m31_dspark_packed_routes.hpp`, `m31_dspark_coalesced_rows.hpp`,
`m31_dspark_session.hpp`, `server_dspark_policy.hpp`; точечная интеграция
в `m31_engine.hpp/.cpp`, `m31_serve.cpp`, `serve_m31.py`, `cmake/m31.cmake`.
Они зависят от согласованных CPU/GPU masked/cached, capture/view, sorted
lookup helpers и MMV marker backend patch: список зависимостей владельцы
передают root отдельно, нельзя коммитить только один .inc.

Research/verification: `m31_dspark_generate.cpp`, `m31_dspark_probe.cpp`,
`tests/dspark_session_check.cpp`, `port7_dspark_native_mini.py`; fixture и
API source audits в этом документе. Компонентные probes dense/cached и
`port7_verify_dump.cpp`/failure probe принадлежат своим владельцам и нужны
для воспроизведения raw proofs. Журнал, `port7-multik-evaluation.md` и
`port7-dspark-hidden-finish-proposal.md` полезны как история/резервы,
последний явно не реализация.

Старые `port7-dspark-feature.patch`, `port7-dspark-verifier.patch` и HTTP
proposal patches — промежуточные снимки, не текущие runtime источники.
`m31_engine.cpp.orig`, build trees и log artifacts не добавлять в source
commit. Unused F32-shadow/proposal headers не подключать как реализованную
функцию. Ничего не удалялось; коммиты и выбор итогового набора делает root.
