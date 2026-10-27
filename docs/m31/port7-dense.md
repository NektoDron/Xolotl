# Port7: задержки dense при декодировании

Работа поверх принятой v8. Новый режим `M31_DENSE_PINNED_IO=1` включается
отдельно; без него прежний путь остаётся выбранным. Измерен исходный v8,
сравнение кандидатов ещё в работе.

## Что действительно измеряет read_ms

В `m31_engine.cpp` при `M31_DENSE_ASYNC_READS=1` поле `compute_ms` охватывает
только вызов `ggml_backend_graph_compute_async`: постановку CUDA графа в очередь.
В `read_ms` входят GPU исполнение этого графа, вызовы трёх D2H и окончательный
join. Поэтому `compute_ms≈0,8` и `read_ms≈12` **не означают**, что GPU считает
0,8 мс, а лишняя синхронизация съедает 12 мс. Отдельные GPU activity интервалы
нужны до такого вывода.

CUDA graphs уже включены в v8, стабильные dense/shared графы живут между
токенами в пределах 256-позиционного bucket. Каждый dense слой читает norm,
IDs и route weights через `cudaMemcpyAsync` в обычные `std::vector`. Для
pageable назначения этот API может ожидать GPU; название Async не гарантирует
асинхронность передачи в таких условиях.

## Кандидат v1

`dense_pinned_io.hpp` выделяет один маленький CUDA host-buffer через ggml
backend API. Проверяется, что backend не откатился к pageable CPU buffer.
Разделены исходящий hidden и входящие norm/IDs/weights. Три входящих copy
ставятся в существующий stream, один join завершает их перед CPU MoE.
Далее точные байты копируются в прежние векторы. Веса, арифметика и порядок
F32 сумм не меняются. Режим действует только при n=1 и decode chain; prefill
использует прежние операции. Очистка дожидается stream перед освобождением.

`M31_DENSE_IO_TRACE=1` отдельно измеряет CPU время постановки norm/IDs/weights,
join и публикации pinned байтов. Старый `read_ms` сохранён. Эти дополнительные
таймеры выключены по умолчанию.

Изолированная сборка `/mnt/ai/Xolotl-m31/build-port7-dense/v1` сохраняет
CPU fused header из HEAD v8 и остальные объекты старого engine archive;
engine/generate/serve пересобраны вместе для Config ABI. CPU и CUDA TU
скомпилированы успешно; ELF generate `cc766acedf2c968d36e753051df79bc7938fb61da93f1908acd3b09408ccb8b5`,
serve `a87303d4bdb44de820de264dfa62de549c5ab175435172ae9ebf67784eb4496e`.
Это диагностическая сборка, полные quality гейты ей пока не приписываются.

## План измерения

Подготовлен `/mnt/ai/logs/m31-port/port7-dense-trace-v1`:
оригинальный v8 и одинаковый кандидат с pinned OFF/ON, каждому отдельный
Nsight CUDA trace. Фиксированный teacher-forced case: prefix128+decode128,
выгружаются все 128 decode logits. Компаратор проверит точные IDs/позиции
и logits между каждой парой. Профилированные времена не заменяют om_bench.

`tools/m31/port7_dense_trace.py` использует окончание logits D2H как границы
токенов и отдельно считает union реальных GPU kernel/DMA интервалов,
CUDA API времена, количества graph launches и роли копий по размерам.
Device idle включает CPU routed работу: его также нельзя целиком называть
задержкой переходов. Первые четыре decode шага исключаются из steady summary,
сырые строки сохраняются полностью.

## Фактический CUDA trace v8

`port7-dense-trace-v1/v8-trace.json`: 128 decode шагов, в steady summary124.
Среднее объединение реальных GPU kernel интервалов **11,037 мс/токен**,
DMA интервалов **0,330 мс/токен**, всего3754 kernel/токен.

| GPU работа | Запусков/токен | Сумма, мс/токен |
|---|---:|---:|
| Quantized MMV |241|6,225|
| F32 MMV |117|0,976|
| Broadcast arithmetic |952|1,410|
| RMS norm |361|0,534|
| Argsort |117|0,499|
| RoPE |240|0,410|
| Flash attention |60|0,395|

57 norm D2H по24576 байт суммарно занимают на DMA **0,0393 мс**, а связанные
вызовы `cudaMemcpyAsync` на CPU — **9,093 мс**: host API ждёт предшествующий
GPU граф. Это прямое измерение, подтверждающее, почему прежние12 мс read
нельзя целиком приписывать потерям переходов. Реальные GPU tensor bytes:
10679794176B dense включая голову1008322560B; из них9219409920B Q6_K.
Свод по ролям: `port7-dense-v1/dense-weight-bytes.json`.

Profiler wall42,709мс заметно больше обычного om; graph-node tracing повышает
цену host API. Это диагностические интервалы, **не новая оценка t/s**.
Данные copy-role API для D2D внутри CUDA graph относятся к родительскому
graph launch и могут повторяться для нескольких activities; они не складываются
как отдельные вызовы memcpy. Norm D2H выше сопоставлен именно cudaMemcpyAsync.

Первый кандидат OFF не дошёл до forward: frozen `m31_gpu_experts.hpp` уже
содержал новые поля Stats соседнего агента, а библиотека сохраняла v8 layout.
Из-за этого planner прочитал неверный slot_count и отказал. Это ошибка
смешивания build artifacts; ни v8 regression, ни математический отрицательный
результат из неё не следуют. V2 восстановил **все** прежние linked headers
изfaeb1332; CPU/CUDA compile завершён rc0, gen `fd38edd5bdd3b5834fd6ef19ac5db09ca6f47f6005559e6a2c3bc46574fb6f70`.
Новый job `port7-dense-trace-v2` сравнит OFF/ON с уже сохранёнными v8 logits.

## Следующий opt-in: decode-only fusion

Согласован `M31_GPU_DECODE_FUSION=1`: batch-prefill и его head принудительно
OFF, n1 dense/shared/head ON. `ggml-decode-fusion.patch` добавляет per-backend
override(-1 наследует ENV,0 OFF,1 ON). Изменение режима завершает работу и
сбрасывает captures; недопустимые значения не меняют состояние. Backend free
удаляет свою запись, Engine destruction возвращает прежний режим заимствованного
backend. Reset включает OFF. По умолчанию новый режим выключен.

API хранит side map по указателю context, его layout не меняется, поэтому
для изолированного эксперимента достаточно пересобрать один `ggml-cuda.cu.o`.
Главный CMake применяет дополнительный patch после общего Xolotl patch к
собственному FetchContent дереву; внешний `STRATA_GGML_DIR` лишь проверяется.
Локальный CMake probe проверил первое применение, повторную идемпотентность,
external verify и независимую применённость общего patch. Модельные результаты
новому fusion на тот момент не приписывались. Принятые v8 бинарники сохранены.

V3 выполнен: `port7-dense-fusion-v3`, job rc0. API component дал6176 checks
(включая6144 проверки конечности значений), RMS fused/unfused maxabs0.
V3 OFF и ON оба совпали с v8 на всех128 teacher-forced decode позициях:
top1=100%, mean/maxKL0, maxabs0. Это короткая диагностика до MSA sparse gate.

При одинаковом exclusive окне без profiler steady124 шага дали dense
10,457→10,358 мс, read9,644→9,551 мс. Такой малый сдвиг пока не объявляется
значимым ускорением. В обоих вариантах основная задержка находится внутри
norm enqueue8,935→8,844 мс; фактический join около0,028 мс. Этот второй
независимый замер также подтверждает, что pageable Async ожидает GPU compute.

Подготовлен V4: `M31_DECODE_FUSION_UNFENCED=1` снимает MUL fences лишь у
графов с явным single_token=true и effective CUDA backend mode1. Init требует
включённого decode fusion; большой batch/mode0 и default сохраняют fences.
Dense/shared/head отмечают single-token явно. Backend API/library полностью
совпадает с V3; пересобран только engine/mains. V4 gen `f7395d343f6c790712af139c67076374d3da7eb34f97b9e1a57ab561854315d2`.


V4 выполнен и **отклонён численно**: top1=100/128=78,125%, meanKL0,36558,
maxKL7,04470, maxabs11,5952. Уже первая decode позиция128 отличается
maxabs0,4959. Dense10,333 мс почти не отличается отV3 ON10,358 мс; более
низкий общий wall связан преимущественно с колебанием CPU routed compute.
Unfenced fusion принимать нельзя; большой sparse гейт этому кандидату не
приписывается. Обычный V3 сохранил точность, но заметного выигрыша не дал.

## Batched dense MMV component для M3

По поручению координатора подготовлены отдельные `port7-dense-mmv-*` probes.
Они выбирают из всех60 слоёв представителей каждого уникального role/type/shape,
включая исходные проекции и packed QKV/indexQK/GU с неизменёнными байтами,
down/O/router/head. Два семейства детерминированных синтетических F32 X
проверяются при K2/4/8: K отдельных stock n1 результатов против stock batch и
прямого forced MMV. Измеряются также CUDA event времена GPU-резидентных
scalar/batch проекций. Это не проверка модельных логитов или DSpark.

На RTX5090 stock Q6_K выбирает MMVQ лишь доK7, приK8 — MMQ; F32 выбирает
MMVF доK3. Прямой entrypoint позволяет проверить, сохраняется ли n1 арифметика
при принудительном MMV. В generic MMVQ K1/2/4 используют4 warps, K8 —2,
поэтому K8 может изменить порядок F32 reduction. Две K4 пачки остаются
отдельной возможностью даже при неточном K8.

`port7-dense-mmv-v1/component.json` получен, job rc0:28 сигнатур,336 случаев.
K2 stock и forced56/56 точны,1395712F32. **K4 forced56/56 побитно,
2791424F32**; stockK4 точен48/56, расходятся лишь F32 indexQ/indexK/packedIndexQK
и router, переходящие на MMF. K8 stock0/56 точен; forced сохраняет F32,
но Q6/Q8 меняют последние биты. Это подтверждает полезность именно K≤4 MMV.

В семейcтве обычных X K4 CUDA-event scalar→batch:
head2455→685мкс, packedQKV59,9→22,6мкс, packedIndexQK18,85→9,40мкс,
sharedGU43,8→16,0мкс. Проекции прогреты по одной и часть весов помещается вL2;
это не модельный t/s, не замер трафика всего10GB прохода и не доказательство
ускорения DSpark. Численная точность проверяется независимо от этих времён.

## Production marker для точного K4

`ggml-exact-mmv.patch` добавляет **per-node** API:
`ggml_cuda_mul_mat_set_exact_mmv(node)` и getter
`ggml_cuda_mul_mat_is_exact_mmv(node)`, macro `GGML_CUDA_HAS_EXACT_MMV_MARKER`.
Сеттер вызывается до allocation/capture и оставляет precision slot0 и hint slot1
неизменёнными. Magic в свободном для MUL_MAT slot15 явно выбирает MMVQ/F.
Node получает output fence, чтобы последующий fusion не обошёл этот выбор.
Поддержаны только contiguous2D F32/Q6_K/Q8_0 weights, F32 X и K1..4,
hintNONE. Если буфер весов уже выделен, допускается лишь usageWEIGHTS.
Неподдержанная форма/тип/hint/чужой slot15 отклоняются **без мутации**.
Dispatch дополнительно проверяет invariants; любой unmarked узел, включая MSA
score с RHS4, идёт прежним путём. Глобального переключателя MMVF нет.

Отдельный backend собран в `build-port7-dense/mmv-api-v2`, SHA библиотеки
`00c6b9f7e470baa1c0e46b666f4e3c66952a63ebb6f72d41294fd089add7b439`.
CPU metadata-only contract105/105 PASS: K1/2/3/4, idempotence,
precision/hint, K5/8/strides/MSA-channel rejection. CUDA не инициализировалась.
Дополнительный API patch применяется после неизменённого fusion patch;
CMake first/repeat/external checks воспроизводят байт-в-байт скомпилированные
vendor source/header. Оба patch независимо проходят reverse-check вместе
с исходным Xolotl patch (`mmv-api-v2/cmake-repro-proof.json`).

GPU API component расширен до K2/**3**/4, marked stock graph сравнивается с
отдельными n1 проекциями на тех же реальных28 сигнатурах. Отдельно unmarked
F32 MSA-like score RHS4/channel4 пересоздаёт CUDA capture после маркировки
и исполнения другой проекции.

V2 fixture остановился до actual-weight проверки: его synthetic score имел
W.ne2=4 и X.ne2=1, что нарушает broadcast условие ggml. Исправленная V3
создаёт все query channels и полностью инициализирует их. V2 library/ELF/
receipt сохранены; это ошибка тестового harness, не математический FAIL API.

Readonly review нашёл и устранил отдельную неполноту контракта: actual MSA
score использует 2D CAST(KV)[128,nk]×query[128,4], поэтому одной проверки
channel-axis недостаточно. В V3 source должен быть **leaf NONE с уже bound
WEIGHTS buffer**. CAST/computed и unbound source отвергаются до allocation
node без мутации. Actual dense/packed весам это дополнительное условие не
меняет арифметику. CPU contract вырос до125/125 PASS, включая actual2D case.
Новая библиотека `mmv-api-v3/libggml-cuda.a`:
`a62d16889b3fb7a4cfe6bee20c489771ee5e3f59aad89d309211361286afc6c1`.

`port7-dense-mmv-api-v3/component.json`: **K2/K3/K4 marked каждый56/56
побитно**, соответственно1395712/2093568/2791424F32 (всего6280704), maxabs0.
Unmarked MSA-like score4096/4096 после пересоздания capture точен. Всего
616 случаев на28 реальных role/type/shape. Это завершённый production API
component proof; batchdense model scheduler и его модельный гейт — отдельная
работа DSpark агента.

## Logical dense bytes и физический трафик

Resident10679794176B не содержит второго набора packed весов: исходные имена
alias-ят bundle rows, а allocator исключает их из самостоятельной allocation.
Отдельный `port7-dense-v1/n1-logical-weight-bytes.json` суммирует именно
используемые n1 роли, заменяя original QKV/indexQK/GU соответствующими pack:
critical dense7007059968B, shared2661285888B, head1008322560B, ещё3125760B
norm/routerbias. **Это unique logical weight bytes, не измеренные HBM reads.**
Кеширование, повторные загрузки и broadcast могут менять физические транзакции.
Прямой вывод о GPU11,037мс получен из CUPTI kernel intervals и не зависит от
модели пропускной способности памяти.

## Stock unary и router

Подготовлен stock-op K2/3/4 proof для RMS→MUL,
Q/K/index norm+RoPE с отдельными абсолютными positions, GLU и router
sigmoid/topK/sum/scale. Входы берутся из реальных исторических taps;
pre-norm Q/K и router logits восстанавливаются n1 GGUF проекциями.
RoPE stress vectors254..257/32766../65534.. используют те же данные:
они проверяют оператор, а не соответствие полной KV истории этим positions.
GLU дополнительно проверит packed `[gate;up]` с nullptr вторым аргументом
против отдельных gate/up. Новых CUDA ядер не добавляется.

`port7-dense-ops-v1/component.json`: **309/309 случаев побитно**, 7151904
проверенных значения, maxabs0. Это реальные historical hidden/FFN taps пяти
слоёв, n1 GGUF проекции для Q/K/index/router, реальные routed gate/up.
Batch сохраняет reduction axis каждого токена; отдельные stress positions
пересекают256 и включают32766/65534. Packed GU с nullptr up совпал с
отдельными gate/up views. При K4 диагностические CUDA-event медианы scalar→batch:
RMS18,55→6,22мкс, norm+RoPE22,70→6,24, packedGLU14,49→4,08,
router68,66→18,53. Component шёл в общем numerical окне со сборками:
это не измерение скорости полной модели и не прогноз t/s.

Добавлен отдельный production helper `m31_dspark_batched_ops.hpp`:
`dspark_ops_rms_rows`, `dspark_ops_norm_rope_rows`, `dspark_ops_glu`,
`dspark_ops_router_rows`. Он не меняет состояние/ENV, не содержит matmul,
MSA score или FA. Группы максимум4; выходы n1 — views, GLU возвращает
матрицу для непосредственного down. Geometry/position state и включение
нового флага принадлежат verifier scheduler. CPU metadata contract168/168
проверяет также N7/N8 разбиение и смещения positions второй группы.

Собран новый CXX-only actual-helper component `ops-v2`, ELF
`8965046f17613953cdb53cfed3a35c698436ca48542bb854ce36870ee3b64aa1`.
Он сравнивает production helper с прежними отдельными n1 subgraphs при K1/2/3/4
на тех же captures; получено444 случаев, из них412 непосредственно helper.
Оставшиеся routed rank4 GLU помечены `production_helper=false` и проверяют
stock mapping отдельно. `port7-dense-ops-v2/{component,summary}.json`:
**444/444 PASS**, 3896640 значений, diff0/maxabs0. Для actual production
helper отдельно **412/412 PASS**, 2913600 значений diff0/maxabs0. CPU
metadata168 и GPU conformance завершены; модельная точность, длинный MSA
контекст и итоговые t/s требуют собственных гейтов уже собранного verifier.

## Shared FFN overlap: следующая изолированная возможность

Serial v8 уже выносит shared FFN в persistent graph после router/norm read,
асинхронно запускает его до CPU routed и делает join перед residual.
В baseline wait около0,09мс/токен: основная shared работа закрывается CPU.
Для batched verifier возможен тот же порядок с отдельным persistent graph
на группу K≤4: sharedGU→GLU→down запускается после norm/ids/weights D2H,
CPU routed работает параллельно, join предшествует ранговой сумме+shared+
residual. Арифметика и порядок F32 сложений должны остаться прежними;
shared weights и source norm должны принадлежать живому persistent state.
Это пока архитектурное предложение, код scheduler не менялся. Нужен
отдельный opt-in pilot, численная сверка и чистое окно измерения; перенос
уже частично скрытой работы не гарантирует общий выигрыш.

## Reserve component: MSA channel batching

Подготовлен `port7-dense-msa-check.cpp`: F16 indexKV CAST→F32, query
`[index_dim,4,K]` с неизменёнными четырьмя RHS columns на query, F16 causal
mask CAST→F32, block MAX, local bias1e30 и topK. Scalar reference сохраняет
отдельные n1 score/MAX/topK, обе стороны используют один immutable CAST
ключей. Все score узлы **unmarked**; exact-MMV API для leaf dense weights
сюда не применяется. FA и движок не менялись.

Frozen job `port7-dense-msa-v1/port7-dense-msa-v1-card-job.json` ожидает
validation после основных V6 model checks. CXX-only сборка rc0, ELF
`337214f44a080059f4fc1d6a2ec41302759449059ee445de2e7ca8e25e8effb2`,
исходная v8 CUDA library. 75 случаев: K2/3/4, Nk256/4096/33280/65536,
пять слоёв, positions252/254/32766/65532. Сравниваются полные CAST keys,
score, masked score, block max, biased score и block IDs.

Queries и keys получены n1 проекциями и norm+RoPE реальных128 attn_norm
capture rows. Длинные immutable keys повторяют128 полученных F16 rows;
это operator fixture с большим числом tie, а не long-context model capture.
Одинаковый Nk у всех запросов каждой пары graph — явное условие proof.
Для будущей интеграции нужен общий Nk bucket, либо отдельное доказательство
изменения padding; нельзя объявлять ранний n1 query эквивалентным query с
большим Nk лишь из-за causal mask. Все KV writes должны закончиться перед
общим CAST.

GPU component завершён rc0: **75/75 побитно**. Отдельно score30366720F32,
masked30366720F32, blockMAX237240, biased237240 и11880 block IDs имеют
diff0/maxabs0; CAST keys323911680 значений также точны. В каждом NkK4
диагностическая CUDA-event медиана scalar→batch:256 —50,08→14,88мкс,
4096 —71,04→23,20мкс,33280 —150,44→78,91мкс,65536 —216,97→132,49мкс.
Это большой operator proof при immutable циклических keys и одинаковом Nk;
модельный scheduler и полный long-context gate остаются отдельной работой.

## Diagnostic Nsight: frozen V6 verifier K8

В `port7-verify-trace-v6` выполнен private main-TU probe: единственная
правка frozen V6 — NVTX `PORT7_VERIFY` непосредственно вокруг Engine.verify.
Engine/GPU/backend objects не менялись. Parity mode96 teacher-forced tokens,
K8, sharedinputs/batchdense/ops/cache ON, decode seed290, fusionOFF. Все12
ranges найдены; исключены первые3, steady9 frames=72 target tokens.
CPU KL loop и ordinary/prefill находятся вне ranges. Draft sidecar и пять
feature rows в parity mode выключены; это не end-to-end M3 benchmark.
Если feature5 включить, verify дополнительно копирует готовый CPU hidden
примерно0,98MB/round; GPU feature-capture graph этот код не добавляет.

Actual kernel union81,806мс/frame=10,226мс/target; DMA union9,814мс/frame,
device-busy union91,620мс/frame. Main stream kernel union53,168мс/frame
(6,646мс/target), exact cached worker28,638мс/frame (3,580мс/target).
Cached GPU работа идёт во время CPU routed, а не одновременно с основной
dense graph; складывать её с main GPU как последовательный wall нельзя.
Wall309,324мс/frame под profiler не является t/s результата без profiler.

На frame в среднем24967 kernels и11123 copies. Main hot families:
quant MMV482/10,946мс, get_rows_vec960/8,246мс, concat_cont6319/6,779мс,
broadcast3854/5,090мс; cached multiply663/27,164мс. DTOD32768B4441 copies
дают3,243мс, DTOD24576B3290 —2,400мс, DTOD2048B1933 —1,414мс.
Это указывает на полезность удаления лишних cont/copy перед следующими
арифметическими оптимизациями; соответствующий opt-in принадлежит DSpark.

Main CUDA API на frame: GraphLaunch121calls/48,039мс, StreamSynchronize428/
41,538мс, MemcpyAsync367/5,580мс. Они частично перекрываются с GPU и между
потоками, поэтому не складываются с union. GraphLaunch делится на121 вызов
из-за default512 ggml nodes per CUDA chunk; zero валиден и захватывает
graph целиком. В то же время v8 Nsight показывал GraphLaunch118calls/
5,766мс на один токен; V6 значение на target token6,005мс сопоставимо.
Большой per-frame API time преимущественно растёт с числом инструментируемых
nodes, поэтому сокращение числа launches не гарантирует пропорциональный
выигрыш. Подготовлен отдельный same-ELF CHUNK_SIZE0 strict128row parity
pilot `port7-v6-chunk0-parity-v1`; его отдельный итог приведён ниже.

`port7_verify_trace.py` сохраняет raw kernel/copy intervals и main/worker
API отдельно. Synthetic contract11/11 проверил NVTX text/textId, union,
overlap и denominator; независимая проверка actual intervals ещё нужна.

CHUNK_SIZE0 pilot выполнен: native rc0,128rows maxabs0/top1diff0. Отдельный
postcheck ошибочно требовал буквально KL==0 и получил rc1 из-за CPU double
округления mean−1,68e−19/max9,33e−18 при тех же логитах. Оригинальный failure
сохранён, corrected-analysis.json фиксирует математический PASS; GPU rerun
не нужен. После первых3 frames default512 compute60,893/dense71,367мс,
chunk0 compute61,108/dense72,086мс. Окна не являются matched A/B; заметного
ускорения не найдено, оснований менять default chunk size нет.

## Actual production VIEW_INPUTS helper proof

`port7-dense-view-v3`: **316/316 побитно**,5099520F32, diff0/maxabs0.
Production `dspark_column/join/projection/FFN` включены без правок в harness
с fakeGraph, несущим только context+view flag. Сравнение flag0 cont-copy
против flag1 retained contiguous views проверяет K1/2/3/4, parent width2D+256
и offset128. Использованы24 projection role/type/shape, включая head,
полный packedGU→GLU→down с ops0 и ops1, RMS и Q/K/index norm+RoPE при
positions254/32766. Входы — actual captured kqv/hidden/FFN и n1 derived QKV.
APIv3 CUDA library a62d1688; ELF baafec10509dd087f03aced41096b295e53b3608492039bea760a721d4adda89.

Диагностические event-медианы K4: packedFFN58,36→37,51мкс,
projection32,44→20,76, normRoPE17,03→10,70, RMS17,16→10,80.
Это подтверждает точность operator/helper aliases, а не корректность
production gallocr lifetime или модельную точность: для них нужен modelgate.
V1/V2 остановились на CXX harness include/#pragma-once ошибках до GPU;
исправленные snapshots сохранены отдельно, отрицательной математики не было.

Следом подготовлен `m31_dspark_batched_msa.hpp`, пока без Engine include.
Согласован context API → strided IDX[top,4,1] с общим F32 keys CAST,
sameNk1..4 и optional debug stage pointers. Actual-helper conformance V2
расширит raw75 до200 случаев: K1..4, F16/F32 masks на APIv3 backend.
Сборка V2 завершена; `port7-dense-msa-v2` дал **200/200 побитно** на
APIv3 backend. Score и masked каждый67481600F32, MAX/biased каждый527200,
strided per-query blockIDs26400 — все diff0/maxabs0. CAST keys863764480
значений также точны. K1..4, F16/F32 masks, пять слоёв и все четыре Nk.
ELF65d6be5dce97d88ef2b22142a701eef9e75f292d9ba38314ad90ca1bf9fd0225.
Это shared numerical окно со сборками: event timings не используются как
доказательство скорости. Header/Engine integration остаётся за DSpark;
sameNk и allKVwrites-before-sharedCAST требуют отдельной модельной сверки.

## Optional reserve: sorted gather + FA batch axis

`port7-dense-fa-v1` собран rc0 и передан validation после основных long
гейтов. ELF e19d752dcbd8a96989b32d8b3cd07c059543c05edfd9f0821eb2078d0b36b5da,
APIv3 backend. Только stock ops: native F16 GET_ROWS, FA c F32 accumulator;
Engine и CUDA kernels не менялись. 60 случаев: K2/3/4, четыре Nk, пять
слоёв, real captured Qcur и128 Kcur/Vcur rows. Длинный immutable KV цикличен,
порядок выбранных блоков fixture не монотонный и различается query/head.

В текущем n1 grouped FA уже использует ne3=KVheads4. Кандидат расширяет
его до4*K при Q sequence ne1=1 и GQA group16 без изменения оси редукции.
Общий KV GET_ROWS получает view с nb2=0 вместо материального repeat; маска
каждой query отдельная. Сравниваются байты gathered K/V/mask отдельно от
F32 attention. Возможный отрицательный FA результат не замаскирован rc0
диагностического процесса: report явно выдаст all_fa_exact=false.

Source review обнаружил, что fattn-common.cuh выбирает Stream-K occupancy
по ntiles_dst, пропорциональному Q.ne3. Расширение batch меняет число блоков
на один output tile и иногда uniform/general fixup. Поэтому n1 sequence
length само по себе не доказывает прежний F32 порядок. Если stock FA даст
diff, останется отдельно проверить batching только gather при прежних n1
FA, либо отдельный per-node dispatch с фиксированным original blocks-per-tile.
Последний вариант пока не реализован и не относится к штатному stock probe.

FA V1 остановился до математики на GGML view bounds: публичный view_3d
проверяет временный contiguous virtual size до установки nb2=0. Это harness
shape failure, не отрицательный attention result. Для V2 подготовлен bounded
VIEW с ne2=1, затем до allocation/capture ne2=K и nb2/nb3=0; проверяются
physical span, root bounds и сохранённый VIEW source dependency. Дополнительный
CPU metadata mode дал54/54 PASS на Nk256..1M/K2..4 до GPU init. Сборка V2
rc0; ELF ac3b52f19728b1c0b3bf8a07e09c7b1cbf63b355eb2283c3d05146c7059000cc.
V2 GPU job завершён rc0 диагностически; Engine не менялся. **Gather60/60
побитно**: K и V каждый147456000F16, mask1152000F16, diff0. Stock batch
FA точен лишь15/60 случаев (все Nk256); на Nk4096/33280/65536 все45 случаев
имеют различия. Всего1072467/1474560F32 отличаются, maxabs0,0269928.
Stock ne3 FA отвергается для exact speculation. Порядок/partition зависит
от occupancy, но отдельный runtime causal диагноз причины этим probe не
доказывается. В качестве резерва остаётся только batch gather + прежний n1
FA или отдельный exact-dispatch эксперимент. Дальнейшая FA интеграция
отложена; root повысил приоритет нового IQ3 n1 CPU-compatible IMMA kernel.

## Engine opt-in: MSA sorted-index lookup

По поручению root интегрирован `M31_MSA_SORTED_LOOKUP=1` (default0) из
отдельного CPU/GPU96case proof. Только n1 idx→tok/row expansion заменён
на immutable table get_rows/cast; score-sorted block IDs, K/V/mask gathers
и FA остаются прежними. Обычный n1 decode и verifier n1 используют один
код. Default0 сохраняет прежние ggml операции в else.

Один `MsaSortedLookupMaps` принадлежит Engine, а не слою:2MiB при64K,
32MiB при1M; shape/точная F32 representability проверяются helper. Graph
получает leaf aliases через Graph::borrow. Init создаёт maps до фиксированного
VRAM snapshot; actual free_after_resident уже учитывает allocation ровно
один раз, поэтому второй subtract/retained-workspace credit не добавляется.
Отдельный feature-on log фиксирует maps=1/capacity/resident_bytes.

Destructor освобождает maps после всех verify/dense/shared/head/capture
graphs, attention inputs и workspaces, до state/backend. Reset/rewind
сохраняют owner, чтобы persistent aliases не устаревали. Capture/session
области prefix и новые selected_idx/kv_written области DSpark не менялись.

Fresh combined live Engine CPU/CUDA syntax PASS: frozen receipt
`build-port7-dense/lookup-engine-syntax-v1/receipt.json`, Engine SHA
1ea8b919759a2865ef244cfe7b0f60398ba67e4c706a84e45f0c251c5f4d0207;
все header/inc совпали с live на момент ACK. Это только syntax, без runtime
или model acceptance. Same-ELF lookup OFF/ON parity/long-tail/reset остаются
за combined V9 validation. Frozen P7-v8 не менялся.
# Дополнение 27 октября, 14:02 UTC: обычный n1 и IQ3 IMMA

Разделение принятого V8 уточнено в
`/mnt/ai/logs/m31-port/port7-dense-v1/v8-bandwidth-breakdown-v1.json`.
Матрицы Q6_K/Q8_0 суммарно содержат 9,553643520 ГБ логических байтов на токен;
соответствующие GEMV занимают 6,225 мс, то есть 1,535 ТБ/с. По отдельным
графам: dense 1,528, shared 1,512, head 1,643 ТБ/с. F32 index/router:
1,123024896 ГБ / 0,976 мс = 1,151 ТБ/с. Это отношение логических байтов к
времени ядер, без аппаратных счётчиков HBM. Повторно исходные имена весов,
которые являются alias packed-бандлов, не учитываются.

11,037 мс GPU union содержит также нормы, RoPE, сортировки, gather и мелкие
операции. Суммы времени отдельных ядер больше union из-за перекрытия shared.
Поэтому нельзя трактовать всё dense-время как чтение матриц и обещать 2× GEMV.
GPU-idle между слоями содержит routed CPU и накладные расходы профилировщика;
host `read_ms` включает ожидание настоящей GPU-работы.

Изолированное новое IQ3 n1 IMMA-ядро прошло GPU-проверку:
`/mnt/ai/logs/m31-port/port7-iq3-lane-v1/component.json`.
1059 raw/basis и 684 реальных случаев, 1 079 336 I32 partials и 414 870 F32
совпали побитно с host reference, CPU vec_dot и старым GpuDeviceDots.
Проверены gate/up всех 57 routed-слоёв, expert 0/127, три реальных входа;
полная ширина 3072 строк проверена на четырёх слоях, на остальных 64 строки.
В реальных случаях I32 проверяется на семи выбранных строках, F32 — на всех
выходных строках. Raw basis проверяет все 256 позиций и все scratch-lanes.

Layer10 gate: warm 43,634 → 15,164 мкс, cold capacity rotation
60,864 → 20,160 мкс; up 43,719 → 15,152 и 59,776 → 20,576 мкс.
Ротация 15 копий даёт дистанцию повторного использования 101 154 816 байт
при L2 100 663 296 байт; аппаратные cache misses не измерялись.
Это quant+dot pipeline компонента, не ускорение полной генерации.
Допустимый Q8K qs: [-127,127], как у производственного квантизатора;
raw -128 исключён из-за семантики AVX2 psignb. Знак Q8K.d сохраняется.
Runtime-интеграция передана владельцу cached helper, default не изменён.

Узкое stock RMS_NORM→MUL fusion отдельно прошло 116/116 случаев,
408 064 F32 побитно: actual weights/derived activations пяти слоёв, четыре
токена. Файл `/mnt/ai/logs/m31-port/port7-rms-selective-v1/component.json`.
Медианы CUDA graph replay: Q 4,095→2,056 мкс, K 3,088→2,055,
indexQ 3,438→2,056, indexK 2,848→2,056; для hidden/FFN6144 выигрыша нет.
Ожидаемый верхний выигрыш порядка 0,30 мс/токен, требует проверки модели.
Новый opt-in `M31_GPU_RMS_FUSION=1` выбирает только этот matcher в mode2
на single-token forward, mode0 при n>1, сохраняя output fences MUL. Однотокенный
prompt и хвост prefill размером один токен также следуют политике n1.
Общий fusion и
unfenced режим с ним несовместимы. Компонент не доказывает точность всей модели.

Backend собран отдельно:
`build-port7-dense/rms-backend-v2/libggml-cuda.a`, SHA256
`bbf4b34f2290329bbc531404803b494b9934f4447507ddc4aca6620fdeae9298`.
CPU Engine syntax и CUDA Engine compile прошли. В сборке переиспользованы
все stock kernel objects, заменён только backend `ggml-cuda.cu.o`.
Patch `ggml-rms-only-fusion.patch` добавляет mode2 и atomic empty-map fastpath,
который избегает mutex при отсутствии backend overrides. Setter/free обновляют
счётчик под тем же mutex, graph-cache invalidation сохраняется.
Согласованный новый header обязателен перед stock include directories;
`GGML_CUDA_HAS_RMS_ONLY_FUSION` закрывает запрос флага со старой библиотекой.
Default0 не переключает backend policy. Первый build v1 остановился до NVCC
на CMake precheck: вложенный git apply без GIT_CEILING пропустил private-файлы.
Сборщик v2 исправляет окружение; старый неуспешный каталог сохранён.

Отдельный аудит восьми steady окон V8:
`/mnt/ai/logs/m31-port/port7-dense-v1/v8-graph-gap-audit-v1.json`.
Внутри 60 dense-графов за токен суммарно только 0,339 мс промежутков без
ядер данного графа, в среднем 5,65 мкс на слой; наибольший отдельный внутренний
промежуток около 1,09 мкс. Для shared сумма внутренних промежутков 0,0085 мс.
Глобальные промежутки между GPU-ядрами составляют 30,20 мс в профилированном
токене 41,79 мс, включая CPU routed, host работу и profiler overhead. Из них
27,50 мс относятся к промежуткам ≥50 мкс. Удалением межъядерных пауз внутри
CUDA graph нельзя получить несколько миллисекунд.

Сумма union отдельных dense-графов в этих окнах 8,525 мс; сумма длительностей
их отдельных ядер больше, около 10,138 мс в общей выборке. Stock PDL включён
по умолчанию и допускает перекрытие соседних ядер. Поэтому сумма длительностей
малых операций также завышает возможную экономию wall time при их слиянии.
Новые цифры являются разбором старого Nsight capture, не новым замером TG.

В 14:25 UTC завершён настоящий model gate на immutable V11
(`m31_generate` SHA256 `de36d1467b961c92b0c95fa12e28c280c2d693fe55eaa4a5af3f116381557d39`):
`/mnt/ai/logs/m31-port/port7-p7v11-rms-lookup-gate-v1`.
Три независимых запуска: OFF, только RMS1, только LOOKUP1; cached decode OFF.
Одинаковые fixed256 IDs, prefix128 и teacher-forced tail128, chunk32768,
head-tail1, сохранены последние128 строк logits. У обоих opt-in top1 100%,
mean/max KL=0, maxabs=0; дополнительно совпали SHA256 полных raw logits и IDs.
RMS arm выполнил переход backend policy decode0→decode1 (mode2 по флагу RMS),
lookup arm сообщил maps1/capacity40192/resident1286144 bytes. Итоговый unit rc0.
Это короткая проверка реального Engine/backend пути, без gate20, длинного MSA,
GSM8K/tools или измерения TG. Job measurement=false разрешал параллельную
сборку; численные timings из этих запусков не используются для оценки скорости.
