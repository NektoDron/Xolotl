# Port7: частичный GPU-кеш декода

Состояние: native hot cache отклонён численно; CPU-exact subset прошёл
component и короткий model gate. Actual greedy калибровка завершена. Полная
скоростная/quality приёмка нового cache-пути ещё не выполнена. Принятая v8 остаётся отдельным baseline.

## Что изменено

Старый `M31_GPU_CACHED_DECODE` работал только при попадании всех четырёх
экспертов и требовал CPU-exact GPU арифметику. Новый вариант требует одновременно
`M31_GPU_CACHED_DECODE=1`, `M31_GPU_CACHED_DECODE_NATIVE=1`, native experts и
`M31_CPU_FUSED=1`. По умолчанию обе новые ветки выключены.

`GpuExperts::forward_cached_hybrid` строит один сохраняемый native ggml граф
на используемый слой. Он сохраняет форму `[used,1]`, GGUF-типы, BF16 accumulation
NVFP4, внешние scales и SwiGLU OAI. Граф выдаёт взвешенные вклады по рангам,
без преждевременной редукции. GPU считает резидентных экспертов, пока callback
`CpuFusedDecode` считает отсутствующих. Отсутствующие GPU-ранги ссылаются на
существующий слот; их результаты отбрасываются. Это сохраняет форму dispatch,
но тратит немного лишнего GPU-счёта. Итог складывается строго в исходном
порядке top-k. Это важно при частичном попадании: сумма GPU + сумма CPU меняла
бы порядок F32-сложений. Реальные native GPU dot/activation результаты всё равно
могут отличаться от CPU, поэтому одного правильного порядка недостаточно для
приёмки модели.

На обычном промахе веса не загружаются. При ошибке CPU callback работа GPU
дожидается завершения до снятия защиты слотов. Сохраняемые decoder allocator
учитываются суммой в `workspace_bytes()`, отдельно от prefill scratch.

`M31_GPU_DECODE_SEED_HOT=N` откладывает загрузку проверенного рейтинга до
первого decode-вызова. Используются существующие `M31_GPU_SEED_PATH`,
`IDENTITY`, `SHA256`, `RECEIPT`, `RECEIPT_SHA256`; обычный `M31_GPU_SEED_HOT`
должен быть0/unset. Текущий preflight по-прежнему требует запас128 streaming
слотов. Например, при7GiB/290slots максимум такого рейтинга162. Decode-only
pins снимаются перед следующим prefill, поэтому seed не урезает его working
set. Загрузка seed — разовая стоимость запроса, отдельная строка профиля.

По запросу DSpark добавлен `release_prefill_workspace()`: после завершённого
прохода native GPU-путь дожидается prefetch/upload/backend, освобождает
transient FFN scratch и создаёт пустой allocator. Pool весов и decoder-графы
сохраняются. Runtime-проверка этого освобождения ещё нужна.

## Калибровка настоящего декода

`port7_decode_calibrate.cpp` запускает обычный Engine и greedy autoregressive
вывод, игнорируя EOS до заданного числа токенов. В trace попадают только
реальные n1 forward-вызовы, а не переставленные teacher-forced prompt routes.
Сохраняются generated IDs и полные ID/weight/position/layer записи.

`port7_cached_build.py` замораживает v8 Engine ABI header и линкует маленький
main с существующим v8 archive. До этого тот же archive/набор библиотек
перелинковывает неизменённый v8 generate object; SHA полученного ELF должен
строго совпасть с canonical `2ab8b3c4…`. Buildv2 прошёл эту проверку и сборку.
Buildv1 завершился на пропущенном `<iostream>` в probe; исправление не касалось
движка. Build receipt: `build-port7-cached/route-build-receipt.json`.

Подготовлен frozen job
`/mnt/ai/logs/m31-port/port7-actual-decode-routes-v1/port7-actual-decode-routes-v1-card-job.json`:
8 разных RU/EN/code/math промптов, по128 generated, всего1016 n1 forward.
Он завершился rc0 и использует v8 profile, CPU80/cache7/seed0, context4096/chunk2048. Это
калибровка маршрутов, не окончательный om/GSM замер. Шесть кейсов training,
два held-out; `port7_decode_rank.py` проверяет полное покрытие позиций/слоёв,
строит совместимые JSON/binary рейтинги и replay hit rate по каждому слою.
Старый `cache-routing-v1.jsonl` состоит из prompt batches и не используется
как доказательство настоящего decode hit rate.

Первый численный тест нового hybrid — короткий code128 с32 n1 decode
против GPU-offload oracle, подготовка `port7_cached_prepare.py`. Даже при
успехе этот тест не заменит gate20, длинные позиции, GSM100 и tools23.

## Измеренные маршруты и потенциал пакетной проверки

8 кейсов дали1016 настоящих n1 forward,57912 routed layer passes и231648
expert lookups. Шесть train и два held-out сохраняются раздельно. Static
hotset на held-out:

| Резидентных слотов рейтинга | Hit rate train | Hit rate held |
|---|---:|---:|
|64|9,322%|7,784%|
|128|15,102%|11,654%|
|162|17,771%|13,205%|
|256|24,159%|17,654%|
|290|26,175%|19,084%|
|400|32,001%|23,394%|
|475|35,508%|25,661%|

Это доля экспертных обращений, а не измеренный GPU speedup. Для162 реальных
decode-only seed существующий preflight успешно проверил GGUF headers/stat,
новые rankJSON/binary и сформировал seal. Payload SHA унаследован из исходного
integrity report, без повторного сканирования модели.

По тем же настоящим generated маршрутам рассчитан union экспертов в полных
непересекающихся окнах K, отдельно для каждого слоя. Читаемые байты — сумма
оригинальных gate/up/down GGUF ролей, без padding fixed slots:

|K|Средний union экспертов|Union / обычные K×4 bytes|Потенциальная экономия чтения|
|---|---:|---:|---:|
|2|7,155|0,894667|1,118×|
|4|11,802|0,737170|1,357×|
|8|18,141|0,565742|1,768×|

Хвосты короче K исключены; окна не переходят границы запросов. Эта оценка
не включает draft time, acceptance, внимание, математику и overhead; она не
является измеренным ускорением DSpark. Полные per-layer числа и fingerprints:
`/mnt/ai/logs/m31-port/port7-actual-decode-rank-v1/{analysis,union-windows}.json`.

## Резерв CPU-exact GPU hit subset

Существующий `GpuCpuQuants::multiply_into` синхронен: CPU квантование→H2D→
GPU CPU-compatible dot→D2H/synchronize на каждую роль. Packed CPU activation
он не принимает, и quant в `CpuFusedDecode` не экспортируется. Поэтому готового
API для enqueue→CPU misses→finish здесь нет. Возможен небольшой отдельный
persistent worker: на нём gate→up→CPU OAI→down для hits, на основном CPU80
пуле одновременно misses. Вызовы не зависят друг от друга. Host OAI для1–4
экспертов идёт последовательно, поскольку меньше threshold32; HOST_THREADS1
избегает лишнего второго пула. Большая ветка пока не реализована.

Для held-out162seed replay это22,606 слоёв с хотя бы одним hit и30,106
hit experts на токен:67,819 синхронных role calls,422217B quant H2D и1479785B
F32 D2H на токен. CPU OAI92487 элементов. Байты малы;68 round-trip и
скалярные exact kernels могут поглотить выигрыш. Верхняя оценка сбережённого
CPU routed21,5ms ×13,205% =2,84ms до накладных расходов. Синхронные dot/OAI
можно перекрыть CPU misses отдельным worker, но это ещё не замер. Предпочтение
остаётся native пути, если он пройдёт численные/quality критерии.

## Бюджет64K

Статический расчёт `port7_cached_admission.py`:

| Владельцы | context40000 (capacity40192) | context65536 |
|---|---:|---:|
| F16 main+index KV | 5556142080B | 9059696640B |
| Dense веса | 10679794176B | 10679794176B |
|7GiB cache,290slots | 7513793280B | 7513793280B |
| Всего до scratch/context | 23749729536B | 27253284096B |
| Остаток из32607MiB | 10441188096B | 6937633536B |

На64K остаётся6,46GiB до CUDA context, dense/shared/FFN scratch, MSA scores,
масок и decoder-графов. Это план, не OOM/forward proof. Начальный кандидат
сохраняет7GiB pool; увеличивать кеш без проверки prefill нельзя, поскольку
уменьшение adaptive chunk может потерять принятый PP. Отдельный actual init +
decode на65536 ещё нужен. Принятый performance profile context40000 не
подменяется этими64K расчётами.

Статический отчёт:
`/mnt/ai/logs/m31-port/port7-cached-64k-static-admission-v1.json`.

## Что ещё проверить

- CPU-only rank merge116/116 PASS; combinedv1 compiled. Runtime проверки остальных API ещё нужны.
- code128 hybrid и generated IDs относительно обычного greedy.
- Actual calibration, held-out static hit rate и seed162 preflight PASS; нужно actual runtime seed coverage.
- Реальное CPU/GPU перекрытие, submit/join overhead и скорость с seed.
- Полная согласованная финальная приёмка при выборе hybrid в итоговый профиль.

## Фактическая численная проверка native cache

Combinedv1, code128:96 prefill +32 n1decode, три отдельных процесса в одном
card job, один frozen ELF/профиль, сравнение с GPU-offload PR oracle:

| Режим | Top1 всего128 | Mean KL | Max KL | Actual cache route coverage |
|---|---:|---:|---:|---:|
| Hybrid OFF |100%|0|0|0|
| Native cached, seed0 |100%|0,000062135|0,004595|439/7296 =6,017%|
| Native cached, deferredhot162 |88,281%|0,382188|15,426102|663/7296 =9,087%|

**Native hot162 отклонён.** Prefill96 позиций остался строго0diff. На32 decode
позициях совпали только17/32 top1 (53,125%), mean KL1,528751; первый
KL>0,05 уже на первой decode-позиции96, первый top1 mismatch на100. Aggregate
88,3% маскируется совпадающим prefill и не является приемлемым decode.
Seed0 затронул лишь четыре поздних слоя в среднем, поэтому его положительный
результат не переносится на посев экспертов по всем слоям.

Все три engine процесса завершились rc0; внешний diagnostic job rc1 отражает
численный отказ, а не ошибку CUDA. Seed hot162 занял разово59,442ms и
3381903360B H2D. Native decoder retained scratch10,85MB. Источники:
`/mnt/ai/logs/m31-port/port7-cached-code128-three-arms-v1/`;
отдельный phase breakdown `native-hot162/rows.json`.

## Exact subset с двумя join — реализованный резерв

После отказа native hot162 добавлен отдельный opt-in
`M31_GPU_CACHED_DECODE_EXACT=1`. Он требует тех же CACHE/NATIVE переключателей
и `STRATA_M31_CPU_QUANTS_CUDA`, но не меняет native prefill. Новый C++ helper
`gpu_cached_exact.cpp` использует уже существующие `GpuDeviceDots` CUDA
kernels: новых NVCC kernels не добавлено.

Один постоянный worker владеет private stream и маленькими фиксированными
GPU/pinned буферами. Он:

1. Загружает x; ставит CPU-compatible gate/up dots для hit subset.
2. Одним D2H и первым join получает смежные gate/up. На CPU применяет
   внешние scales и точный libm OAI.
3. Загружает activation, ставит down dots, одним D2H/вторым join получает
   down; на CPU применяет scale и route weights.

Основной CPU80 пул в это время считает misses. Ранги складываются прежним
`merge_cached_decode_terms` в исходном порядке. GPU quantizer сохраняет
обычные CPU Q8_K/Q8_0 layouts и short AVX2 accumulation. Worker явно восстанавливает полную настроенную `M31_CPU_AFFINITY`,
поскольку `ggml_threadpool_new` уже может закрепить основной поток за одним
ядром до создания helper. Все borrowed веса защищены до drain настоящего raw reader,
в том числе при исключении CPU callback. CMake добавляет helper только при
CPU_QUANTS_CUDA и компилирует CPU OAI с `-frounding-math -fno-fast-math`.

Фактический component proof прошёл:

- 27 полных n1 FFN случаев,387072 weighted F32 terms побитно с CPU Fused;
- 9 прямых CPU vec-dot probes /576 F32 outputs побитно;
- Исходные IQ3_XXS/IQ4_XS/NVFP4, слои3/10/52, настоящие L3 activation rows
 0/63/127, hit subset1/2/4;
- 27 worker jobs и54 joins, GPU277368B и pinned270336B;
- CPU и CUDA без CPU_QUANTS syntax PASS; CUDA helper/consumer compile/link PASS.

Перед nonblocking helper stream сам component harness явно дожидается
default-stream H2D загрузки тестовых весов. Исправление этого test-only fence
вошло в окончательный buildv3; предыдущий component ELF не запускался.
Buildv1 остановился до компиляции из-за отсутствующего parent directory;
в build script добавлен `parents=True`.

Proof:
`/mnt/ai/logs/m31-port/port7-cached-exact-component-v1/component-result.json`.
Это component, не model logits и не измеренный speedup. Следующий шаг —
code128 с тем же hot162 на combined ELF с helper, затем реальные greedy IDs,
скорость и полная приёмка только при перспективном выигрыше.

CombinedV3 exact subset с deferred hot162 прошёл code128 полностью:
top1=100%, mean/max KL0, maxabs0 во всех128 строках, включая32 decode.
Actual coverage659/7296 (9,032%). Worker495 jobs/659 experts/990 joins,
fixed device277368B/pinned270336B. Seed3,3819GB и59,450ms разово.
Это подтверждает обход численного отказа native math через старые CPU-compatible
kernels. Полный gate20/реальные greedy rollout и om ещё не заменяются этим тестом.
Путь: `port7-cached-exact-code128-hot162-v1/compare.json`.

В diagnostic профиле worker143,118ms за32 decode (4,472ms/token,
частично перекрытые CPU), а main join47,929ms (1,498ms/token) и
submit5,448ms (0,170ms/token). Это показывает риск малого выигрыша при9%
реальных попаданий: потолок сокращения21,5ms routed CPU здесь около1,94ms
до накладных расходов. Timing correctness job не объявляется om результатом.

По согласованию добавлен **отдельный decode-only preflight**
`port7_decode_seed_preflight.py`, разрешающий до всех290 существующих слотов
без увеличения VRAM pool. Model/rank/header/stat проверки остаются прежними;
меняется только резерв streaming slots во время fixed cached-only decode.
Обычный `cache_seed_preflight.py` и `seed_hot()` по-прежнему ограничены162.
В JSON сохранены исходный prefill budget, явный mode `decode-only`, контракт
обязательного unpin перед следующим GPU prefill. Payload integrity наследуется
из проверенного reference report, как раньше.

Actual GPU lifecycle proof прошёл9 проверок на собственной synthetic F32
модели с128 экспертами/слой и **ровно290 физическими cache slots**:

- Обычный permanently pinned seed290 отклоняется.
- Все290 decode-only pins загружаются лишь при первом cached decode.
- Следующий n32 GPU prefill использует все128 разных экспертов слоя3,
  из которых лишь34 были в исходном hot set: освобождение pins обязательно.
- Повторное prime290→unpin→prefill даёт побитно тот же результат.
- Prefill allocator освобождается отдельно от decoder buffers, затем повторно
  аллоцируется и даёт тот же результат.

Это настоящий GPU control/lifetime proof с маленькими F32 матрицами, а не
замена original-quants math/model acceptance. Матрицы занимают4,7MB, pool3,56MB.
Receipt: `/mnt/ai/logs/m31-port/port7-cached-lifecycle-v1/lifecycle-result.json`.
Code128 hot290 gate на том же combinedV3 также прошёл:128/128 top1,
mean/max KL0, maxabs0. Actual coverage821/7296 (11,253%),682 jobs/1364 joins.
Primer5,9892GB/105,284ms; worker183,203ms за32 decode, main join41,578ms
и submit7,926ms. Это ещё не om и не окончательный выбор cache-профиля.
При замерах primer, возникающий внутри генерации, обязательно входит в TG;
его нельзя вычитать как model startup.

Последующий actual greedy rollout exact290 прошёл1024/1024 generated IDs
(8 промптов по128) строго против ordinary canonicalV8. Manifest содержит
явную matched-geometry receipt: одинаковые реальные prompt chunks, n1 позиции,
F16 KV, context4096/chunk2048, CPU80/poll50/strict0 и исходный calibrator SHA.
Это неquality/om замер; он исполнялся параллельно сборке, timing не используется.
Receipt: `port7-cached-exact-greedy290-v1/greedy-compare.json`.

Affinity fix также прошёл actual GPU component: main намеренно закреплялся
наCPU0 до создания worker; worker восстановил маскуCPU0-7 (actual count8),
затем повторил27 FFN/9 dot bitproofs. Это исправляет наследование одного ядра
от уже созданного `ggml_threadpool_new`. При отсутствии явной affinity поведение
сохраняется. Receipt: `port7-cached-affinity-component-v1/component-result.json`.

Новый отдельный резерв — cached exact subset внутри K≤8 verifier — описан в
`port7-cached-batched-proposal.md`. Standalone и совместный компонент прошли побитную проверку. После freeze V5
root разрешил применить helper и thin GpuExperts wrapper; Engine интегрирует
отдельный владелец. Direct wrapper probe прошёл: 16 кейсов, 11 520 выходных F32 побитно,
all-hit/partial/all-miss и два восстановления после callback exception.
Receipt `port7-cached-wrapper-component-v1/component-result.json`.
Модельная проверка батчевого режима ещё ожидается. Дополнительный traffic receipt
учитывает union reuse: `port7-actual-decode-rank-v1/cached-union-windows.json`.

## Завершённые ordinary-cache результаты и новые opt-in рычаги

На P7-v9 обычная генерация exactcache290 дала **отрицательный итог**:
25,668→23,736 т/с (−7,53%), 128/128 greedy IDs сохранены. Hit12,222%,
primer108,171мс включён в TG; worker6,378мс/токен частично перекрыт CPU,
join1,286мс включает ожидание GPU и copies. Это исключает объявление старого
exactcache победителем. Receipt:
`/mnt/ai/logs/m31-port/port7-p7v9-ordinary-exactcache290-v1/compare.json`.

Новый contiguous GU helper прошёл60 случаев /1 474 560 F32 побитно с CPU,
включая исходные IQ3/IQ4/NVFP4, actual hot290 assignments и padding-gap
fallback. Nongroup n1 helper1,275–1,401× быстрее; grouped K8 неоднороден
(на L3 регрессия). Поэтому live `M31_GPU_CACHED_GATE_UP=1` действует только
при GROUPED0. Это helper latency, не генерация.
`port7-cached-gate-up-component-v1/component-result.json`.

Новый CPU rank-outer merge сохраняет исходный порядок прибавления по rank,
переносит rank0 memcpy (сохраняя signed zero), затем использует contiguous SIMD.
Проверено1281 случаев /1 648 391 F32, четыре FP rounding mode, cancellation,
subnormal/signedzero, размеры и invalid inputs. Clean CPU0 timing для E6144,
used4, всех14 масок с1–3hits:11,77–18,39→1,81–1,95мкс/вызов. Это экономия
10–16мкс на hit-layer, не устранение всего join. Runtime включается лишь
`M31_GPU_CACHED_RANK_OUTER=1`, default0. Proof/timing:
`build-port7-cached-merge-rank/v1/check-result.json`,
`/mnt/ai/logs/m31-port/port7-cached-merge-rank-timing-v1.json`.

Точные IQ3 n1 masked IMMA ядра из отдельного компонента dense owner перенесены
без изменения kernel body в `gpu_iq3_lane_exact.cuh/.hpp` и включаются один раз
в существующий `gpu_cpu_quants.cu`. Новый метод `GpuDeviceDots::multiply_iq3_n1`
использует прежнюю CPU-compatible GPU Q8_K quant, stage1 I32 CPU-lane partials
и stage2 исходные F32 FMA/Kblock/xor/0,25. Raw Q8−128 не входит в доказанную
область; finite CPU quant выдаёт[-127,127].
`M31_GPU_CACHED_IQ3_IMMA=1` применяется только к IQ3/count1/!tiled;
прочие типы/count>1 остаются на старом multiply. Private scratch4 718 592B
заранее резервируется для6144/3072 и входит в device/scratch SUM.

**Совместный компонент завершён:** тот же ELF
`33b8afd12ae5c1c9b382135c790d929cce58ec738662d2885fe92d2ecfa2655a`,
IMMA1/GU0 и IMMA1/GU1, по60 случаев /1 474 560 F32 точны
(**120/2 949 120** суммарно). GROUP1 GUcalls0, gap fallback и actualIMMAcalls
покрыты. Whole-helper nongroup n1 ускорился1,31–1,72×, включая OAI и2joins;
изолированный матричный3× не переносится на весь helper. Receipt:
`/mnt/ai/logs/m31-port/port7-cached-imma-gu-component-v1/component-GU0.json`
и `component-GU1.json`. Независимый read-only review:
`docs/m31/port7-iq3-cached-integration-review.md`.
Модельная OFF→ON→OFF проверка combined P7-v11 ещё ожидается.

### Coherency/default аудит перед P7-v11

Старые public symbols GpuDeviceDots/GpuCpuQuants сохранены; добавлены отдельные
методы, старый multiply signature не менялся. Объект GpuDeviceDots по-прежнему
содержит один unique_ptr; дополнительные поля находятся в private Impl.
`GpuCachedExact::Stats` расширен хвостовыми counters, поэтому helper, GpuExperts
и все читающие Stats probes пересобираются согласованно. Public GpuExperts::Stats
не менялся. ABI symbol audit:
`/mnt/ai/logs/m31-port/port7-cached-iq3-abi-audit-v1.json`.
Отличающиеся NVCC anonymous kernel/stub symbols не являются public ABI.

При новых флагах0 IQ3 scratch pointer/capacity остаются0, нового cudaMalloc нет,
старый multiply и CPU merge выбираются явно. Дополнительный cudaFree(nullptr)
в destructor находится вне hot path. Другой runtime caller GpuDeviceDots —
`gpu_fast_ffn.cu` CPU-dots diagnostic path — использует только старые методы и
не читает cached flags, поэтому туда IQ3 dispatch не распространяется.
Default-preservation на новом ELF всё равно требуется: rebuild добавляет CUDA
flags−O3/ftzfalse/fmadtrue/prec-div/prec-sqrt, одного source review недостаточно.

Candidate builder замораживает все `.cu/.cuh/.hpp/.cpp/.inc`, пересобирает
**существующий** gpu_cpu_quants.cu с теми же CUDA math flags, затем helper/Gpe/
Engine, и линкует новые gpu_cpu_quants.o +gpu_cached_exact.o **перед** старым
libm31_engine.a. Иначе старый archive мог бы незаметно дать старый dispatch.
Новая CUDA TU и common CMake target не нужны; всё остаётся под существующим
STRATA_M31_CPU_QUANTS_CUDA / M31_CPU_QUANTS_CUDA.
Actual profile footer доказывает активность: `iq3_imma_calls`,
`rank_outer_calls`, `joined_gu_jobs` >0 у включённой ON руки. При OFF линия
может отсутствовать и считается0; cumulative footer берётся последний.

### Файлы для selective commit (не команды git)

Runtime cached область: `gpu_cached_exact.cpp/.hpp`,
`gpu_cached_decode_impl.hpp`, `gpu_cached_batched_impl.hpp`,
`cached_decode_merge.hpp`, `cached_decode_merge_rank_outer.hpp`,
`gpu_resident_dot.hpp`, `gpu_cpu_quants.cu`, `gpu_iq3_lane_exact.cuh/.hpp`,
`m31_gpu_experts.cpp/.hpp`. Engine/CMake/profile изменения принадлежат root и
DSpark owner; их следует выбирать вместе с coherent build, без автоматического
включения всех untracked файлов.

Положительное воспроизводимое новое tooling:
`port7_cached_imma_gate_up_build.py`, `port7_cached_imma_gate_up_check.cpp`,
`port7_cached_merge_rank_build.py`, `port7_cached_merge_rank_check.cpp`.
Матрица IQ3 изолированно: `port7_iq3_lane_imma.cu`,
`port7_iq3_lane_probe.hpp`, `port7_iq3_lane_reference.hpp`,
`port7_iq3_lane_check.cpp`, `port7-iq3-lane-build.sh` (владелец dense).
Документы: этот файл, `port7-cached-gu-merge.md`,
`port7-iq3-cached-integration-review.md` и существующие принятые cached/grouped/
coalesce/packed отчёты. Старые failed patch proposals, growth reserve и временные
build scripts не являются обязательными для runtime commit. Frozen receipts/
raw logs сохраняются в `/mnt/ai/logs/m31-port/`, ничего не удаляется.

### P7-v11 ordinary OFF→ON→OFF: завершён, +10% не доказан

`/mnt/ai/logs/m31-port/port7-p7v11-ordinary-imma-rank-aba-v1/aba-comparison.json`:
один ELF, prompt3324/greedy128, exactcache290 IMMA1/GU0/RANK1,
все128 IDs во всех трёх руках совпали. TG **26,182→26,154→25,059 т/с**;
conservative ratio к лучшей OFF0,99895, OFF drift−4,29%. Primer108,198мс
включён. Это не победитель+10% и не sustained/om/quality acceptance.
GU1 пока имеет integrated-helper proof, но не эту model ABA руку.

All128 routed_compute: OFF22,979 /ON23,108 /OFF2 24,951мс/токен;
dense13,269 /13,224 /13,143мс. Диагностическое исключение первых4шагов даёт
ON routed22,172 противOFF22,925; primer из опубликованного TG **не вычитается**.
ON footer: actualIMMAcalls10020, rankmergecalls2806, GUcalls0; hit12,222%.
Worker484,745мс/128=3,787мс/токен, gateup1,585, OAI1,020, down1,168;
join0,431, submit0,153, seed0,845мс/токен. Старый P7-v9 worker6,378 иjoin1,286
были больше, но это разные ELF/окна; итоговый causal speedclaim использует
только новую ABA тройку. CPUmiss876,479мс/128=6,847 относится лишь2806hitjobs,
а не всем57×128layercalls. Worker и CPU перекрываются, времена не складываются.

Starvation не подтверждён: default affinity не задана; model OAI36,6мкс/expert
сопоставим с idle component. Данных sched trace о tail delays нет. Возможные
резервы (не реализованы/не обещаются): назначение отдельного CPU для host OAI,
terms-only CPU callback без unused_sum/copies, уменьшение двух GPU↔CPU joins
с сохранением exact OAI. Ни один не объявляется готовым ускорением. По решению
root новые ветки прекращены, v8/default сохраняется. Независимые stage/counter
данные и SHA логов: `cached-independent-stage-analysis.json` рядом с ABA report.
