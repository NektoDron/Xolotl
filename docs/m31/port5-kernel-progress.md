# Port5: исходные IQ3/IQ4 веса, батч CUDA с CPU-квантом активаций

Итог speed этапа: TG4K/16K/32K=27.510/26.385/25.440ток/с против
23.33/22.63/21.92 у MiniMax в llama.cpp. PP=215.05/231.30/237.90,
цель префилла выше425–433 не достигнута. Эти числа — совокупный frozen v4,
а ниже отдельно отмечены component proofs и границы их применимости.
Финальный M1-complete-logits gate принят:20/20 промптов,28723 позиции,
top1=100%,mean/maxKL0. GNU cmp для отдельного byte claim сводит координатор.

Реализован `M31_GPU_CPU_QUANTS_BATCHED=1`. Для IQ3_XXS/IQ4_XS при n≥32
новый kernel вычисляет плитку 16 строк весов ×16 назначений одному эксперту.
Исходный compressed weight блок декодируется один раз на плитку и повторно
используется из shared memory. Активации Q8_K также переиспользуются по16
строкам весов. Integer dot использует DP4A; scales применяются в integer,
затем сохраняются прежние256-value reduction и упорядоченные FMA по блокам.
Для экспертов с <8 назначениями сохраняются восемь независимых float
аккумуляторов и исходное дерево сложения. Histogram относится к полному
исходному батчу, routes выводятся в прежние позиции.

NVFP4 и n<32 сохраняют scalar kernel. Старый путь также доступен с env0.
Маршрутизация, исходные GGUF и activation quantizer не меняются.

Компилятор CUDA13.1 sm120a: IQ3 REG40, IQ4 REG44, shared10112 байт;
local spill/stack отсутствуют. Это описание ресурсов, не измерение скорости.

## Проверка арифметики

`codex-m31-port5-kernel-build-v1` rc0, затем
`codex-m31-port5-kernel-proof-v1` rc0 через штатный card_launch/aictl.
Все36 cases побитно совпали с CPU ggml и старым scalar kernel:

- default24: IQ3/IQ4/NVFP4 ×n1/7/128/2048 ×broadcast/per-route inputs;
- tiled disabled6: все три типа, n128, оба input layouts;
- tiled forced6: все три типа, n7, оба layouts (n7 сохраняет scalar fallback).

Default n128 содержит специально экспертов с7/8/9 назначениями, поэтому
проверяется переключение vecdot/tiled в рамках одного вызова. Использовались
исходные GGUF rows64. Это начальный component gate; real code128 и large64
позднее прошли в общем candidate v4, как указано ниже. Сам этот gate не full20.
Результат: `/mnt/ai/logs/m31-port/port5-kernel-proof-v1-summary.json`.

## Host storage и runtime profile

После v1 арифметической проверки добавлено persistent high-water storage
для quant bytes: gate/up/down не выделяют и не обнуляют packed vector заново.
API host quantizer получает logical byte length. `build-v2` rc0; этот ELF
новее v1, арифметический `.cuh` совпадает. Новый `print_profile(label)`
позволяет вывести counters до завершения HTTP worker. Детальные CUDA events
собираются только с `M31_GPU_CPU_QUANTS_TIMING=1`; без переменной события
не создаются и дополнительных GPU ожиданий нет.

`H2D_wall_ms` — host wall интервал memcpy активаций/IDs/worklist/tiles;
`kernel_event_ms` — CUDA stream interval вокруг kernel submissions для timed calls;
`kernel_D2H_wall_ms` включает kernel, выделение result и синхронный D2H.
Эти два последних счётчика нельзя складывать. `prepare_ms` включает
резервы scratch и построение scheduling metadata, `quant_ms` — CPU quant.

Подготовлен `port5_kernel_speed` для отдельного окна: обе реализации вызывают
тот же production wrapper, executor и quantizer; различается только batched
env0/1. Полные исходные rows, n128/2048; перед скоростным измерением сравнивает
полный выход побитно. Component результат нельзя выдавать за om_bench M2.

## Изолированное component окно v1

`codex-m31-port5-kernel-speed-v1` rc0, measurement=true, без сборок и других
тяжёлых CPU задач. Исходные полные строки весов,23 slots,16 host threads,
две измеренные итерации после прогрева. Полные выходы scalar/batched побитны.

| Тип | n | Scalar wall, мс | Batched wall, мс | Ускорение |
|---|---:|---:|---:|---:|
| IQ3 gate,3072 rows |128|5.999|2.154|2.79×|
| IQ3 gate |2048|99.125|55.349|1.79×|
| IQ4 down,6144 rows |128|7.117|2.688|2.65×|
| IQ4 down |2048|127.575|112.444|1.13×|
| NVFP4 down fallback |128|7.348|7.369|1.00×|
| NVFP4 down fallback |2048|132.343|135.183|0.98×|

CUDA events для IQ4 n2048: scalar98.608 мс на вызов →batched19.161 мс,
то есть GPU kernel ускорился≈5.15×. Преимущество wrapper ограничивается
созданием и обнулением200MiB `std::vector<float> result` на каждый вызов,
а также D2H. Это не ускорение полного сервера и не доказательство M2.
Raw: `port5-kernel-speed-v1.json`, `job-port5-kernel-speed-v1-00.log`.

После окна добавлен `multiply_into(..., std::vector<float>& output)` для
повторного использования раздельных gate/up/down readback buffers в engine.
Старый returning-vector wrapper сохранён. Новый output не может быть тем же
vector object, что input. При равном logical size resize/allocation/zero нет.
Результат v1 относится к старому returning wrapper; reuse измеряется отдельно.

## Tensor core IMMA: source и ограниченный math gate

`M31_GPU_CPU_QUANTS_TENSOR=1` использует четыре warps на CTA,16 строк весов
и до64 назначений одному эксперту. Каждый K32 sub-block вычисляется двумя
WMMA K16 операциями signed-int8→int32. Результат умножается на исходный
integer scale строки; восемь sub-blocks суммируются как int32. После K256
выполняется прежний F32 multiply+FMA в исходном порядке. Используется только
CPU tiled policy histogram≥8 или forced; короткие эксперты остаются DP4A.
NVFP4 остаётся прежним scalar fallback. Оригинальные веса не меняются.

Shared staging K16 половин имеет отдельные32-byte-aligned bases и stride32.
Compiled resources: IQ3 REG64/IQ4 REG63, shared10880B, stack/local0.
SASS содержит `IMMA.16816.S8.S8`. Это подтверждение tensor core dispatch,
не измерение скорости. `codex-m31-port5-kernel-tensor-build-v1` rc0.

`codex-m31-port5-kernel-tensor-proof-v1` rc0:36cases, оригинальные GGUF,
rows67 (включая partial M), n1/7/128/2048, broadcast/per-route input,
default/disabled/forced процессы — все выходы побитно CPU и старому scalar.
Default cases включают hist7/8/9, поэтому проверяются DP4A и WMMA в одном
вызове. Исторический forced probe использует n7, то есть scalar fallback;
новый `port5_tensor_check` позднее проверил forced n128 в PINNED v4 gate,
описанном ниже. Main code128/large также принял последующий candidate v4;
этот начальный standalone receipt не подменяет его модельную проверку.
Receipt: `port5-kernel-tensor-proof-v1-summary.json`.

## Подготовленный gate/up streaming pair

Новые `gpu_prepared_pair.hpp/.cu`: один исходный full-batch histogram и
Q8_K quantization, затем независимые группы logical expert→physical slot.
После события готовности группы ставятся асинхронно её gate/up kernels;
загрузка следующей группы может выполняться параллельно. Окончательный
readback сохраняет исходные route positions. Не требует заранее назначать
все физические слоты. Между группами проверяется отсутствие alias слотов;
повторная группа/незакрытый batch/incomplete read отклоняются.

CUDA bridge намеренно привязан к pinned vendor ABI: CUDA registry/device
проверяются до преобразования ggml event context в cudaEvent_t. Borrowed
event должен быть enqueue-wait до следующей записи uploader; storage и
все slot bytes живут до read/abort. Abort/destructor сначала drain default
CUDA stream. Host executor borrowed и должен пережить helper.

`codex-m31-port5-prepared-pair-build-v1` rc0. Подготовлен probe на исходных
IQ3/IQ4,67rows,n128/2048, непоследовательном logical→physical mapping,
реальных ggml CUDA events и негативных lifecycle случаях. **Numeric/event
probe пока не запускался; helper в основной путь не подключён.** Приоритет
текущего candidate — next-layer prefetch и параллельное CPU packing.

## Private nonblocking CUDA stream

`M31_GPU_CPU_QUANTS_STREAM=1` создаёт отдельный nonblocking CUDA stream
на весь срок жизни `GpuCpuQuants`. Все H2D копии quant/IDs/worklist/tiles,
scalar/DP4A/WMMA kernels, profiling events и D2H идут в этот stream.
Копии вызывают cudaMemcpyAsync; перед возвратом ожидается только этот stream.
Default0 сохраняет прежний путь. Actual `private_stream=1` выводится в profile.

Порядок current weight uploads сохраняется: `GpuExperts` уже завершает
свой uploader до execute. Параллельный next-layer uploader пишет в другие
защищённые cache slots. Никакой синхронизации с ним в обычном multiply нет.
Allocator growth по-прежнему использует cudaFree/cudaMalloc и может иметь
широкую синхронизацию; после high-water reserve этот код не выполняется.

Локальный guard объявлен после всех host vectors, используемых как async
upload source. При исключении он ждёт stream до уничтожения этих vectors.
Destructor также сначала ждёт outstanding stream work и только потом
освобождает GPU buffers/events/stream. Result и high-water quant storage
живут до своего join. После failed API/synchronize guard делает drain.

Первая версия сохраняет pageable result/quant storage. Standalone math
gate ниже включает этот stream. Main v3 trace показал: background submit
уменьшился с≈5.19с до≈0.075с, но quant H2D host API вырос с≈0.392с до≈3.186с.
Это перенос блокировки, а не доказательство итогового speedup. Main v3
large case не прошёл по логитам; CPU arms отдельно разбирают query tiling.

`kernel_D2H_wall_ms` в async режиме включает оставшееся queued H2D время,
kernel и D2H; `kernel_event_ms` остаётся интервалом между CUDA events вокруг
kernel submissions. Возможный host dispatch gap внутри этого интервала
отдельно не вычитается; фактическую kernel activity подтверждает Nsight.

## Pinned quant/readback с лимитом: v4 component math принят

`M31_GPU_CPU_QUANTS_PINNED=1` включает private stream и owned
`GpuPinnedScratch`. Суммарный бюджет quant+readback≤1GiB, включая page
rounding; каждый buffer хранит high-water capacity. CPU quantizer пишет
непосредственно в pinned quant span. D2H идёт асинхронно в pinned readback,
затем свой stream join и memcpy в прежний caller-owned output vector.
Когда pinned активен, дополнительный pageable quant staging не выделяется.

Grow/free сначала ждут свой stream. Перед новой аллокацией освобождается
старый buffer, поэтому transient pinned bytes тоже ограничены бюджетом.
Expected OOM/unsupported/OS allocation failure освобождает storage и
отключает последующие попытки, возвращая явный pageable fallback. Иные
CUDA errors не маскируются. Runtime profile содержит actual active/fallback,
capacities, budget, growth count. Pinned owner уничтожается до private stream.
`quant_ms` включает pinned ensure/growth перед CPU quantization; на первом
большом запросе это также cudaHostAlloc/free/stream synchronize. При готовых
high-water capacities ensure не делает CUDA вызовов. Growth count позволяет
отличать эти интервалы при разборе cold/steady запросов.

`codex-m31-port5-kernel-pinned-build-v1` rc0. Затем
`codex-m31-port5-kernel-pinned-proof-v1` rc0:36cases, original GGUF rows67,
all outputs побитно CPU и старому scalar. Default n1/7/128/2048,
disabled n128, forced n128, оба input layouts. Forced действительно
проверяет WMMA для экспертов с<8 назначениями. Проверены переходы размеров
и повторное использование pinned storage; реальный allocation failure
не провоцировался. Actual pinned active1 подтверждён в worker logs.

Receipt: `port5-kernel-pinned-proof-v1-summary.json`. Immutable artifacts:
`build-port5-kernel/frozen-pinned-v4a/`, ELF `port5_tensor_check`
SHA256 `1e966b1e89e56ddc7cf661f6115bcae8aed466eb2aa188a39b95dd7a0ee0482e`.
Общий candidate v4 затем прошёл code128 и large64/64 позиции побитно oracle.
Этот сокращённый модельный gate и component gate не выдаются за full gate20.
Итоговые om_bench4K/16K/32K приведены ниже; отдельный финальный full20
на frozen candidate принят: M1-complete-logits,20/28723,top1=1,mean/maxKL0.

Read-only SASS check отверг конкретную гипотезу constant-cache serialization:
CUDA macro в ggml-common задаёт static const __device__ таблицы. В WMMA
все LDC имеют uniform fixed offsets; table lookups идут LDG. Shared copies
могут снизить lookup latency, но эффекта32× из constant-memory divergence
здесь не установлено. V3 kernel от этой проверки не менялся.

## Следующий резерв после финальных M2 измерений

Pinned readback сейчас копируется в caller-owned vector обычным serial memcpy.
V3 trace на≈3K показал≈31GiB D2H данных (≈33GB; для округлённой оценки можно
взять33GiB). При условной CPU memcpy полосе16GB/s копия33GiB заняла бы≈2.21с.
Это оценка возможной цены новой CPU копии, **не измеренный memcpy timer**.
CUDA D2H device time и CPU memcpy wall нельзя автоматически складывать или
подменять друг другом.

Если финальный профиль подтвердит bottleneck, можно использовать уже
существующий host executor с32 workers для disjoint readback spans. GPU
join должен предшествовать CPU копиям; все workers должны завершить копии
до повторного использования pinned readback или изменения caller vector.
Порог≥1MiB оставит маленький decode readback serial. Копируются исходные
байты; F32 арифметика и порядок reduction не меняются. Дополнительный pool
не нужен, background packing уже имеет отдельный executor. Эта оптимизация
**не включена** в текущий candidate; сначала требуются итоговые speed/profile.

В v3 trace runtime memcpyAsync wall сильно отличался от device memcpy time:
private stream H2D≈3.185с host API против≈0.591с device, D2H≈5.052с API
против≈1.482с device. Background pinned H2D≈0.061с API против≈3.062с device.
Это scope разных часов/интервалов, а не три последовательные compute стадии.
Доказательство перекрытия — пересечение реальных CUPTI kernel/memcpy device
интервалов в одном context и разных streams; одновременные host API вызовы
такого доказательства не дают. V3 expert kernels∩background H2D составили
только≈18.7мс; само это число не является измерением throughput.
В v4 device trace627 background H2D copies пересеклись с expert kernels,
суммарное пересечение≈176.8мс против≈18.7мс v3. Это фактическое перекрытие
device activity; v4 одновременно изменил несколько настроек/операций,
поэтому весь эффект нельзя отдельно приписывать одному pinned flag.
Изменение overlap interval не является таким же множителем throughput.

## Сопутствующий разбор MSA для основного candidate

Read-only проверка нашла уже собранный vendor `ggml_top_k_mask`:
CUDA/CPU op формирует итоговую F16 mask напрямую из causal mask, top-k block
IDs и forward map `[128,blocks]`. Это позволило dense-направлению убрать
expanded F32 mask/gather/transpose copies, сохранив whole-query FA и original
MoE batch histogram. Параметры/strides отдельно проверены: picks layout
`[top,queries,1,index_heads]`, output `[nk,queries,1,index_heads]`, causal0/-inf
и padding сохраняются побитно. Фактический модельный gate проводил dense
агент: prefix2968/decode-tail32,64 выбранные позиции, TOP_K_MASK+inplace score
с query tiling0 прошли побитно oracle. Финальный общий gate20 затем принял
совокупный candidate; отдельное полное MSA-only доказательство не заявляется.

Query tiling512 до этого не прошло large math даже без fusion. Independent
read-only FA/concat review не обнаружил явной ошибки strides/offsets, поэтому
этот режим нельзя автоматически считать принятым. Fused mask обеспечивает
нужное снижение памяти без изменения query geometry; decode n1 прежний.

## M2: принятые результаты и граница результата

В совокупный v4 вошли strict CPU-compatible DP4A/IMMA dots, persistent host
quant/output storage, private CUDA stream и bounded pinned quant/readback.
Exact-model code128 и выбранные64 large позиции прошли побитно oracle.
Самостоятельный component proof36 cases также побитен, включая forced
WMMA для<8 назначений, partial rows и рост/reuse pinned capacities.

Итоговый om_bench frozen v4, медианы двух полных кругов, токенов/с:

| Контекст | Native TG | Reference TG | Native PP | Reference PP |
|---|---:|---:|---:|---:|
|4K|27.510|23.33|215.05|425|
|16K|26.385|22.63|231.30|—|
|32K|25.440|21.92|237.90|433|

Full speed unit завершился rc0. Peak GPU0=30653MiB, OOM не было. Actual32K
prompt выполнялся чанками8192×3,7680, затем762/763 токена. Финальный full20
завершился rc0:20 промптов,28723 позиции,top1=100%,mean/maxKL0,
scope M1-complete-logits. Результат:
`/mnt/ai/logs/m31-port/port5-gpu-v4-final-gate20-v1/compare.json`.
GNU cmp для отдельного byte claim сводит координатор.
TG выигрывает≈16.1–17.9%, PP остаётся ниже reference425/433. Цель кратного
ускорения префилла не достигнута. Разница всего candidate одновременно
включает изменения CPU/dense/cache/streaming направлений; эти TG/PP нельзя
приписывать только одному CUDA kernel или pinned flag. Сравнение относится
к MiniMax в llama.cpp; фоновые дежурные модели не являются reference arm.
Native GSM8K100 в этой сессии не запускался. Tools23=20/23 относится к
предыдущей сессии и не объявляется новой проверкой v4. До M3 не переходили.

## Следующая сессия: GPU-resident FFN вместо host roundtrips

Оставшийся принципиальный резерв — сохранить gate/up/down/OAI, routing
weights и final reduction на GPU. Сейчас strict путь по-прежнему делает
CPU quantization и OAI, передаёт gate/up/down на host, затем копирует
pinned readback в caller vectors. Private stream улучшает очереди, но эти
данные и CPU стадии сохраняются.

Предлагаемый порядок следующего этапа:

1. Точная GPU Q8_K quantization: reproduce CPU max-absolute с tie по первому
   индексу, RN division/multiplication, nearest-int и block scales. Q8_0 для
   NVFP4 проверить отдельно по фактическому CPU trait. Сначала сравнить
   quant fields/isolated dots, затем code128/model gate.
2. Gate/up device outputs →GPU OAI→GPU down-input quantization→down GEMM.
   CUDA expf не тождествен CPU libc expf. Для OAI нужен отдельный исходный
   tolerance gate top1≥99%, meanKL<1e-5/maxKL<1e-3; старое bitwise
   доказательство на такой режим автоматически не переносится. Подготовленный
   isolated `GpuPreparedPair::read_oai_into` не является принятой main интеграцией.
3. Суммировать GPU contributions в исходном rank0..3 порядке, с отдельными
   F32 scale/weight multiplications и additions. Это допускает сохранить
   CPU arithmetic contract для reduction без regrouping/FMA.
4. Возвращать только final MoE vector, затем связать его напрямую с resident
   dense residual/shared следующего слоя. На≈3K три текущих readbacks дают
   около31GiB; только final MoE вместо gate/up/down сократил бы этот объём
   примерно в8 раз. Это расчёт по shapes, не измеренный будущий speedup.

`CUDA_DEVICE_MAX_CONNECTIONS=32` оставлен как отдельная гипотеза для следующей
сессии. В этой сессии такой A/B не проводился, эффективность не установлена.
Ещё один read-only резерв — существующий API
`ggml_flash_attn_ext_set_n_kv_max()`. У MSA row не больше2048 finite keys
(16 выбранных блоков×128), а текущий grouped_fa bound не задаёт. Явный bound
может разрешить vendor sparse FA dispatch на длинном контексте; этот вариант
не реализован/не измерен, нужны отдельные math и speed gates. Reference PR
helper также не задаёт bound явно, поэтому это не объяснение сравнительных
цифр текущего этапа.
После текущих итоговых окон новые изменения математики/прототипы остановлены.
