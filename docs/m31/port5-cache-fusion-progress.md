# Port5: кеш экспертов и локализация fusion

GPU-cache API и диагностика,27.10.2026. Это промежуточные результаты;
скорость всей конфигурации измеряет координатор отдельно.

## Выполнено

- `GpuExperts::forward_cached` выполняет исходный полный `[used,n]` graph
  только если все запрошенные эксперты уже находятся в VRAM. На промахе
  не выполняет upload/eviction/LRU updates, output не меняет. На попадании
  дренирует borrowed backend перед raw-stream exact CUDA dots.
- `exact_cpu_math()` позволяет engine запретить cached decode без строгого
  CPU-compatible CUDA пути. Обычный stock CUDA остаётся отдельным контрактом.
- `M31_GPU_PREFILL_COMPACT=1` подключает `enqueue_spans`: исходные bytes
  каждой роли и backend-readable quant padding пересылаются отдельно;
  fixed-slot alignment gaps не пересылаются. Требует async uploader.
- Допущен `max_chunk` до8192. Это лимит API, а не обещание, что allocator
  вместит такой chunk при любом context.
- Добавлены live `print_profile` и cumulative host-wall counters:
  prepare, pack, upload submit/wait, execute, gate/up/down, OAI/reduction.
  Async upload wall submit включает staging wait и host pack; отдельные
  uploader counters уточняют эти интервалы. `execute_wall` включает
  завершение устройства и readback, не является чистым kernel time.
  Exact kernel собственный print_profile выводится из того же live API.

## Проверки кеша

`codex-m31-port5-cache-build-v1` rc0, link-only build из2C++TU.
`codex-m31-port5-cache-probe-v1` rc0 через immutable card_launch/aictl.
Оба режима compact0/1 прошли seed/protection/evictions/recovery fixture,
добавленные cold/partial miss no-side-effects и warm cache bitwise hit
без дополнительных uploaded bytes. В fixture F32,6слотов/2hot/7токенов;
это проверка metadata/lifetime/bytes, не реальная M31 math acceptance.
Настоящий combined code128 выполняется на общей сборке координатора.

Сборка probe использует согласованные старые quants/executor/ops objects;
итоговый engine должен пересобрать весь набор с текущими заголовками.
Ничего из frozen port4 artifacts не перезаписывалось.

## Fusion: найден работающий выборочный барьер

Диагностический snapshot `build-port5-cache/fusion-source/m31_engine.cpp`
добавляет `ggml_set_output` для выбранного op перед graph allocation.
Основной `m31_engine.cpp` этим экспериментом не редактировался.
Все строки ниже: GPU0,F16,64CPUthreads,исходный strict CPU oracle.

|Fusion|Барьер|Промпт|Результат|
|---|---|---|---|
|ON|ADD outputs|RU64|Fail:top1.3125,meanKL3.596|
|ON|MUL outputs|RU64|Побитно:top1=1,KL/maxabs0|
|ON|RMS_NORM outputs|RU64|Побитно:top1=1,KL/maxabs0|
|ON|MUL outputs|code128|Побитно:top1=1,KL/maxabs0|
|ON|MUL outputs|code3000/decode32,30выбранныхстрок|Побитно:top1=1,KL/maxabs0|

Long проверка включает позиции63,127,128,129,2047,2048,2049,2999 и stride128.
Это облегчённый sampled gate, не full3000/full20. API head вычислял полный
chunk исходной формы, только сохранялись выбранные строки.
Первый long job v1 отказал `invalid rows spec`: старый frozen CLI не знал
`last:32`. Исправленный v2 использовал `stride:128`, rc0.

Вендорный `ggml_can_fuse` допускает output последнего узла fusion. Поэтому
MUL output fence оставляет RMS_NORM+MUL разрешённым, но предотвращает
поглощение MUL в следующий RoPE/ADD. Успешные RMS_NORM/MUL fences указывают
на семейство fusion RMS/MUL и его продолжений; конкретный неверный shader
пока не доказан. Standalone norm/RoPE fixture подготовлен для уточнения.
Чисто по исходникам multiADD/MUL и RMS+MUL имеют более слабые alias guards,
но объявлять их причиной без отдельного probe нельзя.

Отчёты: `port5-fusion-{add,mul}-compare-v1.json`,
`port5-fusion-rms-ru64-compare-v1.json`,
`port5-fusion-mul-code128-compare-v1.json`,
`port5-fusion-mul-decode-compare-v2.json` в `/mnt/ai/logs/m31-port/`.
Native chain/direct изменения другого агента этими gates автоматически
не покрываются. Их combined configuration проверяется отдельно.

## Далее

Сначала isolated speed windows и combined gates. Затем partial-hit decode:
CPU/GPU должны возвращать weighted terms на исходных rank slots и складывать
их в исходном rank0→3 порядке. Сложение двух partial sums не сохраняет
строгий FP контракт. Частичный путь без перекрытия может оказаться медленнее,
поэтому прежде его default нужны реальные counters и замер.

## Combined GPU candidate v1 и второй fusion bug

Frozen `build-port5-cache/candidate-v1/`: согласованно пересобраны generate,
serve, engine, GPUExperts, quants и оба host executor TU. Receipt содержит
SHA каждого source и двух ELF. Первый build отказал из-за отсутствующих
`json.hpp` include и copied `.cuh`; build v2 исправил оба и завершился rc0.

Code128: tensor-batched exact dots/reusable gate+up+down/OAI buffers,
parallel reduction opt-in, compact pinned ring, verified rank seed20,
cached-only decode и next-layer prefetch0/1 прошли побитно CPU oracle.
Cache248 slots/6GiB, arena170GiB, F16/context4096/maxchunk8192.
Cached decode в этом prefill-only случае не исполнялся.

Next-layer prefetch работает на отдельном backend/ring в CPU worker:
резервируются slots, не используемые текущим слоем или pinned seed;
cache metadata публикуется после успешного join, исключение дренирует DMA
до сброса metadata. Router, полный histogram, shapes и arithmetic не меняются.

Диагностические code128 forward wall (НЕ om_bench):

|Путь|Wall|Uploaded|
|---|---:|---:|
|Prefetch0|8.583s|115.29GB|
|Prefetch1, blind ascending next IDs|10.671s|162.76GB|

Prefetch1 переслал почти все128 next experts, тогда как code128 запросил
только около70% экспертов. Его excess bytes и CPU contention замедлили этот
малый batch. Первый foreground packing:7.029s/115.29GB,~16.4GB/s;
DMA staging wait13ms/submit65ms, CUDA kernels441ms. Это host copy bottleneck.

**Ошибка env этих диагностик:** задано несуществующее
`M31_GPU_CPU_QUANTS_HOST_THREADS=32` вместо `M31_GPU_HOST_THREADS=32`.
Constructor при отсутствии правильного env явно выбирает1 поток.
Следовательно OAI/quant/reduction этих probes исполнялись последовательно.
Это не результат host32 и не production speed. Также legacy route env
`M31_GPU_SORT_ROUTES` неверен; правильный `M31_GPU_EXPERT_SORT_ROUTES`.
Новый executor теперь отвергает такие известные неверные имена.

Large3000/c8192/decode32/headtail64 выявил **второй fusion bug**:
GPU candidate fusionON+MUL fence провалил64 saved rows:
top1=.546875,meanKL1.83797,maxKL12.5109. Тот же ELF с CPU-experts,
без арены/кеша, дал **побайтно те же51,216,384bytes**. Таким образом,
ошибка этого прогона локализована вне новой GPU-expert арифметики.
CPU-experts fusionOFF прошёл matching fresh oracle c8192:
top1=1,meanKL2.12e-9,maxKL9.42e-8,maxabs.002531.
Это numeric gate, не побайтовая эквивалентность oracle.

Large GPU prefix2968 wall27.83s, kernel5.23s,OAI8.16s,quant4.38s,
reduction3.95s; последние три статьи относятся к **host1**, как пояснено выше.
Prefetch join после execute~.454ms; это **не доказывает** одновременное
выполнение H2D и kernels: execute wall может сам ждать CUDA context/copies.
Host1 compute остаётся дорогим. Large speed window v1 отменён до
исправления второго fusion bug. Ни одна из этих строк не выдаётся за om_bench.

Артефакты: `port5-gpu-candidate-code128-p{0,1}-compare-v1.json`,
`port5-gpu-candidate-large-p1-compare-v1.json`,
`port5-large-cpu-fusion{0,1}-compare-v2.json`,
`port5-gpu-large-cpu-placement-byteproof-v1.json`.
Первый CPU-arm v1 отказал CLI `--arena-gib0`: нулевой режим задаётся
отсутствием флага, исправленный v2 выполнил оба arm rc0.

## Source для candidate v2

В canonical GPUExperts подготовлены и отдельно заморожены в v1 не были:

- `M31_GPU_PACK_THREADS=4|8`: bounded synchronous copy callback на
  persistent CpuPackSpans pool; shared main/background ring, lifetime/drain
  проверены независимо. Helper CPU byte gates прошли1/4/8 threads.
- `M31_GPU_PREFETCH_RANK=1`: next-layer order из уже проверенного rank.
  `M31_GPU_PREFETCH_LIMIT=N`: дополнительный cap числа speculative uploads.
- Отдельные cumulative background pack/wait/submit counters.

Итоговый candidate должен также включить правильный HOST32 env,
WMMA `M31_GPU_CPU_QUANTS_TENSOR=1`, исправленный prefill fusion fence
и пройти свежий code128/large gate перед первым speed window.

## Candidate v2: оба real gates пройдены побитно

`build-port5-cache/candidate-v2/` immutable58 source files и coherent ELF.
`codex-m31-port5-cache-candidate-build-v3` rc0, build receipt рядом с ELF.
Согласованная конфигурация:
HOST32,TENSOR1,PACK4,compact ring,rank seed20/cache6GiB,CPU fused decode,
projection concat,native F16 gather,adaptive8192,corrected safe fusion.
Code128 использовал prefetch0; large3000/decode32 — ranked prefetch64.
Cached GPU decode отключён в этой конфигурации.

`codex-m31-port5-gpu-candidate-v2-gates` rc0 через aictl/card lock:

- Code128: все128 rows **побитно** strict CPU oracle.
- Large3000,c8192,d32,head64,last64: все64 rows **побитно** исправленному
  strict CPU oracle, включая prefill last32 и32 decode шага.
- В обоих сравнениях top1=1,KL/maxabs0.

Перед этим обнаружена отдельная ошибка bounded oracle: crop до последнего
FFN менял исходный экспертный histogram/CPU dispatch. Новый reference
`port5-oracle-full-last-tail64-v2/oracle-c8192` вычисляет весь last FFN
перед crop (`full_last_layer_before_crop=true`). Старый native CPU fusionOFF
тоже **побитно** совпал с этим исправленным reference; прежние maxabs.00253
были ошибкой сравнения. Старые oracle файлы не перезаписаны.

Крупная fusion ошибка исправлена владельцем engine: при prefill n>1
gate/up matmul outputs сохраняются до конца graph, предотвращая неверный
MMQ+OAI fusion. Одного MUL fence было недостаточно. Decode n1 сохраняет
прежние разрешённые fusion. Вся GPU-expert arithmetic v2 принята выше.

Диагностические forward времена (не om_bench):

|Промпт|Forward|Ключевые статьи|
|---|---:|---|
|128|5.091s|Packing3.636s/115.29GB,~31.7GB/s; kernels.502s; OAI.046s|
|2968-prefix|14.913s|Execute9.754s; kernels3.640s; OAI.963s; quant.681s; reduce2.179s|

Large ranked prefetch64:80.31GB background,71.42GB foreground;
join после compute7.7ms. Background pack3.04s,submit5.19s; foreground
pack2.66s,submit27ms. Потенциальный driver/context stall в async submit
исследует kernel агент через private nonblocking stream. Его влияние на
полный speed нельзя получить простым сложением перекрывающихся counters.

Первое настоящее v2 om_bench4K окно организует координатор. До его результата
скоростной выигрыш v2 над llama.cpp не заявляется.

## Следующий source checkpoint

`GpuExperts::forward_into` пишет результат в retained caller-owned vector,
включая stock и exact пути. Прежний `forward` — совместимый owning wrapper.
Engine GPU prefill при host-read reuse передаёт `host_reads.moe`, исключая
новые zero/mmap73MB на каждом из57 слоёв. Caller output не может быть тем
же vector object, что x или weights; failure требует discard/reset.
Cached miss по-прежнему не меняет output. V2 ELF не менялся; новый путь
входит только в следующий coherent snapshot и требует своих gates.

## V2: настоящее4K окно

Замер координатора: TG26.09t/s,PP214.3t/s,4345 prompt tokens одним chunk,
prefill20.273s,peakGPU20263MiB. Frozen v2/PACK4/P1rank64,HOST32,cache6/seed20.
CPUv5 в соседнем изолированном arm:TG26.15,PP185.1. Референс llama.cpp:
TG23.33,PP425. Генерация v2 выше ref на11.8%, префилл всё ещё ниже49.6%.
M2 целевой префилл этим результатом не закрыт.
Подробности `port5-gpu-v2-short-p1-report-v1.json`.

## V3: query-tiling gate провален, реальные CUDA overlaps измерены

Frozen `build-port5-cache/candidate-v3/`,60 source files: private CUDAstream,
forward_into,PACK8,query tiles512/F16mask и прежняя WMMA/host32 схема.
Code128 прошёл побитно. Large3000/head64 **провалился**:
top1=.578125,meanKL2.3565,maxabs19.754. Конфигурация v3 не принята,
её full speed не запускался. Владелец engine локализует querytile/F16mask
тремя CPU/no-arena arms; остальные изменения не объявляются причиной.
Tail1 тест был запланирован, но worker корректно остановился после fail.

Nsight capture large native сохранился. Два независимых SQLite analyzer
дали одинаковый результат:

- Routed339 kernels,private stream14.
- Background H2D10752 transfers,80,310,435,840bytes,stream18.
- 61 H2D transfers пересекались с routed kernels,union18,700,973ns.
- Это только~0.52% kernel time. Малое перекрытие измерено непосредственно;
  утверждать полное перекрытие/скрытие H2D было бы неверно.

Main private-stream profiling сместил wait: background API submit5.19s→75ms,
однако pageable quant H2D на main caller .392s→3.186s. Нулевой late join
не означает отсутствие serialization. Owner kernels подготовил opt-in
pinned quant/readback scratch для следующего snapshot; ещё нужны gates.
Fresh result reduction2.179s→.259s подтверждает устранение повторного
выделения/обнуления caller buffer. Эти диагностические timings включают
Nsight overhead и не являются сравнением om_bench.

Трасса `port5-gpu-v3-prefetch-events-v1.nsys-rep`,SQLite рядом;
proof `port5-gpu-v3-prefetch-event-proof-v1.json`, независимый
`port5-prefill-overlap-independent-v1.json`. Математический fail этой
конфигурации указан отдельно; event proof его не отменяет.

## Финальная конфигурация V4

Заморожен 61 файл исходников в `build-port5-cache/candidate-v4/source/`, receipt рядом.
Generate SHA `e1b9fb435b51757f47b9c352c5d87bdf2673ea866690b4fa2c57f26f04cdef90`;
serve SHA `825ed0d7f6c6fc8ac8718aee971668c07e266aca28fde08dad90667fcca12b8d`.
Каноническая сборка проверяется отдельно; цифры ниже относятся к этим ELF.

Конфигурация performance:

- CPU80,HOST32,PACK8,original IQ3/IQ4/NVFP4,без requant.
- Cache7GiB/290slots,verified seed20,ranked next-layer prefetch128.
- PINNED1/STREAM1/TENSOR1,reusable gate/up/down/OAI/caller result buffers.
- Querytile0,TOPK_MASK1,inplace score ADD,peak-stage adaptive planner.
- Projection concat,native F16 gather,safe fusion,cached GPU decode0,
  CPU fused decode; maxchunk8192,context40000 в full speed.

Querytiling512 отключён после локализации: CPU/noarena TILE512/GATHER0
повторил GPU fail; TILE0/GATHER1 иTILE0/GATHER0 оба побитно oracle.
Direct top-k F16 mask устраняет expanded F32 mask без изменения полного
indexer GEMM/FA, Wout/FFN shape и полного routed histogram. CPU80 cheap
gate нового mask/inplace пути прошёл побитно перед GPU запуском.

`codex-m31-port5-gpu-candidate-v4-gates` **rc0**:
code128 все128 rows и3000/decode32/head64 все64 rows **побитно** strict
CPU full-last oracle,top1=1,KL/maxabs0. CPU80 decode покрыт tail32.
PINNED runtime active1,171calls,0fallback.

Мой и независимый Nsight анализатор совпали на339 kernels/627 H2D
intersections,union176,815,570ns (~4.9% kernel duration). Background H2D
stream18 перевёз160,175,554,560bytes; private routed kernels — stream14.
Это реальное, но всё ещё небольшое перекрытие. Не следует считать весь
transfer скрытым по короткому join. V4 отличается отV3 также размером
кеша/глубиной prefetch: causal attribution одному PINNED не заявляется.

### Полный om_bench: два прогона каждой глубины

Медианы из завершённого isolated full окна,точный request как reference:

|Глубина|Native TG|llama.cpp TG|Native PP|llama.cpp PP|
|---|---:|---:|---:|---:|
|4K|27.510|23.33|215.05|425|
|16K|26.385|22.63|231.30|—|
|32K|25.440|21.92|237.90|433|

TG выше ref на17.9%/16.6%/16.1%. PP4K ниже ref на49.4%,PP32K ниже45.1%.
Цель кратно превзойти llama.cpp по префиллу **не достигнута**; M2 частичный:
получен generation win,но PP acceptance не закрыт. M3 не начинался.
PeakGPU30653MiB,reportedRSS320.36GiB (file mappings входят).
32K проходы allocator завершил без OOM: после первых трёх8192 четвёртый
chunk7680 и tail762/763,а не8192×4. Это реальный adaptive путь.

Итоговый full20 gate после speed окна **пройден**:
`codex-m31-port5-final-gate20-v1`,20/20 prompts,все28,723 позиции,
top1=100%,mean/maxKL0,maxabs0,scope `M1-complete-logits`.
Отчёт `/mnt/ai/logs/m31-port/port5-gpu-v4-final-gate20-v1/compare.json`.
Тот же frozen generate SHA e1b9fb43; runtime gate F16/ctx4096/chunk2048/
CPU80/decode32/cache7 и все финальные env. Performance cfg ctx40000/
maxchunk8192 дополнительно покрыта code128 и matching3000/c8192/head64
мини-гейтами выше и runtime full speed/allocator проходами. Не смешивать
полный suite20/chunk2048 с облегчённым64-row сравнением chunk8192.
Отдельный GNU cmp20 выполняет координатор перед bytewise claim.

## Три следующих performance эксперимента

1. Тот же V4 ELF и request: unset против
   `CUDA_DEVICE_MAX_CONNECTIONS=32 CUDA_DEVICE_MAX_COPY_CONNECTIONS=32`.
   Это непроверенная гипотеза о ложной сериализации очередей. Сначала
   code128/large gate,потом отдельный Nsight capture и warmup+2×4K окно.
   Измерять blocked API,kernel/DMA intervals и PP,а не только late join.
2. Один compact prefix DMA на эксперт вместо трёх role DMA: в pinned stage
   сохранять исходные offsets и backend-readable padding,передавать только
   `[0,max(offset+allocation_bytes))`,исключая большой slot alignment tail.
   Сначала byte/padding/lifetime gates для меняющихся layer geometries,
   затем real logits и paired speed. Изменение числа CUDA API без изменения
   модели; прирост bandwidth не обещается заранее.
3. Подготовить gate/up pair до запуска next-layer transfers: общая CPU
   activation quantization при одинаковом vec_dot_type,оба CUDA matmul и
   combined pinned readback,затем OAI/down. Сохранить полный original
   histogram,expert→slot mapping,FP accumulation и rank reduction. Это
   проверит,можно ли уменьшить up/readback waits и организовать очередь
   compute раньше background H2D. Изолированный helper уже подготовлен
   kernel агентом,но основной runtime его ещё не использует.

В этой сессии эти три эксперимента не запускались. Новая математика и
runtime tuning отложены; после full20 — локальный commit и cleanup.
