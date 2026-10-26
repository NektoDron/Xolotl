# M2: persistent GPU decode, projection bundles and safe fusion

Текущий source: `tools/m31/m31_engine.cpp`, `m31_engine.hpp`.
Первый принятый быстрый checkpoint направления — frozen v5;
`/mnt/ai/logs/m31-port/port5-dense-source-v5.json`,
`port5-dense-elf-v5.json`. Последующие GPU-кандидаты собираются отдельно
из согласованного snapshot всех headers/CPU/CUDA исходников.

## Изменения

- Убрана повторная `ggml_backend_synchronize` после уже синхронного
  `ggml_backend_graph_compute`. Графы получают уникальный uid после
  окончательной allocation, когда доступен внутренний pinned ggml API.
  Их metadata/addresses/strides/op parameters больше не меняются: меняются
  только содержимое inputs. Backend может пропустить полный metadata scan.
- `M31_DENSE_ASYNC_READS=1`: H2D на compute stream; несколько D2H outputs
  ставятся в тот же stream и завершаются одним join. RAII drain объявлен
  после локальных host-векторов и синхронизирует при исключении до их free.
- `M31_GPU_DECODE_CHAIN=1`: только n1, CUDA, persistent dense cache, no taps.
  Dense leading layers соединены GPU leaf aliases. Следующий routed graph
  начинается с прежнего `(CPU_moe + GPU_shared) + GPU_residual`; в head
  заканчивается последний residual. Сохраняется исходный F32 порядок.
  Shared input идёт D2D. Residual/shared/hidden больше не читаются на CPU
  каждый слой. Alias tensors не содержат producer op: предыдущий graph
  повторно не обходится и не вычисляется.
- `M31_GPU_PROJECTION_CONCAT=1`: при загрузке original rows Q/K/V,
  index Q/K и FFN gate/up одинакового типа пакуются в общие GPU matrices.
  Original named weight tensors alias те же байты в bundles. Дополнительной
  копии весов в VRAM нет; GGUF/quant/scales не меняются. На настоящей модели
  180 groups/420 aliases: QKV и FFN Q6_K, index QK F32. На n1 выполняется
  один matvec на bundle и views результата. Prefill сохраняет прежние
  отдельные matmuls, поэтому batch geometry не меняется.
- `M31_CPU_FUSED=1`: persistent CPU executor агента cpu_executor вызывается
  на n1 и использует тот же ggml pool/traits, без второго CV pool. Создаётся
  после pool; при reset/деструкторе ownership и borrowed backend сохранены.
  Старый `M31_CPU_DIRECT` остаётся отдельным экспериментом, для скорости
  отключён: measured component sweep показал его существенное замедление.
- `M31_GPU_CACHED_DECODE=1`: интегрирован complete-hit GPUExperts API,
  side-effect-free miss возвращается к CPU. CPU-exact GPU math обязательна.
  Partial hit API и перекрытие его работы остаются отдельным направлением.
- CPU pool всегда печатает actual threads/persistent/poll/strict/affinity.
  GPU cache profile печатается до завершения процесса, не только destructor.

## Две ошибки fusion и рабочий обход

`M31_SAFE_FUSION=1`, fusion enabled (`GGML_CUDA_DISABLE_FUSION=0`):

1. До gallocr reservation помечаются output все MUL nodes. Это устраняет
   исходную RMS/norm/MUL/partial-RoPE fusion проблему. Истинную причину и
   минимальный vendor fix отдельно исследует stream_prefill.
2. В FFN только при n>1 помечаются output gate/up matmul inputs к OAI GLU.
   Это запрещает ошибочный large MMQ+GLU fusion. Пометка лишь конечного
   GLU output не запрещает fusion. n1 граф этим вторым fence не меняется.

Первое diagnostic om_bench v3 4K дало 23.40 TG/182.6 PP, но его large
prefill fusion позднее оказалось некорректным. Эти цифры не являются
принятым результатом M2. v4 production speed не принимать без второго
fence.

Принятое изолированное окно v5 CPU routed: **TG4K26.15 t/s, PP4K185.1**
против llama.cpp23.33/425. Тот же unchanged om_bench/ref low request,
CPU64/F16/chunk2048/ctx8192, новый safe fusion и fused CPU включены.
Main dense12.389ms/token, CPU MoE23.662, shared enqueue.555/join.089,
head.721; peak GPU12985MiB. Это +12.1% TG, префилл остаётся медленнее.
Отдельный объединённый GPU v2:26.09/214.3, не результат только этого
направления. Источники: `port5-cpu-v5-short-report-v2.json`,
`port5-gpu-v2-short-p1-report-v1.json`. Новые full4K/16K/32K публикует
координатор после окончательной конфигурации.

## Проверки

- v2..v5 link-only CUDA builds: rc0, sm120a/gcc14, coherent frozen source;
  CPU tiny NumPy chunks1/7/29 maxabs0.00011285, arena/workspace/pool parity.
- v3 baseline code128: все128 rows побитно CPU oracle. Chain и затем
  chain+safe fusion: prefix120+tail8, все128 rows побитно baseline.
- v4 concat и concat+CPU fused: те же128 rows побитно baseline.
- v4 GPU fixture lifecycle: reset, 256→512 bucket, n128→n1, taps,
  no dense cache, GPUExperts placement fallback и caller-owned outputs:
  все побитны. Probe artifact: `port5-dense-lifecycle-v4.cpp` и ELF в logs.
- v5 safe fusion large: prefix2968 одним chunk +32 decode, head64/last64,
  CUDA/F16/ctx4096/t64, CPU routed. Все64 rows ПОБИТНО исправленному
  CPU strict oracle: `port5-dense-safe-large-compare-v5.json`.
  Reference: `port5-oracle-full-last-tail64-v2/oracle-c8192`.
- v5 native F16 gather + adaptive API: code128 prefix120+tail8, все128
  logits побитно baseline: `port5-dense-nativegather-code128-compare-v5.json`.
- Финальный frozen v4 прошёл полный suite20 отдельно:20/20 cases,
  28 723 positions, top1=100%, mean/maxKL=0, maxabs0,
  scope M1-complete-logits. Отдельные ранние gates не подменяют этот итог.
  [Полный comparator report](/mnt/ai/logs/m31-port/port5-gpu-v4-final-gate20-v1/compare.json).

Старый bounded oracle crop до FFN59 менял routed histogram. Его небольшие
расхождения в v4 long64 больше не служат эталоном. Новый oracle оставляет
полный последний FFN и обрезает только head. Компоненты с разным
`full_last_layer_before_crop` нельзя сравнивать как один контракт.

## Native gather и adaptive prefill

`M31_GPU_NATIVE_GATHER=1`: n1 MSA использует имеющийся CUDA GET_ROWS с
F16 destination для K/V и mask. Public ggml helper всегда создаёт F32,
поэтому локально строится тот же supported op с F16 type. Удалены три
промежуточных cast на слой. CPU/taps/default сохраняют старые операции.
Никакой арифметики KV не меняется: F16 cache bytes и mask0/-inf сохранены.

`Engine::prefill_chunk(requested,error)` вызывается main generate/serve
перед каждым prefill block. При default env0 просто ограничивает request
контекстом/max_chunk. `M31_PREFILL_ADAPTIVE=1` использует `plan_prefill`,
реальную свободную VRAM после resident weights/full KV/expert pool и
учитывает защищённые seed slots. Snapshot free оставляет credit памяти,
уже удерживаемой scratch allocator после предыдущего chunk. Два GiB
reserves включают до~925MiB lazy routed buffers и backend overhead.
При росте nk предложение уменьшается, int32 MSA bound проверяется заранее.
В manifest записываются actual chunks. Это conservative proposal; реальный
allocator gate 16K/32K и peak memory обязателен и входит в final benchmarks.

Все долгие jobs запускались systemd User=nekto; GPU0 только aictl/card lock
и immutable launcher с backup/rollback/down. GPU1/production/up*/m31 не
менялись. Source не коммитился агентом; итоговые commits делает координатор.

## Следующий объединённый v3: source freeze после v5

- GPU routed output теперь caller-owned через `forward_into` в тот же
  `host_reads.moe`; это устраняет повторный allocate/zero больших outputs.
  Complete-cache hit тоже использует его; miss не меняет output.
- `M31_MSA_QUERY_TILE=512` или1024: full QKV/index projections и KV writes
  сохраняются; query slices score→pool/top-k→mask→FA вычисляются последовательно
  и соединяются left-deep concat перед единственным full-batch attention output
  projection/FFN/router. Полный n/histogram экспертов не делится, полный поток
  весов не загружается повторно на каждый query tile. Int32 guard относится к
  score tile. Metadata ограничена128..8192 query rows кратно128,0 выключает.
- При native gather prefill selected/transpose/causal-add mask хранится F16.
  Её значения только0/-inf и совпадают с прежней цепочкой F32→F16. Финальный
  FA mask type тот же; planner учитывает F16 storage, полный causal input и
  full-n projection/output buffers.
- cfg.tap и CPU backend оставляют прежний untiled путь. Общий `ikv` cast и
  cell-block input принадлежат полному graph; ранние tiles не получают dangling
  inputs. Последний query tile1 намеренно остаётся batch-mask веткой, поскольку
  весь layer n>1; отдельный cheap1025/tile512 shape gate подготовлен.
- Источник query tiling просмотрен независимо stream_prefill; настоящие math
  gates и allocator/speed32K ещё обязательны и исполняются координатором.

Counter `host_reads` отражает только wrapper Graph::read_host, а не все физические
копии: CPU fused output memcpy и GPUExperts provider в него не входят. Падение
228→171 calls/2.8→1.4MB при CPU fused нельзя выдавать за дополнительное сокращение
PCIe. Native F16 gather сокращает GPU temporaries/cast kernels; chain устраняет
GPU residual/shared readbacks. Причину каждого эффекта отделяем от счётчика.

## Query tiling не принят; рабочий обход TOP_K_MASK

Объединённый v3 code128 прошёл, но large2968/querytile512 дал top1.578125,
meanKL2.3565. Изоляция на том же frozen ELF с CPU experts/arena0:

- tile512/gather0: тот же FAIL, в том числе при полностью выключенной fusion;
- tile0/gather1 иtile0/gather0:64/64 побитно исправленному full-last oracle.

Это исключает новые WMMA/stream/output-reuse и F16 mask как причину. Multi-query
split остаётся экспериментом, в итоговый preset его включать нельзя. Диагностика:
`port5-query-isolate-v1/`, secondary L0-only probe в `port5-query-l0-v1/`.
Один wrapper job ошибочно назывался job.json и был отвергнут из-за уже занятого
basename backup до изменения aictl; исправлен уникальным именем v2, failed log
сохранён. Все три math arm после исправления реально выполнены.

Рабочий source v6: **M31_MSA_QUERY_TILE=0, M31_MSA_TOPK_MASK=1**. Оставлен
полный index-score GEMM и полный FA с исходной геометрией. Score causal ADD
выполняется inplace (единственный consumer, no taps, n>1). Имеющийся в vendor
GGML_OP_TOP_K_MASK строит сразу финальный F16 mask по causal0/-inf, strided
пикам[top,queries,1,heads] и identity block map128×blocks. Нет materialized
expanded F32/transpose/causal-add copies. На n1 этот флаг граф не меняет.

Native raw top_k — strided view; его переставляем напрямую, reshape к нему
неприменим. Native mask op явно поддерживает все strides и duplicate writes
одинакового значения. Independent source/byte/lifetime reviews exact_kernel
и stream_prefill блокирующих замечаний не нашли.

`codex-m31-port5-topkmask-cpu-v6` rc0: **64/64 побитно CPU oracle**, полный
prefix2968 одним chunk8192 и32 decode, head64/last64, CUDA/F16/CPU80/fusedCPU.
Native gather/adaptive/chain/concat включены, querytile0. Proof:
`port5-topkmask-cpu-v6/compare.json`. Fast build v6 менял только Engine TU и
planner/header, остальные объекты взяты согласованно из frozen v3 family;
для финального GPU/benchmark координатор собирает всю current family coherent.

Planner staged TOP_K_MASK mode использует MAX полной score и mask stages,
а не сумму обоих больших промежуточных buffers. Это позволяет сохранить
слоевой экспертный batch8192 при росте контекста, если реальный allocator
подтвердит бюджет. Готовность32K и полный speed/gate20 здесь не подменяются
аналитической оценкой: эти прогоны выполняются общей финальной конфигурацией.

После независимых reviews canonical init теперь явно отвергает
`M31_MSA_QUERY_TILE>0` и предлагает query0/TOP_K_MASK1. Неисправленный loop
оставлен только для отдельного исследования; frozen v3 diagnostic ELF
по-прежнему позволяет воспроизвести сбой. Принятый final query0 graph от
этого guard и уточнения comments численно не меняется. CPU/CUDA canonical
relink выполняет cpu_executor после source freeze.

## Итоговое изолированное окно M2, два круга

[Полный отчёт окна](/mnt/ai/logs/m31-port/port5-gpu-v4-full-report-v1.md) ·
[Машинная сводка](/mnt/ai/logs/m31-port/port5-gpu-v4-full-summary-v1.json)

Один GPU0, одинаковый om_bench/ref low request, 256 generated, F16 KV,
context40000/maxchunk8192/adaptive, CPU80/fused, CUDA strict CPU-compatible
эксперты, cache7GiB/hot20, ranked next-layer prefetch128, PACK8/host32,
private stream/pinned exact buffers, output reuse, chain/concat/native gather,
SAFE_FUSION1/TOP_K_MASK1, QUERY_TILE0. Ref — записанный лучший1GPU arm
Q8_0 KV/context65536/ubatch2048/3GPU expert layers из m31-ref/REPORT.md;
KV/placement/fusion отличаются, веса GGUF и метод запроса совпадают.
Полные runtime/source/ELF receipts и
сырые запросы сохраняет `port5-gpu-v4-full-report-v1.{json,md}` в logs/m31-port.

| Метрика |4K|16K|32K|
|---|---:|---:|---:|
| Native TG, медиана двух|27.51|26.385|25.44|
| llama.cpp TG|23.33|22.63|21.92|
| Native PP, медиана двух|215.05|231.3|237.9|
| llama.cpp PP|425|не измерен|433|

TG лучше записанного ref на17.9%/16.6%/16.1%. PP4K составляет50.6% ref,
PP32K54.9% ref: цель кратного ускорения префилла НЕ достигнута. M2 частично
успешен по генерации; M3/DSpark в этой сессии не начинается.

Реальный32K allocator gate пройден без OOM; peakGPU30653MiB. Adaptive
планировщик сохранил крупные layer batches:8192×3,7680, затем762/763 tail
в двух запросах. Произвольного дробления всего префилла на query512 нет;
все эти batches исполняют полный original routed histogram. 4K actual chunks:4349 и4348 одним batch;16K:8192/8192/258 и8192/8192/254.
Все request lengths и границы приведены в итоговом bench report.

Итоговый full20 на том же frozen v4 **PASS rc0** после окна
(`codex-m31-port5-final-gate20-v1`):20/20 cases,28 723 positions,
top1=100%, mean/maxKL=0, maxabs0, scope M1-complete-logits; actual geometry
audit подтвердил20 полных cases. GNU byte comparison оформляется отдельно
координатором, numerical equality не подменяет byte proof.
[Итог M1 comparator](/mnt/ai/logs/m31-port/port5-gpu-v4-final-gate20-v1/compare.json).
Canonical source после approved guard/comments
перелинкован CPU/CUDA rc0; Engine SHA
`8811d150292a0581a55d1782aa788773d4de5be8ed15dbe6c6fb88f50996a526`.

## Ограниченный следующий план после M2 partial

Не пытаться скрыть недобор PP спекулятивной генерацией или новым quant.
Следующая крупная работа — resident-device FFN/activation flow, с заранее
заданными промежуточными gates и одним согласованным сравнением.

1. На frozen полной конфигурации разложить FFN wall на CPU quant/OAI,
   weight H2D, activation H2D, gate/up/down D2H и ожидания. Kernel_D2H wall
   включает kernel wait и transfer; нельзя объявлять его чистым PCIe временем.
   Слабое пересечение H2D с kernels в старом v3 trace не доказывает ту же
   величину для v4 с pinned buffers: требуется отдельная проверка очередей.
2. GPU-resident FFN: сохранить исходные Q8_K/Q8_0 activation bytes и CPU
   FP/FMA/OAI контракт, но реализовать device quant/OAI/weighted reduction,
   чтобы gate/up/down и activation vectors не возвращались на CPU между
   стадиями. Сначала per-stage byte proof одного слоя на настоящих inputs,
   затем code128 и3000/full histogram, полный20 только итоговому candidate.
3. Соединить resident FFN с device hidden/residual/shared в префилле,
   оставив CPU только cache placement/router-ID control. Next-layer expert
   queue должна ждать реальные ready/readers events; high-water buffers
   выделяются заранее. Связь очередей и точная стоимость текущих transfers —
   гипотезы для профиля, а не уже доказанная единственная причина недобора.
   После реализации — одно изолированное om_bench4K/16K/32K тем же запросом.

Текущий TOP_K_MASK/inplace путь корректен и выдержал32K; к исследованию
непринятого query loop здесь не возвращались. Первопричина его большого
расхождения пока не установлена и явно изолирована runtime guard.

Полная request-level сводка подтверждает dominant combined интервал:
CPU-quants kernel_D2H wall≈14.6с/4K,50.1с/16K,92.7с/32K. В него входят
queuedH2D/kernel/wait/D2H; активное время kernels отдельно не измерялось в
speed окне. CPU quant≈0.8–1.2с/4K и5.9с/32K, main dense decode растёт
12.445→13.573→14.534ms, CPU MoE21.686→22.118→22.611ms. Поэтому resident
FFN/соединение очередей — проверяемое направление, а не обещание скорости.
Если event profile покажет преобладание active kernel compute, следующая
работа должна улучшать exact-compatible matmul tile/scaler reuse, вместо
перестройки transfer очереди на основании одного combined counter.
