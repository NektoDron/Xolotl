# CPU-направление port5

Основное реализованное ускорение — `M31_CPU_FUSED=1`: один FFN custom node
на существующем persistent ggml pool вместо отдельных graph nodes для каждой
операции. Исходные CPU vec_dot/quant traits, external NVFP4 scales, scalar OAI
и последовательная F32 сумма top-k ranks сохранены. Пять barrier фаз разделяют
quant input, GU, OAI, quant down и down перед итоговой суммой. Второго worker
pool у fused пути нет.

Отдельный CV direct путь оказался медленнее и оставлен выключенным;
его результаты не используются как положительный M2 результат.

## Компоненты и реальный benchmark — разные результаты

На пяти реальных слоях3/10/52/58/59, четырёх исходных экспертах и переставленных
ID/weights fused совпадает побитно с ggml CPU graph, включая NVFP4 scales.
В изолированном warm component при64 потоках время снижено на7.6–17.4%.
Отдельный probe без CV direct дал средние времена по пяти слоям:

| Потоки | Fused, мс |
|---:|---:|
|64|0.342903|
|80|0.292626|
|96|0.867050|

Все15 случаев побитны CPU graph.80 быстрее64 на14.66% в этом component;
96 дал сильную регрессию, причину не установили. Affinity default не менялся.
Engine разрешает fused80/96; ограничение64 относится к отрицательному CV direct.
Для полного benchmark выбран `--threads 80`.

Полный v4 benchmark завершён rc0: warmup и по два4K/16K/32K,256 generated,
тот же om_bench и reasoning_effort=low. Медианы:

| Depth | Native TG | Ref TG | Прирост TG | Native PP | Ref PP |
|---|---:|---:|---:|---:|---:|
|4K|27.51|23.33|17.92%|215.05|425|
|16K|26.385|22.63|16.59%|231.3|—|
|32K|25.44|21.92|16.06%|237.9|433|

Это совместная конфигурация GPU dense/CPU fused/prefill, поэтому её прирост
нельзя целиком приписать CPU80 или fused. Генерация обогнала reference;
prefill остаётся на49.4%/45.1% ниже ref для4K/32K. M2 частичен, цели кратного
PP выигрыша нет. GSM100/M3 в этой сессии не запускаются.

Средние decode stage counters двух запросов, мс/токен:

| Depth | CPU MoE compute | Dense total | Dense enqueue | Dense read/completion |
|---|---:|---:|---:|---:|
|4K|21.686|12.445|0.925|11.171|
|16K|22.118|13.573|0.866|12.366|
|32K|22.611|14.534|0.887|13.319|

CPU setup≈0.006ms. Read counterCPU равен0, поскольку memcpy в caller-owned
output находится внутри fused forward/compute; это не отсутствие CPU копии.
Dense enqueue — host submission, read/completion включает ожидание async
GPU работы и readback; эти counters не разлагают чистое время GPU kernels/D2H.

Actual config: CPU80/HOST32/PACK8, affinity unset, F16 main/index KV,
context40000, requested chunk8192, expert cache7GiB/290slots.32K реально
использовал8192/8192/8192/7680/762 либо763 токена. Runtime counters:
1083 GPU forwards=19 prefill chunks×57,3249 multiply calls=3×1083,
5415 host parallel jobs=5×1083,167865 worker ranges=31×5415, serial jobs0.
GPU cached decode0; pinned calls3249/fallback0/host_quant_bytes0. GPU peak
30653MiB, сумма RSS320.36GiB (с файловыми mappings, не unique physical RAM).

Числа и counters: `/mnt/ai/logs/m31-port/port5-gpu-v4-full-report-v1.json`
и `port5-gpu-v4-full-summary-v1.json`.

Финальный full20 завершён rc0:20/20,28723 позиции, top1=100%, mean/max KL0,
scope `M1-complete-logits`. Отчёт:
`/mnt/ai/logs/m31-port/port5-gpu-v4-final-gate20-v1/compare.json`.
Это20промптов64–3000 токенов с финальными flags CPU80/HOST32/PACK8/pinned;
speed результаты длинных16K/32K не выдаются за отдельную oracle сверку logits.
Независимую GNU cmp проверку raw bytes выполняет bench агент.

## Упаковка и allocation

`CpuPackSpans` использует существующий bounded executor и независимые1MiB
stripes. Один job обслуживает все role spans эксперта, завершение синхронно
предшествует H2D. Новых weight/staging allocations helper не создаёт.

Byte gate25MiB с полными guards/reuse прошёл для1/4/8 потоков. Отдельный
pageable DRAM component: source1GiB/40 rotating segments больше aggregate L3,
две destination области по25MiB,3.355GB копирования на плечо. Получены
21.12/86.15/134.51GB/s для1/4/8 потоков, выбран PACK8. Это не pinned H2D
и не end-to-end PP; реальное окно PP остаётся ниже reference425–433.

По source audit добавлен caller-owned `GpuExperts::forward_into` и reuse
`host_reads.moe` вместо нового≈73/201MB zero-filled result на каждом слое
при2968/8192 токенах. В large correctness diagnostic v2→v3 reduce time
2179.325→258.929ms (−88.12%,−1.920s), но это combined diagnostic:
изменились также stream/pack; общий PP из этого процента не выводится.
Одновременно H2D wall вырос392.070→3186.058ms при почти неизменном kernel
event3639.730→3620.425ms. Именно после этого проверяется pinned v4;
причина роста не приписывается конкретной внутренней операции CUDA без trace.

## Конфигурация и приёмка

- Actual executor logs показывают threads/workers/threshold/grain и CPU affinity;
  GPU profile показывает serial/parallel jobs. Known env typos отвергаются.
  В прежнем correctness job ошибочное HOST имя оставляло serial1; подготовленные
  server benchmarks использовали правильное `M31_GPU_HOST_THREADS=32`.
- Canonical CPU build rc0 и CUDA sm120a/CUDA17 build rc0. В build.sh теперь
  explicit CPU_QUANTS_CUDA ON/OFF и jobs24. После guard/comments CPU+CUDA
  Engine TU rebuild/relink также rc0; численная q0 ветка не менялась.
- Comparator4/4, tiny NumPy1/7/29 maxabs0.00011285 и arena/workspace/pool/
  residual parity пройдены.14 protocol tests и8 real vocab/tokenizer tests
  прошли; реальные vocab tests выполнялись с CUDA_VISIBLE_DEVICES пустым.
- Canonical GPU inference не запускался отдельно: numerical gates и speed
  относятся к immutable согласованным candidate ELF. Архивирование больших
  canonical ELF и hashes во время speed window не выполнялось.
- Independent private-stream/pinned/CpuFused lifetime review не нашёл blockers;
  конкретные source SHA и ограничения — `port5-stream-lifetime-review.md`.

Все CPU-направления systemd units завершены. GPU, aictl config, miners и prod
этот агент не менял. Коммиты выполняет координатор. Подробные filenames,
ELF/source receipts и хронология — `port5-cpu-progress.md`.
