# Port5: review GPUExperts, background prefetch и caller buffers

Проверен source checkpoint `build-port5-cache/candidate-v4/source/` и его
`build-receipt.json`. GPU correction gates: code128 все128 rows и
code3000/c8192/decode32/head64 все64 rows побитно strict full-last oracle.
Кеш7GiB/290slots,seed20,rank128,HOST32,PACK8,PINNED1/STREAM1,CPU80.
Это не автоматическая приёмка других env или изменения source после receipt.

Блокирующих lifetime/alias ошибок в проверенном пути не найдено:

- При reserve защищены все current expert keys, pinned seed и уже имеющиеся
  next-layer entries. Pending slots удалены из maps/free list и повторно
  не выбираются. Для128current+128next+20hot достаточно276slots; cache290.
- Background worker получает только immutable source descriptors и disjoint
  destination slots. Он не меняет main cache map/clock/scales. Metadata
  публикуется main caller только после успешного wait.
- Partial DMA failure дренируется до metadata reset. Main forward error
  очищает entries/pins/free slots и пересоздаёт uploader; неподтверждённые
  bytes не превращаются в cache hit. Failed env seed делает init terminal.
- Destructor сначала join background, затем main ring drain, затем освобождает
  GPU pool/slot metadata. Arena/backend заимствованы и живут дольше GPUExperts
  в Engine и в direct fixture. Copy pool переживает все callback/ring users.
- Общий CpuPackSpans pool вызывается main и background последовательно:
  main заканчивает свои uploads до запуска background, а join заканчивается
  до следующего main load. Callback synchronous и не хранит borrowed spans.
  Quant/OAI executor отдельный; callbacks не вызывают другой executor.
- CPU-compatible CUDA stream после upload initial weights получает уже готовые
  current slots. Next DMA не пишет ни одного current slot. AsyncDrain в
  GpuCpuQuants дренирует stream раньше локальных dispatch/tiles vectors;
  host readback используется только после собственного stream sync.
- `forward_into` запрещает alias output с x/weights и сохраняет caller capacity.
  Exact и stock paths оба пишут весь logical output. Failure требует discard
  и Engine reset. Cached miss оставляет output и cache metadata без изменений;
  hit использует retained capacity, для alias input оставлен local fallback.
- NVFP4 external scales не зависят от DMA tail: каждый forward заполняет их
  из immutable layer metadata по current expert→slot mapping. Порядок router
  IDs, полный histogram и rank0→3 reduction не менялись.

Actual runtime logs подтверждают HOST32, private stream и PINNED active,
171 calls/0fallback для large prefix. Результаты других компиляторов/CPU,
HIP/SYCL, CUDA fatal-context recovery и большой memory pressure этим review
не заявляются. Итоговый full20 того же V4 ELF принят координатором:
20/20,28,723 позиции,top1=100%,mean/maxKL0,maxabs0;F16/ctx4096/chunk2048/
CPU80/decode32/cache7,all finalenv. Performance maxchunk8192 отдельно
проверен на real3000/head64 и полном om_bench4K/16K/32K. GNU cmp и
canonical build validation хранятся в общем финальном отчёте.

Physical prefetch overlap измерен, но невелик: два независимых Nsight analyzer
совпали на627 H2D intersections/339 routed kernels,union176.816ms (~4.9%).
Короткий late join не доказывает полного скрытия transfers: waits способны
попасть внутрь другого CUDA API. Следовательно по этим данным нельзя обещать
префилл быстрее llama.cpp; решают итоговые om_bench numbers.

## Следующий конкретный performance experiment

Без изменения weights/ELF сравнить текущую unset-конфигурацию с
`CUDA_DEVICE_MAX_CONNECTIONS=32 CUDA_DEVICE_MAX_COPY_CONNECTIONS=32`.
Для реального v4 worker PID482696 оба env и CUDA_LAUNCH_BLOCKING были
проверены через whitelisted `/proc/PID/environ`: unset. Никакой runtime
конфигурации в текущем full benchmark не меняли.

Гипотеза: ложные зависимости между CUDA stream connections или scheduling
copy queues участвуют в длинном up/readback wait. Это **не доказано**.
Установить env до CUDA init в новом immutable aictl job на том же v4 ELF,
сначала пройти code128/large gates, затем одно изолированное окно warmup+2×4K
тем же om_bench и CUDA/Nsight overlap capture вне speed окна. Сравнить
wall PP, actual kernel/DMA intervals, blocked API, а не только late join.
При отсутствии эффекта не включать env в performance preset.
