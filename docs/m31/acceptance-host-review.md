# Независимый review host stages и повторного использования readback

26.10.2026, cache_ranking, перед candidate benchmark. Код не изменялся.
Проверены CPU/GPU host executor, quant/OAI helpers, wiring в exact-CUDA path
и `M31_REUSE_HOST_READS`. **Blocking P0/P1 для текущего inference path не найдено.**

## Неблокирующая находка

**P2, косвенная reentrancy двух executor может зависнуть.**
`cpu_gpu_host_executor.cpp:14–24` хранит только последний `active_executor`.
Путь A callback → B.parallel_rows → B callback → A.parallel_rows обходит
проверку `active_executor == this` на `cpu_gpu_host_executor.cpp:132–134`:
последним активным становится B, затем повторная A ждёт уже удерживаемый
`A.submission`. Это статический lock-path finding, воспроизводящий тест не
запускался.385 fixtures проверяют только прямую reentrancy на A.

Текущие callbacks `cpu_gpu_host_ops.cpp:41–44,59–67` вызывают только исходный
quantizer или scalar OAI и не подают работы другому executor. Поэтому finding
не блокирует данный benchmark. Перед расширением публичного API стоит
запретить cross-executor nesting документированно либо отслеживать весь
thread-local стек активных executor. В этом review правка не выполнялась.

## Что подтверждено чтением кода

- **FP environment:** `cpu_gpu_host_executor.cpp:86–111,145–175` применяет
  caller fenv/MXCSR к каждому range, RAII восстанавливает control bits при
  исключении, стандартные sticky flags и все6 MXCSR status bits объединяются
  после завершения workers. Scalar fallback не меняет FP environment.
- **Lifetime/exceptions:** `cpu_gpu_host_executor.cpp:121–129,149–175`
  удерживает callback до `pending == 0`, очищает pointer только после drain,
  затем rethrow. `shutdown` на178–185 сериализован с submissions и join всех
  threads. Конструктор при частичном thread-create failure тоже join.
- **Concurrent submissions:** `cpu_gpu_host_executor.cpp:132–135` удерживает
  submission mutex на весь job, state mutex защищает callback/count/generation.
  Direct reentrancy/close из callback отклоняются до попытки этого mutex.
- **Default без32 workers:** `gpu_cpu_quants.cu:134–139` принудительно ставит
  `threads=1`, если `M31_GPU_HOST_THREADS` отсутствует. Конструктор executor
  создаёт `threads−1` workers (`cpu_gpu_host_executor.cpp:73–76`), то есть0.
  CPU-default Engine не создаёт `GpuCpuQuants`: объект существует только в
  opt-in GPU cache/exact-quants ветке `m31_gpu_experts.cpp`.
- **Исходная serial math:** quant helper вызывает тот же
  `traits->from_float` по полным независимым строкам (`cpu_gpu_host_ops.cpp:41–44`).
  OAI выражение на62–64 дословно совпадает со старым loop в
  `cpu_gpu_host_serial_reference.json`: min(gate,7), clamp(up,−7,7),
  `a/(1+expf(1.702f*(-a)))*(b+1)`. Route weights и reduction здесь не меняются.
  CMake задаёт host sources `-frounding-math;-fno-fast-math` отдельно от CUDA.
- **Readback alias/lifetime:** `m31_engine.cpp:307–324` хранит раздельные
  private vectors input/residual/shared/weights/moe/ids. F32 read_into на224–227
  resize и синхронный tensor_get; все reads на652–676 закончены до освобождения
  temporary graph на680. CPU MoE пишет в отдельный moe vector на689–691,
  GPU MoE сохраняет owned-vector API на693–695. Hidden input уже скопирован
  в graph до его resize/readback; host vectors не являются ggml weight buffers.
- **Reset/taps/default:** opt-in требует env=`1`, reuse-allocations и отсутствие
  taps (`m31_engine.cpp:766–768`). `reset` на826–828 сохраняет capacity, но новые
  input/IDs/weights/readbacks полностью overwrite текущие logical sizes. Caller
  logits остаются owned output (`m31_engine.cpp:917`), не ссылкой на host scratch.
  Без env остаются локальные vectors и Graph::read, taps идут прежним путём.

## Проверенные артефакты и границы

В этом review **не пересобирались** исходники и **не запускались**150GB model,
GPU,385-case suite или host-read fixture заново. Выполнены только чтение кода,
разбор сохранённых результатов и SHA256 небольших fixtures/source.

- `cpu-gpu-host-build-20261026T195111Z.log`: сохранённый PASS385, threads1/16/32/64,
  n1/7/128/2048, synthetic+реальные input/gate/up/OAI. FP suite включает4 rounding
  modes, FTZ/DAZ, sticky flags; lifecycle — exception recovery, disjoint ranges,
  concurrent submissions, direct reentrancy, close во время job. Все4 fixture
  SHA256 совпали с `cpu_gpu_host_serial_reference.json` при этом review.
- `cpu-host-read-check-v1.json` и его build log: сохранены bitwise comparisons
  для CPU residual0/1, taps fallback, no-reuse fallback, shape128→2048→1→7,
  reset/recovery и owned прежних logits. Engine source SHA256 сейчас точно
  совпадает с записанным в fixture report. Capacity evidence: после первого
  grow повторный2048 и decode имеют0 growth.
- `gpu-candidate-probe-host32-accept-v1.json`: сохранённый PASS36 cases/117 stages,
  21,307,392 F32 values bitwise с CPU oracle. Это проверка экспертных stages,
  не всего M1 prompt suite.
- `gpu-candidate-code128-combined-compare-v1.json`: сохранённый PASS128 rows,
  top1=1, mean/maxKL0. Полный frozen gate20 относится к своей замороженной
  конфигурации; этот review не распространяет его автоматически на новую
  комбинацию host32/read-reuse и не объявляет новый полный gate завершённым.

Source SHA256 на момент review:

| Файл | SHA256 |
|---|---|
|`cpu_gpu_host_executor.cpp`|`4e9da8d4c8c9d6ccba2a5026ea29de7bf7f44f03a8a676583d91be07c878ddfb`|
|`cpu_gpu_host_ops.cpp`|`d7c38eb109a22dbad2ad93543c74b844c9e7b1ffa0eccec4c2723bab20df5ce3`|
|`m31_engine.cpp`|`0ca279404cf22ed98b942eca2f9d22e24ccbcb61b7d8682875160a01e2930ec2`|
|`gpu_cpu_quants.cu`|`d4d93ee10c08de6e718b67ebbc3aad63e56fae31dac28aa6db3754053385b81d`|
