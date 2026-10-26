# Private CUDA stream и CpuFused: независимый review

26.10.2026 UTC, агент cpu_executor. Read-only review исходников ниже;
новых runtime тестов в рамках review не запускалось. Blocking ошибок не найдено.

## Snapshot SHA256

```text
d193a10b17e7c243b9b2b8c3c68c2b2ce47ad6a316f342a83c99f3d1e2d99d76  tools/m31/gpu_cpu_quants.cu
89311416cc517308c09211d413445395f33ca7c0d4023c90766dfb659693eba5  tools/m31/m31_gpu_experts.cpp
cfbcdf67505b6f0acd60845a5101d5355faebd92445cbd196ae909107f616d06  tools/m31/m31_engine.cpp
57fa17390bfbd6c388a239d0fbd607b0c595f7be987f033d4e15fe5857030ec2  tools/m31/cpu_fused_decode.hpp
adb3e925ac2b9eb5d4eb060b8e42b89da237c477dda0d9342e0baa0d163ed7dd  tools/m31/prefill_background.hpp
50dd4fb89f2c141541256bdcb3987e993632d57b4a1cc845de75c5303e411f3f  tools/m31/prefill_pipeline.hpp
```

## CUDA ownership и порядок

- AsyncDrain объявлен после всех local upload sources: dispatch_ids,
  tiles, tensor_tiles и scheduled_routes. При исключении его destructor
  завершает private stream до уничтожения этих vectors. Packed activation
  принадлежит Impl и живёт ещё дольше.
- `pending=true` ставится до async API. Positive quant upload предшествует
  любому kernel launch. Failed enqueue/launch/read поэтому также проходит drain.
- H2D, kernels, timing events и D2H идут в одном private stream. Normal read
  ждёт этот stream до возврата caller-owned output. Timing events уже готовы
  после этого ожидания; отдельный default-stream event wait не требуется.
- Все scratch reserves выполняются до enqueue текущего multiply. Предыдущий
  multiply синхронно закончен. `cudaFree` при росте high-water scratch может
  дать глобальное ожидание; оно не создаёт use-after-free.
- Impl destructor выполняет drain, затем device frees, затем event/stream
  destroy. В этот момент source buffers и host executor ещё принадлежат Impl.
- Исходные expert weights остаются живы: uploads.synchronize перед execute
  сохранён. Next-layer prefetch резервирует disjoint slots и защищает current
  expert keys; background join/drain предшествует освобождению pool/arena.

## CpuFused

- GU/OAI/down/sum ranges полностью покрывают выходы без пересечений writers.
  Каждая quant row записывается один раз. Пять ggml_barrier используют тот
  же persistent pool и actual n_graph/nth, что исполняет custom node.
- Shapes, ID bounds и unique top-k проверяются до compute. Векторы выделены
  заранее; callbacks не выделяют память и не бросают C++ exceptions.
- CPU vec_dot nrc1, quant traits, external scales, scalar OAI и rank0→rank3
  F32 сумма сохраняют исходный контракт. Component tests отдельно доказали
  побитность пяти реальных слоёв; это не заменяет integrated gate20.
- Engine освобождает fused helper до pool/backend/weights. Reset может
  переиспользовать graph: userdata обновляется перед каждой синхронной работой,
  graph не хранит старые input vectors как GGML source tensors.

## Ограничения вывода

Review относится к serialized owner и текущему single-device runner GPU0,
static pinned ggml/CUDA13.1/gcc14. Internal ggml_barrier привязан к этому
vendor; shared-library/другие backend/compiler здесь не проверялись.
Pageable cudaMemcpyAsync может блокировать host; из наличия nonblocking
stream не выводится фактическое перекрытие или прирост t/s. Числа должны
приходить из отдельных code128/large gates и изолированных om_bench окон.
Сохранность численных значений output после error не обещается: Engine
poison/reset контракт сохраняется, ownership и асинхронный lifetime разобраны выше.

## Дополнение: pinned quant/readback v4

Read-only review дополнительных исходников:

```text
9c271468cb2f7b43895a6f106f276b7a51f3559f815d2f4d11563c4e71fef778  tools/m31/gpu_pinned_scratch.hpp
5867ce4164c05fb0945463542ab6ac00959aef9394c48c79e836e9516f5e4ee7  tools/m31/gpu_cpu_quants.cu
```

Blocking ошибок не найдено. Сумма двух page-rounded high-water capacities
ограничена1GiB; stream drain предшествует grow/free, free-before-allocate
сохраняет лимит и для промежуточного состояния. Upload rings в этот scratch
лимит не входят и имеют отдельный небольшой бюджет.

Budget refusal оставляет старые buffers; ожидаемый HostAlloc failure очищает
их и выключает повторные попытки. Иные ошибки CUDA пробрасываются. При
активном пути quantizer пишет непосредственно в pinned buffer, pageable
quant staging не растёт и не копируется. Pinned D2H завершается на ownstream
до CPU memcpy в caller-owned result. Pinned object уничтожается до borrowed
stream; constructor failure также освобождает stream. Исходный AsyncDrain
перед local-vector destruction сохранён.

Фактические active/fallback GPU probes выполняет exact_kernel отдельно.
Из source review не выводится успешность allocation-failure injection или
прирост производительности.
