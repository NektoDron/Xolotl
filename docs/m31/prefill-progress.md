# M31: pipeline загрузки экспертов и выбор chunk

2026-10-26, направление `streaming_prefill`. Новые файлы, opt-in интеграция;
арифметика GPU-экспертов и основной engine этим агентом не менялись.

`tools/m31/prefill_pipeline.hpp` содержит header-only `PrefillUploads`.
Два буфера pinned host memory выделяются один раз при создании. Отдельный
backend того же GPU выполняет H2D. Пока он читает предыдущий буфер, CPU
пакует исходные `[gate|up|down]` байты следующего эксперта во второй.
Весь `slot_stride`, включая промежутки и quant padding, сначала обнуляется.
В `enqueue` нет выделения host/device memory. На CPU и backend без async/events
используется синхронный fallback с тем же byte layout.

Порядок использования:

```cpp
PrefillUploads uploads(backend, slot_stride, 2);
uploads.begin_batch();
// Для каждого cache miss, offsets/raw_bytes/slices принадлежат исходному Layer:
uploads.enqueue(slot_proxy, {
    {gate_offset, gate_source, gate_bytes},
    {up_offset,   up_source,   up_bytes},
    {down_offset, down_source, down_bytes},
});
uploads.finish();
// Далее исходный полный layer execute, без изменения n/used/histogram.
```

`begin_batch` нужно вызывать после постановки всех GPU consumers предыдущего
батча на borrowed compute backend. Он записывает event окончания readers и
заставляет transfer stream ждать, прежде чем перезаписывать VRAM-слоты.
Ring reuse ждёт отдельный copy event, защищающий pinned staging от CPU.
`finish` записывает ready event и ставит wait на compute backend; host не ждёт
последней передачи. Все destination tensors и compute backend должны жить
дольше uploader. Его destructor синхронизирует оба backend перед освобождением.
После исключения внутри открытого batch uploader следует пересоздать.

Важное условие текущего прототипа: `GpuCpuQuants` использует raw CUDA default
stream. Он не получает wait из ggml compute stream автоматически. Для этого
пути **после `finish` нужен `synchronize` перед execute**. Обычный ggml execute
на borrowed backend упорядочивается самим `finish`.

Это перекрытие CPU packing и H2D. Загрузка всех выбранных экспертов завершается
до полного layer execute; перекрытие загрузки с вычислением MoE здесь не
реализовано. Subset batching нельзя добавлять, изменяя формы `[used,n]` обычного
GPU `mul_mat_id`. В CPU-compatible CUDA dispatch необходим histogram исходного
полного батча: >=8 назначений эксперту выбирают другой FP accumulation path.

## Планировщик

`prefill_plan.hpp` принимает **реально свободную VRAM после** выделения dense,
полной KV capacity и fixed expert pool. Эти resident allocations повторно не
вычитаются. Из свободного объёма вычитаются явные reserves и оцениваются MSA
score/mask/index, dense и routed temporaries. Результат — proposal до2048 строк,
обычно кратный block128; short tail разрешён, KV pad всегда256 как в M1.
Дополнительно ограничивается худшее объединение маршрутов
`min(experts, used*n)`: маленький cache не должен менять router ради размещения.

Это консервативная оценка, **не измеренный peak allocator**. Новый chunk требует
успешного ggml reserve и same-chunk oracle gate. Runtime-интеграция планировщика
в engine пока не сделана.

Примеры по настоящему GGUF (`prefill-cpu-check-v1.json`), GPU32110MiB,
F16 main/index KV,248 slots ×25909632B,1GiB reserve+256MiB extra scratch:

| Context | Position | KV capacity | Chunk proposal | Scratch estimate | Остаток после reserves |
|---|---:|---:|---:|---:|---:|
|4K|2048|566231040B|2048|2216689664B|12439294464B|
|32K|16384|4529848320B|2048|3868196864B|6824169984B|
|64K|32768|9059696640B|2048|5755633664B|406884864B|

У64K это менее0.4GiB headroom даже в этой оценке; allocator gate необходим.
Q8 KV не реализован этим модулем.

## Проверено

- Отдельная link-only сборка `build-streaming-prefill`, только1C++TU, без
  пересборки vendor. Финальный unit `codex-m31-port4-prefill-build-v3` rc0.
- Чистое CPU-only плечо без CUDA link: `build-streaming-prefill/cpu-only`,
  `codex-m31-port4-prefill-build-cpu-only-v1` rc0;
  `prefill-cpu-only-check-v1.json` повторно прошёл51byte outputs и7planner checks.
- CPU fallback:17батчей по3 VRAM/RAM слота с немедленным повторным использованием
  source arrays и этих же destination slots. Каждый батч ставит отдельный
  consumer copy graph; сравниваются все51outputs,208896байт, включая zero padding.
  Проверяются malformed offsets/overlap и7случаев planner, включая отказ при
  недостатке cache slots и overflow. `prefill-cpu-check-v1.json`:PASS.
- Настоящий CUDA event path через aictl/card_session, толькоGPU0:
  `codex-m31-port4-prefill-events-v1` rc0,
  `prefill-gpu-check-v1.json`:51outputs/208896байт побитно, pinned=true,
  asynchronous=true,49ring reuses, staging8192байт. Consumers ставились без
  host synchronization между батчами, поэтому проверялся overwrite hazard.
  down+точечный rollback завершились; GPU1 не занималась.
- Логи `/mnt/ai/logs/m31-port/prefill-build.log`,
  `job-prefill-events-v1*`; backup/rollback строка в session log.
  Первая попытка compile получила namespace error в probe, исправлено;
  неуспешный лог сохранён. Измерений t/s этого модуля нет.

## Дальше

Подключить header opt-in к `GpuExperts`: uploader создаётся после slot pool,
`begin_batch`/`finish` вокруг исходного load loop; cache miss вызывает enqueue
трёх исходных slices. Прежний путь остаётся default. Старый однобуферный staging
при включённом uploader выделять не требуется. Сначала byte/code128 gate и
проверка сброса/ошибок, затем gate20 и отдельное свободное окно скорости.
Ни pool layout, ни metadata external NVFP4 scales не меняются.
