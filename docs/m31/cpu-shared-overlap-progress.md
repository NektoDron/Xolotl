# CPU routed MoE и GPU shared FFN: opt-in overlap

26.10.2026, port4. Изменён только `m31_engine.cpp` и новые
`cpu_shared_overlap_*` tests/build tools. Config/header ABI, GPUExperts, CUDA
kernels и frozen full-gate binaries не менялись. Source frozen:
`c797b6e94f4fd961486354461f54857488dce586d145c66f1021535c249216c4`.
Независимый lifetime review cache_ranking: blockers не найдено.

## Условия и порядок

`M31_CPU_SHARED_OVERLAP=1`, default OFF. Init разрешает режим только на CUDA
backend при reuse_allocations + cpu_residual + dense_decode_cache и без taps.
Неподходящий Config/CPU backend безопасно оставляет прежний путь. На каждом
слое отдельно проверяется n=1, routed layer и фактическое CPU placement: если
GPUExperts включён для decode порогом1, overlap не применяется.

Только в подходящем n1 main dense graph вычисляет attention/residual/norm/router,
а shared FFN исключается из него. Отдельный persistent shared graph каждого
слоя имеет собственные context, allocator, scratch, F32 input и output. Norm
копируется прежними F32 bytes через host и synchronous tensor_set; shared graph
не содержит views/pointers на attention scratch или общие AttentionInputs.

Порядок: main dense compute/read → shared graph async submission → прежний
CPU MoE graph → synchronize/read shared → прежние `(moe+shared)+residual` F32
сложения. `ggml_backend_graph_compute_async` напрямую вызывает backend compute;
в отличие от обычного wrapper, общий synchronize сам не добавляет.

## Lifetime и failure

- Inflight flag выставляется до async API, поскольку failed submission может
  уже поставить часть работы в очередь.
- Локальный SharedDrain действует до выхода из слоя и синхронизирует backend
  при исключении; context/input/scratch/weights всё это время принадлежат Engine.
- CPU MoE использует CPU backend. Его Graph destructor CUDA-cache clear является
  no-op для CPU, поэтому не освобождает CUDA captures во время shared compute.
- После CPU MoE выполняется join до shared read и F32 residual суммы.
- Shared graph не пересоздаётся на каждом токене и не сбрасывает CUDA captures.
  Его shape не зависит от nk. Смена AttentionInputs/nk может оставить этот graph,
  поскольку input/scratch у него отдельные. CUDA caches main graphs чистятся
  только после drain; shared capture затем при необходимости прогревается заново.
- reset сначала drain, затем освобождает shared/dense/head graph ownership.
  Destructor drain выполняется до graph buffers, weights и backend cleanup.
- Compile-only failure hook проверяет исключение после shared enqueue, poisoned
  state, reset и точный повторный forward. В обычном server build hook отсутствует.

Default graph operations, CPU routed arithmetic и ownership logits не меняются.
`M31_CPU_SHARED_OVERLAP_TRACE=1` пишет host timestamps enqueue/CPU begin/end/join;
для speed окна TRACE должен оставаться0. При profile выводятся отдельные
shared enqueue/wait/read counters; это не следует складывать как последовательное
GPU+CPU время, поскольку часть работы перекрывается.

## Проверки точности

`codex-m31-port4-cpu-shared-overlap-build-v1` rc0: CPU backend safe fallback,
context/reset/taps/no-dense-cache побитны; независимый NumPy tiny chunks1/7/29
и arena/mmap checks проходят с прежним maxabs0.00011285 против NumPy.

`codex-m31-port4-cpu-shared-overlap-correct-v1` **rc0**, штатный immutable
card_launch/aictl/card lock, down+rollback завершены:

- CUDA supported tiny HD64/block128: flag0/1 logits побитны, включая transitions
 256/512, n128→n1, reset и сохранённые caller-owned outputs.
- Taps, отсутствие dense cache и GPUExperts decode placement: fallback побитен.
- Исключение после async shared submission → poisoned Engine → reset/recovery
  даёт прежние logits; stale scratch/inflight use не наблюдается.
- Настоящая модель, все3000 позиций, prefix2968/chunk128 +tail32/chunk1:
  baseline и overlap оба **ПОБИТНО CPU oracle**, top1=1/KL0/maxabs0.
- Отдельное сравнение baseline/overlap: также все3000 rows/maxabs0.

Raw reports в `/mnt/ai/logs/m31-port/`:
`cpu-shared-real-base-compare-v1.json`, `cpu-shared-real-overlap-compare-v1.json`,
`cpu-shared-real-compare-v1.json`. Это целевой decode regression gate; новый
полный suite20 для overlap в этом этапе не объявляется.

## Реальное event proof, не benchmark

После math gate выполнен короткий Nsight Systems CUDA/NVTX trace на code128,
prefix120 +tail8. Proof-only build добавляет NVTX range вокруг CPU-routed вызова;
обычная сборка этих вызовов не содержит.

`codex-m31-port4-cpu-shared-overlap-events-v1` rc0, cleanup выполнен.
SQLite kernel activity действительно пересекает CPU MoE NVTX interval:

-456 CPU-routed ranges (57слоёв×8tokens);
-455 из456 имеют пересечение с GPU kernel activity;
-суммарный union пересечений внутри CPU ranges15,148,077ns.

В примерах видны shared `mul_mat_vec_q`, `swiglu_oai_kernel`, `quantize_q8_1`.
Main attention завершён до этих CPU ranges, GPUExperts не используется, поэтому
это активность отдельного shared graph. Scope — CPU-routed вызов целиком,
включая его setup/readback; per-instruction overlap конкретного CPU dot здесь
не измеряется. Nsight overhead не позволяет считать эти числа TG приростом.

Артефакты: `cpu-shared-overlap-events-v1.nsys-rep` (2.5MiB), одноимённый SQLite,
`cpu-shared-overlap-event-proof-v1.json` с конкретными start/end/intersection
примерами. Analyzer: `tools/m31/cpu_shared_overlap_events.py`.

## Обычный согласованный server build для окна

`codex-m31-port4-cpu-shared-overlap-serve-build-v1` **rc0**. Generate и serve
собраны вместе из того же frozen source; CPU tiny/NumPy/arena gate повторён.
В compile flags проверено отсутствие M31_SHARED_OVERLAP_NVTX/TESTING; только
M31_ENGINE_CUDA=1. Прежний proof ELF сохранён отдельно до relink:
`cpu-shared-overlap-proof-m31_generate-v1` и SHA receipt.

Бинарники:

- `/mnt/ai/Xolotl-m31/build-cpu-shared-overlap/m31_serve`
  SHA256 `f53d9a9247623d4d78dfedce583a6540b5e8de5ae69425b2f6ce7c0cc3d6448d`.
- Согласованный `m31_generate`:
  SHA256 `00f972c36f76ef31e9fcb0ad6b6435c5c22eee517a32551cf5e052e7fce52a14`.

Полный freeze receipt: `cpu-shared-overlap-serve-freeze-v1.json`.
Обычная сборка: `tools/m31/cpu_shared_overlap_serve_build.sh` (уникальные artifact
paths/unit names при повторе); proof build включает instrumentation явно через
`cpu_shared_overlap_build.sh` и не должен запускаться в окно замера.

Для paired CPU-routed speed: arena0/GPUExperts0, F16 ctx8192/chunk2048/threads64,
обе decode caches, reuse allocations, CPU residual, M31_REUSE_HOST_READS=1,
M31_CPU_SHARED_OVERLAP=1, TRACE0, nofusion и точный OM_OFF reasoning_effort=low.
Изолированное окно организует координатор. До его результата прирост скорости
над21.3/llama.cpp23.33 не заявляется.

## Заключительное изолированное CPU окно: фактическая скорость

`codex-m31-port4-cpu-overlap-window-v1`, frozen обычный serve без NVTX/failure hook.
Никаких сборок/тяжёлых CPU задач во время плеча. Тот же om_bench.py, warmup и
два4K, `OM_OFF={"reasoning_effort":"low"}`, nofusion, CPU routed64threads,
F16 ctx8192/chunk2048, arena0/GPUExperts0, обе decode caches и host reads1/shared1.

| Путь | TG4K, два прогона | PP4K, два прогона |
|---|---|---|
| Раннее strict-low/быстрый ACK, до host buffers/shared overlap |21.34 /21.31|124.5 /124.6|
| Host buffers + shared overlap |22.38 /21.93|181.2 /185.4|
| Лучший1GPU llama.cpp ref |23.33|425|

Среднее новое TG22.155: +3.89% к21.325 предыдущего strict-low плеча, но ниже
ref на5.04%; лучший22.38 тоже нижеref на4.07%. PP183.3: +47.2% к124.55,
но ниже425 на56.9%. Время prefill снизилось с≈34.9с до24.0/23.4с.
Два запроса имеют разные UUID и несколько отличающиеся prompt lengths/routes;
это два4K прогона, не статистическое доказательство малого TG эффекта.
Ни один CPU прогон не превысил reference.

Это combined изменение: shared overlap применяется только к n1 и **не может
объяснять PP рост**. Основной доказанный механизм префилла — reuse host vectors:
на chunk2048 dense_read стал≈0.49с против≈4.2с, CPU routed_read≈0.24с против
≈1.44с до изменения. Копируемые bytes и арифметика сохранены.

Decode profile, среднее мс/токен по255 шагам:

| Этап |Прогон1|Прогон2|
|---|---:|---:|
| Main dense |15.525|15.546|
| CPU MoE setup |0.246|0.250|
| CPU MoE compute |25.855|26.729|
| CPU MoE read |0.093|0.096|
| Shared enqueue |0.785|0.792|
| Shared wait после CPU |0.072|0.071|
| Shared D2H |0.506|0.505|
| Head setup+compute+read |0.709|0.713|
| Sampling/call |0.342|0.350|
| Interactive ACK/wait |0.184|0.177|

До split main dense был≈18.25–18.32мс, но отдельные enqueue/join/read shared
теперь стоят ещё≈1.36мс. Поэтому скрыта только часть shared работы, а не весь
attention/indexer/dense путь. API accounting remainder≈0.129/0.131мс/шаг;
остаток небольшой, главным ограничением остаются CPU MoE и main GPU dense.
Из этого профиля нельзя обещать дальнейший TG прирост без нового изменения и
проверки. Новые features в этой сессии остановлены координатором.

Сырые результаты: `/mnt/ai/logs/m31-port/port4-cpu-overlap-speed-v1/speed.tsv`,
server log и runtime/freeze manifest в том же каталоге. Полные API timers,
средние stages и prefill chunks обоих strict-low плеч:
`/mnt/ai/logs/m31-port/port4-cpu-overlap-window-analysis-v1.json`.
Stock-GGML GPU плечо измеряется отдельно на другом frozen ELF; его результаты
не приписываются этому режиму CPU-routed/shared-overlap.
