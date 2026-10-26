# M31: порядок GPU work items по экспертам

2026-10-26. Независимый прототип `streaming_prefill`, вторая задача.
Замороженные `gpu_cpu_quants.cu`, GPUExperts и engine не изменялись.

## Что реализовано

`prefill_route_worklist.hpp` строит stable counting-sort permutation исходных
route indices по expert slot ID за O(routes+slots). Это только порядок заданий:
исходные IDs, router weights, объединение экспертов и последующая сумма
сохраняются. Histogram считается по **полному исходному слоевому батчу**.
Тестовые rare experts имеют7/8/9назначений и проверяют точную границу tiled
dispatch. Disabled/forced tiled override сохраняет прежнюю семантику.

`prefill_route_quants.cu` — отдельная копия текущей CPU-compatible реализации
с единственным изменением CUDA work mapping:

```cpp
row = scheduled_item % rows;
original_route = worklist[scheduled_item / rows];
input_row = broadcast ? original_route / used : original_route;
// прежние ids[original_route], полный histogram, dot и FP accumulation
output[original_route * rows + row] = acc;
```

Порядок операций внутри одного dot не меняется, межэкспертная сумма здесь
не выполняется. Флаги tiled читаются по original_route; histogram подбатча не
используется. В режиме `sort_by_expert=false` независимый прототип повторяет
исходный token-major порядок, чтобы отдельно выявлять ошибки переноса TU.
Дополнительный device worklist имеет4байта на route:32KiB при n2048/used4.
Он копируется отдельно; CPU counting-sort и H2D metadata входят в будущий wall
benchmark, поэтому уменьшение переключений экспертов ещё не означает прирост.

В начале прототипа записан SHA256 исходного frozen CUDA source:
`5cae72878899ab37d3f712c4e899f63fd0d5af10341926cd6608123b5f7ed56d`.
Сборка откажется, если оригинал изменится до отдельного review/rebase.

## Проверено сейчас

- `codex-m31-port4-prefill-route-build-v1`:rc0. Отдельная сборка
  `build-streaming-prefill/routes`, дваCUDA TU и маленькие probes, без vendor
  rebuild. CMake sm120a/CUDA13/g++14, immutable source snapshot и manifest в
  `build-streaming-prefill/routes/source/manifest.json`.
- CPU worklist tests: **48layoutcases,458640F32 values побитно**, n1/7/128/2048,
  used1/4,3randomseeds,broadcast/per-route inputs. Проверяются stable bijection,
  исходный input row/output scatter, полный histogram и tiled flags;
 3invalid ID/shapecases отвергаются.
  Артефакт `/mnt/ai/logs/m31-port/prefill-route-cpu-check-v1.json`.
- `git apply --check tools/m31/prefill_route_integration.patch`:PASS.
  Это проверка применимости, **patch не применялся**.

## GPU proof v1: сортировка побитна; найден дефект harness

`/mnt/ai/logs/m31-port/job-prefill-route-proof-v1.json`, measurement=false,
выполнен координатором через `card_launch.py` после frozen gate20.
Бинарник `build-streaming-prefill/routes/m31_route_probe` сравнил36cases:

- IQ3_XXS,IQ4_XS,NVFP4 — исходные GGUF bytes первых64строк23экспертов первого
  найденного tensor данного типа, без requant или изменения external scales;
  сравниваются raw matmul до внешнего масштаба.
- n1/7/128/2048,used4,broadcast и per-route activation layouts;
  дополнительный disabled tiled n128 и forced tiled n7.
- Независимый CPU ggml `mul_mat_id` → frozen исходный CUDA kernel → новая копия
  с token-major worklist → expert-sorted worklist; требуется bitwise equality.
- Сохраняются tensor names/shapes, raw numeric metrics, все cases и только
  оффлайн число expert transitions. Скорость этим correctness job не измеряется.

JSON `/mnt/ai/logs/m31-port/prefill-route-gpu-proof-v1.json`:

- **36/36** expert-order GPU outputs побитны исходному CUDA baseline;
  также36/36token-orderclone побитны baseline.
- В default dispatch **24/24**baseline CUDA побитны независимому CPU ggml.
- В disabled/forced dispatch CPU comparator дал2/6каждый. GPU агент выявил
  дефект тестовой обвязки: CPU ggml читает tiled env через `call_once`, а v1
  переключал переменные внутри одного процесса. CPU оставался в default,
  CUDA helper читал новые env. Это не расхождение сортировки: её сравнение с
  исходным CUDA kernel по тем же env побитно во всех36cases.
- Итоговый v1 job rc1; это known harness diagnostic, не полный parity pass.
  Frozen queued ELF/source snapshot v1 не изменялся и не отменялся.

## Исправленный proof v2 подготовлен

`--dispatch-mode default|disabled|forced` теперь выбирает один режим до CPU
backend init. Job `job-prefill-route-proof-v2.json` запускает **три отдельных
процесса**, также задавая env до загрузки ELF. Затем `prefill_route_summary.py`
проверяет точный набор24+6+6=36cases и побитность всех трёх сравнений.

Новый отдельный binary `build-streaming-prefill/routes-v2/m31_route_probe`,
`codex-m31-port4-prefill-route-build-v2`:rc0, CPU48cases/458640values повторно
прошли. Baseline CUDA берётся из первоначального immutable source snapshot,
а не из меняющегося production `.cu`. Новый job передан координатору,
GPU v2 результат пока ожидается. Он пишет отдельные
`prefill-route-gpu-proof-v2-{default,disabled,forced}.json` и итоговый
`prefill-route-gpu-proof-v2-summary.json`.

Ускорение и full-model M1/M2 приёмка новым порядком пока не заявляются.

## Reviewable интеграция и замер

`prefill_route_integration.patch` предлагает opt-in env
`M31_GPU_EXPERT_SORT_ROUTES=1`, действует только при n>=32. Default прежний,
decode не затрагивается. Изменяются только route mapping/scatter и host/device
worklist storage. Для приложения нужны отдельные code128 и same-chunk gate20.

Probe имеет `--bench-iters N`, но требует явный
`M31_ROUTE_SPEED_WINDOW=1` в согласованном окне. Сохраняет две пары с
чередованием порядка arms; wall per call включает CPU Q8 quantization,
worklist metadata/sort/H2D, CUDA dot и D2H, исключает загрузку весов.
Для осмысленного cache reuse replay использовать `--rows 0` (полная исходная
высота матриц) либо достаточную высоту, чтобы рабочий набор превысил L2.
Leading64rows correctness case может целиком поместиться в L2 у обоих порядков.
Это матричный replay, не скорость генерации/префилла всей модели.

CUDA block scheduler не гарантирует последовательность между blocks; сортировка
только сближает work items одного эксперта в launch order. Реальный эффект
нужно измерить. Если kernel ограничен integer decode/инструкциями, уменьшение
weight DRAM traffic может не дать выигрыша. TensorCore kernel в этой задаче
не реализуется.
