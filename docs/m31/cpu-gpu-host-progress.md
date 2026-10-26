# Parallel host quantization и OAI для точного GPU прототипа

26.10.2026, port4, CPU-only подготовка. Основной engine/GPU исходники не менялись;
полный GPU gate20 продолжает использовать свой frozen binary. GPU не запускался,
скорость этого изменения ещё не измерялась, интеграция ещё не применена.

## Результат

Новые `tools/m31/cpu_gpu_host_executor.{hpp,cpp}` дают bounded persistent pool:

- threads1..64, при32 —31 фоновых worker и caller; tested1/16/32/64;
- `parallel_rows(count, callback)` и `parallel_count` дают disjoint contiguous
  ranges; нет thread spawn на слой/операцию;
- threshold32rows, grain4rows/worker по умолчанию. n1 gate/up и down4rows идут
  на caller без notify/wake workers; n7 down28rows тоже последовательный;
- submissions сериализованы; исключение ждёт завершения остальных ranges и
  перевыбрасывается caller. Следующий job продолжает работать;
- рекурсивный submit/close из callback того же executor отвергается без deadlock;
- close ждёт текущий job, будит/join workers и идемпотентен. После close новые
  submissions отвергаются;
- перед каждым parallel job worker получает fenv/MXCSR caller, затем восстанавливает
  прежний worker environment. Sticky FP exceptions сводятся обратно caller.
  Callback не должен менять FP control settings; независимость строк обязательна.

`cpu_gpu_host_ops.{hpp,cpp}` вызывает исходный `trait->from_float` ровно один раз
на каждую полную activation row: IQ3_XXS/IQ4_XS →Q8_K, NVFP4 →Q8_0. Округления
внутри строки, блоки и число элементов не меняются. OAI сохраняет старый scalar
порядок `min/clamp → a/(1+expf(1.702*(-a))) → glu*(b+1)`; нет нового приближения
exp и нет переноса OAI на CUDA. Caller владеет выходными буферами и может их
переиспользовать; raw pointer API и vector wrappers поддерживают CUDA C++17 header.

Конфигурация для будущей интеграции:

- `M31_GPU_HOST_THREADS=1|16|32|64` (валидны все1..64; default32);
- `M31_GPU_HOST_MIN_ROWS=32`;
- `M31_GPU_HOST_GRAIN=4`.

Нельзя создавать отдельный executor для quant и OAI: предложение интеграции
использует один pool внутри GpuCpuQuants и отдаёт его GPUExperts для OAI.

## Доказательство точности и lifecycle

`codex-m31-port4-cpu-gpu-host-build-v2` завершён **rc0**. Отдельная сборка
`build-cpu-gpu-host`, gcc14/O3, готовые CPU vendor archives read-only, без CUDA
library/runtime. Логи `/mnt/ai/logs/m31-port/cpu-gpu-host-build-*.log`.
СборкаCPU test и385проверок — correctness, времена исполнения не объявляются
микробенчмарком.

Проверено:

- Threads1/16/32/64; n1/7/128/2048; synthetic и сохранённые реальные activation rows.
- Все packed Q8_K/Q8_0 **байты побитно** равны старому serial loop.
- OAI все F32 **побитно** serial; для реальных n128 также побитно сохранённому
  ggml CPU oracle `l10_n128_full_cpu_id_oai.f32`.
- In-place OAI, clip limits, signed zero, subnormal input, overflow expf.
- Четыре rounding modes ×FTZ/DAZ on/off, output bytes, FP status flags и control
  settings caller/worker. Worker-only overflow доходит в caller flags.
- Нулевые/малые jobs не будят workers; исключения, последующее восстановление,
  reentrant rejection, concurrent submissions, close во время active job, repeat close.
- Invalid buffers/shapes, overlap, unsupported types и overflow sizes отвергаются.
- Публичные headers отдельно компилируются с C++17, как текущий CUDA target.

Старый serial arithmetic сохранён независимо в тесте. Provenance, исходные
SHA256 и точные snippets: `tools/m31/cpu_gpu_host_serial_reference.json`.
Там же hashes реальных входов и ожидаемого ggml OAI. Исходный GGUF не читался.

Воспроизведение:

```bash
sudo systemd-run --unit=codex-m31-host-NEW -p User=nekto \
  /bin/bash /mnt/ai/Xolotl-m31/tools/m31/cpu_gpu_host_build.sh
```

## Предложение интеграции, пока не применено

`tools/m31/cpu_gpu_host_integration.patch`, `git apply --check` **rc0** на рабочем
дереве в момент генерации; checksum provenance записан отдельно.
Он меняет только opt-in CPU-compatible CUDA path:

1. GpuCpuQuants владеет одним persistent executor.
2. Старый quantization loop заменяется helper по rows.
3. GPUExperts host OAI использует тот же executor.
4. Canonical CMake собирает два новых C++ файла только при
   `STRATA_M31_CPU_QUANTS_CUDA=ON`, с Threads и без fast-math.

Manual `gpu-build*.sh` также потребуют добавить два C++ objects при последующей
интеграции; текущая patch их намеренно не меняет. CMake CUDA_STANDARD17 менять
не нужно. Флаги default GGML GPU path не затронуты.

После интеграции нужны per-expert intermediate byte gate, code128/tail и suite20
для нового binary: доказательство независимых host rows не заменяет full-model
проверку allocator/ownership/порядка. МикроCPU скорость16/32/64threads и фактический
GPU-prefill throughput — только отдельным окном координатора. CPU OAI/GPU
roundtrips и uploads исходных expert weights остаются отдельными расходами.

## Следующие резервы, только read-only аудит

В strict low4K профиле координатора chunk2048 `dense_read≈4.16–4.20с` против
`dense_compute≈0.36–0.43с` на57routed слоях, `routed_read≈1.44с`, даже после
упаковки IDs. Это ещё не доказывает причину: нужны perf/minor faults/memcpy и
выделение времени std::vector zero-initialization, pageable D2H staging и копий.
Следующий кандидат — persistent caller-owned readback buffers и bounded pinned
staging вместо больших новых vector на каждом слое. В данном этапе engine freeze
не нарушался; эти изменения не реализованы.

CPU speed до сих пор запускался `arena_gib=0`. `m31_arena.hpp` при170GiB пакует
152.51GiB, но пользуется ggml CPU aligned malloc без собственного THP advice.
Системные THP enabled=madvise, defrag=madvise. Простое задание
`STRATA_ARENA_THP=1` **не влияет** на этот M31 allocator: это настройка другого
V4.1 allocator (`src/core/pinned.cu`).

У V4.1 полезен проверенный подход: отдельный2MiB-aligned anonymous mmap,
`MADV_HUGEPAGE` перед first touch и явное ownership/unmap. Для M31 следует
оборачивать такую память `ggml_backend_cpu_buffer_from_ptr`, сохранив original
arena bytes/stride/tail, и измерять фактический AnonHugePages через smaps.
Нельзя применять madvise к произвольному malloc buffer без page alignment и
ясного ownership соседних страниц.

Но сейчас свободных contiguous2MiB+ блоков **лишь около2.55GiB** на152.51GiB
арену (read-only snapshot `cpu-gpu-host-thp-readonly-v1.json`). V4.1 специально
отказывается автоматически просить THP при free_huge_bytes<arena/2: defrag=madvise
может загнать first touch в длительный direct compaction. MemAvailable включает
большой file cache и не обещает нужных contiguous blocks. HugePages_Total=0;
MAP_HUGETLB pool не настроен. Копирование150GiB, mlock, hugepage/sysctl операции
не выполнялись. Такой эксперимент требует отдельного согласованного запуска и
окна: сначала измерить baseline arena170 безTHP, затем backing/pack latency и
TG на THP, не выводить выигрыш из одной лишь замены allocator.
