# M31: opt-in THP backing экспертной RAM-арены

2026-10-26. Направление `streaming_prefill`, третья задача.
Изменён только `m31_arena.hpp` и новые `prefill_thp_*`; GPU kernel/engine
математика и системные настройки не менялись.

## Allocation и lifetime

По умолчанию остаётся прежний
`ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), bytes_)`.
Его исходная строка сохранена. Раскладка `[gate|up|down|scale-tail|padding]`,
логические bytes, все strides и прежние budget/MemAvailable проверки с16GiB
reserve сохранены. Owner добавлен последним private field ExpertArena;
смещения прежних fields не изменены, однако размер C++класса увеличился:
engine/GPU TUs необходимо пересобрать согласованно.

`M31_ARENA_THP=1` включает попытку отдельного Linux anonymous mmap. Payload
выровнен на2MiB; края имеют PROT_NONE guards, чтобы не сливаться с соседними
VMA и измерять AnonHugePages именно этой арены. Guard reservation добавляет
4MiB виртуального адресного пространства, не faulted RAM. Длина payload
округляется лишь до обычной page size; неполный последний2MiB участок не
вынуждает лишнюю huge page. `ggml_backend_cpu_buffer_from_ptr` видит ровно
исходный `bytes_` и не владеет mmap.

`MADV_HUGEPAGE` выполняется до pack/prefault, а helper сам память не трогает.
THP advice не означает реальную huge-page backing. После pack печатаются
`mapped/advised/forced`, размер mapping, измеренный AnonHugePages и note.
В режиме kernel `always` redundant MADV_HUGEPAGE пропускается, как в
`src/core/pinned.cu`: это избегает добавления direct reclaim к автоматическому
THP. `never`, неизвестный режим, mmap/mprotect/madvise failure дают явно
сообщённый fallback на прежний CPU allocator. На других ОС — такой же fallback.

Destructor ExpertArena сначала освобождает borrowed ggml buffer wrapper,
затем RAII owner делает munmap полного reservation. Стандартный allocator
по-прежнему освобождается стандартным ggml buffer free. Никаких mlock,
MAP_HUGETLB, hugetlb pool, sysctl или изменений глобального THP policy нет.

## Fragmentation guard

Изучен upstream `src/core/pinned.cu`: THP madvise+defrag=madvise на
фрагментированной системе может тратить минуты на direct compaction при faults.
Для M31 выбран более консервативный guard: свободные buddy blocks не меньше2MiB
считаются только в Normal/Movable, без DMA/DMA32 reserves. Нужно не меньше
полного eligible huge payload плюс25% (минимальный запас64MiB).
Если buddy info неизвестна или крупных блоков мало, mmap вообще не создаётся
и используется прежний allocator. Это heuristic и возможна гонка с другим
allocator; реальную backing всё равно нужно измерять.

Отдельный `M31_ARENA_THP_FORCE=1` при включённом THP обходит guard и явно
пишет предупреждение про direct reclaim/compaction в note. Такой большой
запрос не запускался и не включён в подготовленные jobs.

Во время small proof было341835776B (326MiB) свободных подходящих блоков.
Настоящая арена163754029056B потребовала бы204692520960B (190.635GiB).
При таком состоянии обычный THP opt-in безопасно откажется от huge advice;
его нельзя объявлять успешным THP speed arm. Состояние buddy динамическое.

## Проверено

`codex-m31-port4-prefill-thp-build-v1`:rc0, отдельная CPU-only link сборка
`build-streaming-prefill/thp`, одинTU, vendor не пересобирался.
Лог `/mnt/ai/logs/m31-port/prefill-thp-build.log`.
Артефакт `/mnt/ai/logs/m31-port/prefill-thp-small-v1.json`:PASS.

- Tiny F32 GGUF:147456B,12экспертов.8циклов создания/уничтожения THP ExpertArena
  побитно равны стандартной packed arena, включая padding/tail. Exact logical
  budget проходит, ggml numBytes и все slice bases/strides сохранены.
-5невалидных budgets (0,negative,NaN,infinity,too small) и2невалидных mapping
  sizes отвергнуты. Повторная инициализация живой арены отвергнута.
- Borrowed wrapper free не освобождает mmap; после destructor `mincore`
  подтверждает ENOMEM на старом payload — RAII действительно unmap-ит память.
- Синтетический guard request170GiB при free blocks0 остановился **до mmap**:
  reservation_bytes=0. Большой production arena не выделялся.
- Настоящий16MiB mapping, force=false:2MiB-aligned, MADV_HUGEPAGE до memset;
  kernel smaps сообщил **AnonHugePages=16777216B,exact=true**. Все16MiB получили
  huge-page backing в этом малом конкретном запуске. Это не обещание для152GiB.
- Tiny147456B меньше одной huge page: advice успешен, но реальный
  AnonHugePages=0,exact=true — hint и backing честно разделены.

## Подготовленные реальные проверки — не запускались

`prefill_thp_jobs.py` только пишет reviewable JSON, без чтения весов/запуска.
Результат `/mnt/ai/logs/m31-port/prefill-thp-prepared-v1/manifest.json`:

1. Parity job `job-arena-parity-port4v1.json`: CPU routed+CUDA dense,
   packed arena170 и guarded THP170, оба code128 против исходного M1 oracle.
   measurement=false; запускается после окончания frozen gate20.
2. Три изолированных speed jobs: mmap/arena0, packed170/THP0, packed170/THP1
   с FORCE0. Тот же om_bench warmup,4K,4K, `reasoning_effort=low`,
  8192/F16/chunk2048/64threads и opt-in CPU+dense decode caches.
   measurement=true/exclusive build lock; запускать только в окне координатора.

У каждого готов `card_launch.py` argv (immutable runner snapshot), отдельные
out dirs и временный preset/backup/rollback. Новая согласованная main сборка
должна включать этот header. После guarded arm сначала смотреть его
`mapped/advised/AnonHugePages`: fallback сравнивает прежний allocator, а не THP.
Startup pack и steady decode — разные расходы. Выигрыш t/s здесь не измерен.
