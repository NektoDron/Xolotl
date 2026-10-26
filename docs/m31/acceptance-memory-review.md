# Независимая приёмка seed/receipt и THP memory ownership

26.10.2026, port4. Рецензент `cpu_decode`, не автор seed/THP реализации.
Проверены текущие source, CPU-negative suite и сохранённые GPU/THP proof reports.
Новых GPU запусков, больших арен, hugepage/sysctl/mlock операций и замеров
скорости в этом review не было.

## Итог

**Блокирующих ошибок для текущего candidate benchmark не обнаружено.**
Найдена и по поручению координатора исправлена ошибка Python preflight:
`read_bytes()` rank binary выполнялся до проверки размера; JSON inputs также
читались без границы. Конкретный триггер — ошибочный `--binary` на GGUF150GiB
вместо маленького rank. Это могло запросить чтение/память на весь GGUF прежде,
чем decoder сообщил неправильную длину.

Исправление только Python, native candidate и математика не менялись:

- binary rank проверяется на точную длину `56 + 4*(layers-leading)*experts`
  до open/read;
- rank/tensor/integrity JSON ограничены32MiB до open/read;
- проверяются regular file, stat/fstat и смена размера/mtime при чтении;
- сам read получает `opened_size+1`, поэтому concurrent growth тоже не превращает
  его в неограниченный read;
- rank JSON и tensor SHA берутся из того же проверенного read snapshot, без
  повторного неограниченного reread.

`cache_seed_test.py` с обоими готовыми C++ readers: **18/18 PASS**, включая
новый sparse150GiB negative для четырёх входов. Sparse files занимают меньше1MiB
физических блоков; mock запрещает даже open oversized path. Следовательно,
ошибка действительно выдаётся до попытки чтения, а не после150GiB allocation.
`py_compile` и whitespace check прошли.

## Rank и runtime receipt

- C++ rank reader проверяет ограниченную expected geometry, точный размер файла
  до allocation, внешний ожидаемый SHA256, magic/version/count/identity и полную
  уникальную permutation routed IDs. Dense/out-of-range/duplicate IDs не доходят
  до обращения к expert descriptors.
- При максимальной допустимой geometry rank занимает около16MiB, SHA helper
  имеет явный64MiB bound; arithmetic count/offset на aiserver64-bit не переполняется.
- Receipt ограничен1MiB, shard count1024, path length16384 и header32MiB.
  Cursor сохраняет `position<=size`; take/string/digest не выходят за buffer.
- Runtime сверяет внешний seal SHA, identity/rank SHA, canonical rank path,
  hot count/cache slots и текущие model paths/size/mtime/header boundary/hash.
  Rank заново читается с ожидаемым SHA: перестановка файла после preflight
  не принимается.
- Model loader/read_geometry до arena/kernel paths проверяют shapes, types,
  whole quant blocks и `GgufModel::in_bounds`; payload bytes и offset checks
  вычитают из остатка file payload, не используют опасное `offset+bytes`.
- Direct seed API документирован как requiring trusted preflight от caller;
  env runtime path делает current-model проверку сам. Candidate использует
  `cache_seed_launch.py`, который создаёт свежий receipt перед exec.

Важно для смысла приёмки: это **header/stat integrity guard**, не свежая проверка
полных150+GiB payload. Content SHA унаследован из ref integrity report. Изменение
только payload с сохранением header/size/mtime остаётся явно документированной
границей; соответствующий negative/limitation test намеренно показывает её.
Review не утверждает обратного.

## Seed budget, публикация и DMA recovery

- Seed budget оставляет как минимум `experts` stream slots. Поэтому даже если
  все128 экспертов текущего слоя понадобятся одновременно, hot pins других
  слоёв не делают допустимый batch неразмещаемым.
- LRU victim исключает как current batch protected IDs, так и committed seeds.
  Rank полностью проверен до создания pins/upload.
- `seed_hot` разрешён лишь на cold cache. Pins/seeded_slots публикуются после
  enqueue всех seed slots, finish, uploader synchronize и compute synchronize.
  Внутренние pending entries не доступны следующему compute до этого барьера;
  API объекта явно serialized.
- При partial seed failure объект становится terminal, uploader сначала
  дренируется, затем entries/pins/occupied counts очищаются. Частично записанные
  bytes не превращаются в usable seeded hits.
- При обычном forward failure async uploader дренируется до invalidation;
  очищаются все entries и hot pins, восстанавливается free list. Неудача создания
  нового uploader/exact-dot state делает дальнейший forward запрещённым.
- Host staging ограничен depth≤16 и общим default256MiB. Сравнение
  `slot_bytes<=limit/depth` предшествует произведению; enqueue не выделяет память.
  Частичный отказ pinned buffer/event allocation проходит release/drain/free
  всех уже созданных resources до перевыбрасывания ошибки.
- Borrowed backend и destination slots переживают uploader. Его destructor
  синхронизирует transfer/compute, освобождает events/stages/backend; pool
  освобождается после этого.

Изучен существующий GPU proof `cache-seed-probe-v1/report.json`: cold seeds2,
slots6/stream4,4032bitwise values,28evictions, hot hits после all-layer scan,
forward failure recovery и terminal partial-seed failure — PASS. Review не
повторял GPU probe и не подменяет им полный mathematical gate20.

## THP backing, границы и lifetime

- При выключенном `M31_ARENA_THP` остаётся старый ggml CPU allocator. При guard
  fallback также используется он, а THP owner не владеет его pointer.
- THP helper проверяет размер перед округлением/reservation. На aiserver page4KiB
  alignment и reservation обеспечивают2MiB-aligned payload внутри PROT_NONE
  guards; mprotect/madvise failure unmap-ит reservation и возвращает fallback.
- Logical `bytes_`, expert role offsets/stride, scale-tail и budget/MemAvailable
  проверки не меняются. Mapping page rounding и guards не выдаются GGML за
  дополнительные expert bytes.
- `ggml_backend_cpu_buffer_from_ptr` не владеет payload. ExpertArena destructor
  освобождает borrowed wrapper в теле; только затем member owner делает munmap.
  В engine сначала уничтожаются GPUExperts и weight tensor contexts, затем arena.
- Guard считает Normal/Movable buddy blocks без DMA/DMA32 reserves и с запасом
  до mmap/first touch. FORCE0 на текущей фрагментированной системе способен
  законно дать fallback; FORCE1 отдельно разрешает обход и явно отражён в stats.
- THP hint и AnonHugePages разделены. smaps attribution использует payload VMAs,
  guards предотвращают обычное слияние с соседними anonymous mappings.

Изучен существующий `prefill-thp-small-v1.json`:8tiny arena lifecycle cycles,
packing bytes/offsets/strides, borrowed wrapper free и последующий RAII unmap,
invalid sizes/budgets и pre-mmap fragmentation guard — PASS. Small16MiB mapping
имел16MiB AnonHugePages в том запуске; это **не** свидетельство backing или
ускорения настоящей152.51GiB арены. Реальный THP выигрыш здесь не заявляется.

## Замечания и границы

Неблокирующий найденный defect bounded Python inputs исправлен и покрыт тестом.
Неисправленных конкретных memory blockers для текущих prepared arguments нет.
Настройки cache budget покрывают только pool; dense/KV/compute scratch должны
учитываться отдельно, как и до seed integration. Успех этого review не является
benchmark acceptance и не заменяет runtime peak/backing/quality measurements.

Source hashes и отметки проверок:
`/mnt/ai/logs/m31-port/acceptance-memory-review-v1.json`.

Python fix SHA256:

- `cache_seed_preflight.py`: `9191a8046f4c3eaaa2c79232f0fd6f8d3fa39f2db32b751850b436b149551e66`.
- `cache_seed_test.py`: `0762cda6456a5950036827f39454c9b6e2f26dca27e8b7269e43c0aad1d871d9`.
