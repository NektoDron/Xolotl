# M31: рейтинг экспертов и бюджет VRAM

26.10.2026, port4, субагент cache_ranking. Движок и GPU-кеш не менялись.
Новый `cache_calibrate.cpp` читает `Config.route_observer` после исходного
router и пишет только выбранные IDs/веса; повторная маршрутизация, pruning и
переквантование отсутствуют. Калибровка использует принятый CPU-routed путь,
CUDA dense, nofusion, chunk128/threads64/F16 KV.

## Реальные данные

`codex-m31-port4-cache-calibration-v1` rc0, штатный card_session/aictl,
down и точечный rollback выполнены. Логи в `/mnt/ai/logs/m31-port/`:

- `cache-routing-v1.jsonl`: 4,938,344 байт, 960 токенов, все57 routed слоёв.
- train: RU64, EN127, code128, math129, всего448 токенов/102144 маршрута.
- held: другие RU/EN/code/math промпты, первые128 токенов каждого, всего512.
- `cache-inputs-v1.json`: исходные и усечённые IDs, SHA256, train/held split.
- `cache-ranking-v3.{json,bin}`:7296 экспертов, частота, сумма route weights,
  model/tensor/input identity и held-out replay. Порядок: частота, weight sum,
  затем layer/expert для разрешения равенства.
- Бинарный заголовок: little endian `<8sIIII>`, magic `M31RANK\0`, version1,
  layers60, experts128, count7296; далее32 байта identity SHA256 и7296 uint32
  глобальных IDs `layer*128+expert`. Плотные0..2 не входят.

SHA256 содержимого пяти GGUF взяты из уже проверенного ref
`integrity-ud-original.json` (revision02c81e4...), заново150+ГиБ не хешировались.
Сохранены текущие paths/sizes/mtime_ns и SHA256 tensor directory. Перед будущей
загрузкой rank надо проверять identity, dims, count, полный уникальный набор
routed IDs и происхождение GGUF. Иные веса требуют новой калибровки.

## Held-out replay, это не замер ускорения

Decode replay переставляет захваченные batch-маршруты в token-major порядок,
а prefill replay обрабатывает уникальные эксперты слоя на chunk128. Маршруты
реального autoregressive decode и длинные контексты ещё не записаны.

| Slots | LRU decode hit | Hot+stream decode hit | Hot seeds | LRU prefill hit | Hot+stream prefill hit |
|---:|---:|---:|---:|---:|---:|
|128|0%|0%|0|0%|0%|
|248 (6GiB)|22.085%|23.166%|20|0%|2.298%|
|384|23.562%|28.684%|156|0%|4.871%|
|512|32.048%|32.865%|284|0%|7.269%|

Для decode оставлено228 stream slots =57 layers×4 selected experts, остальное
можно закреплять под hot seeds. Слишком агрессивный pinning ухудшает результат:
244hot+4stream при248slots даёт16.680% против22.085% LRU. Такой seed режим
не следует включать по умолчанию. Полный sweep сохранён в JSON, включая
цену начального seed upload. Для prefill оставляется128 stream slots.
Hit rate считается по экспертным запросам; prefill и decode denominators разные.

Самые частые train пары: (layer23,expert20)262 раза, (18,116)228,
(22,31)197. Это результат короткой выборки, не универсальные hot experts.
Один code128 тем более не даёт репрезентативного production rank.

## Бюджет карты0

`cache_rank.py budget` использует настоящие1191 tensor descriptors без чтения
payload. Воспроизведён точный текущий fixed slot stride **25,909,632 байт**,
включая LCM блоков98/136/36 и128-byte alignment. При6GiB ровно248 slots,
как в реальном логе GPU-кеша. Некрупнейшие эксперты занимают такой же slot:
это цена единого quant-stride, уменьшить её можно несколькими пулами по layout.

`cache-vram-budget-v1.json`: карта32110MiB, CUDA dense weights9.947GiB,
chunk2048. После весов, KV, явного MSA envelope и2GiB резерва остальных
scratch/context/transfer получаются **оценочные**, а не измеренные бюджеты:

| ctx | Native F16 KV | MSA envelope | Native slots | Q8 main + F16 index slots |
|---:|---:|---:|---:|---:|
|4K|0.527GiB|0.346GiB|768|777|
|32K|4.219GiB|2.768GiB|514|587|
|64K|8.438GiB|5.536GiB|225|371|

Native q8 KV **не реализован**; последняя колонка — сценарий для планирования.
MSA envelope складывает2 FP32 scores, F16 attention mask, converted index K
и общие masks/positions. Реальный gallocr использует aliases, а другие живые
тензоры могут увеличить peak; перед реальным auto-budget нужен измеренный
reserve/peak и safety margin. Нельзя автоматически выделять всё число slots
по этой таблице без CUDA allocator проверки.

Ref64Kq8/ubatch2048 держит3 полных слоя10..12, CUDA compute6801.21MiB,
main KV4080MiB, index KV1020MiB. Ref indexer выделяет и dummyV; native хранит
лишь F16 indexK. У ref три слоя10..12 занимают меньше максимальных fixed slots;
поэтому `slots//128` нашего универсального пула не равен числу ref GPU layers.
Необходимо учитывать layout padding, а также измеренный scratch.

## Продолжение и воспроизведение

1. Настоящий cache seed loader: identity validation, rank permutation guard,
   upload top budget без изменения router, protected seed entries отдельно от
   текущих batch protected IDs; зарезервировать stream working set.
2. Сначала source q8_K-compatible GPU correctness gate, затем replay exact
   routing/weights и сравнение logits seed0/seedN. Текущий GPU default min32
   вообще оставляет decode на CPU, поэтому этот rank ещё не ускоряет сервер.
3. Снять production-like decode и long-context калибровку. Выбор seed count
   обучать отдельно, held-out не использовать для заявления выигрыша.
4. Измерить VRAM allocator peak на4/32/64K, затем окно om_bench с тёплым и
   холодным кешем. Hit rate сам по себе не доказывает прирост tok/s.

Сборка `cache_build.sh` через systemd-run использует отдельный
`build-cache-ranking`, shared build lock и готовые vendor/engine archives.
Engine archive должен соответствовать текущему `Config` ABI/route_observer;
после изменения заголовка сначала пересобирается CPU-decode engine.
Сборка rc0; `python3 tools/m31/cache_test.py`4/4, включая запрет eviction
защищённых batch entries и отказ испорченному/неполному trace. Коммитов нет.

## Port4: preflight и runtime reader, ещё без подключения кеша

`cache_seed_preflight.py` проверяет JSON/bin rank, повторный identity SHA256,
полную уникальную перестановку7296 исходных routed IDs, JSON count/weights/order
и совпадение бинарного порядка. Он проверяет текущие canonical paths, sizes и
mtime_ns всех пяти GGUF, читает только headers и сравнивает1191 настоящий
tensor descriptor с исходной таблицей. Payload bounds, архитектура и geometry
проверяются отдельно. Header fingerprint строится из текущих descriptors.
Повторного сканирования163+ГиБ payload здесь нет.

Настоящий preflight прошёл за0.82с:
`/mnt/ai/logs/m31-port/cache-seed-preflight-v1.json`. В receipt явно указано:
content SHA256 наследуются из проверенного ref integrity, а текущие header/stat
проверки **не являются новым content verification**. Изменённый payload при
сохранённых header/size/mtime требует полного хеша; отдельный тест показывает
эту границу проверки. Изменение tensor descriptors или проверенных geometry
полей при тех же size/mtime отклоняется; остальные metadata значения без
исходного полного header checksum заново не удостоверяются. Receipt сохраняет
полные текущие header SHA256, чтобы затем можно было сравнивать и их.

`cache_seed_rank.hpp` — независимый C++17 reader без ggml/CUDA/JSON.
Caller передаёт `ExpectedRank{layers,experts,dense_lead,identity,file_sha256}`
из доверенного свежего preflight receipt. Reader ограничивает размер файла,
проверяет SHA256 всего29KB бинарника, magic/version/dims/count/identity,
dense/out-of-range/duplicate IDs, truncation и trailing bytes. Поэтому даже
перестановка двух допустимых IDs после preflight отвергается по content hash.
Ожидаемый hash нельзя брать из самого непроверенного binary.

`cache_seed::budget` требует reserve128 stream slots для полного слоя
префилла; maximum_hot=slots−128. Рекомендация дополнительно оставляет228
stream slots для предыдущего decode working set: при248slots рекомендует20
hot/228stream, допускает максимум120hot. Это рекомендация короткого replay,
не установленный production optimum.

CPU-only сборки `codex-m31-port4-cache-seed-build-v1/v2` rc0:
`build-cache-ranking/seed-reader/m31_cache_seed_reader`.
`cache_seed_test.py --reader ...`12/12: настоящий tiny GGUF preflight,
искажение model stat/header, JSON identity/directory, duplicate JSON keys,
binary corruption/permutation/truncation/dense IDs, бюджет и SHA256 known
vectors (в том числе пустая строка, Unicode и многоблочное сообщение).
Проверен настоящий7296-ID binary: hot20/stream228/max_hot120,
`cache-seed-real-reader-v1.json`. Лог `cache-seed-build-tests.log`.

### Подготовленный integration patch

`tools/m31/cache_seed.integration.patch` первоначально был только proposal.
`git apply --check` прошёл на исходниках, хеши которых сохранены в
`cache_seed.integration.sources.json`. Затем координатор разрешил применить
его к рабочему дереву (см. следующий раздел); замороженный full gate20
продолжает использовать предыдущую копию исходников. CLI/engine wiring
в исходный patch не включены.

Patch предлагает `GpuExperts::seed_hot(rank_path,ExpectedRank,hot_count)` для
холодного кеша до первого forward. Он проверяет actual geometry и читает
rank через строгий reader, загружает seeds штатным load, завершает DMA перед
публикацией pinned IDs, исключает их из eviction и сохраняет seeded counter
после reset_stats. Ошибка частичного seed переводит кеш в terminal not-ready,
поэтому он не продолжит вычисления с частично достоверной метаинформацией.
Существующая recovery ветка forward при отказе DMA сбрасывает и seeds;
счётчик0 отражает потерю seed, дальнейший reseed — новый Engine.

До применения обязательны code128 seed0/seed20 bit gate, gate20, прежний
NVFP4 scale/rounding gate, async failure/recovery с pinned seeds и проверка
actual peak/seed upload time. Patch ещё не собран и не проверен на GPU.

### Decode cached-only hybrid — проект, без engine реализации

На однотокенном шаге можно отдавать **только уже резидентные** expert hits
GPU, а misses считать исходным CPU ggml. На decode miss не загружать веса
через PCIe; небольшие активации/выходы всё равно передаются. Prefill продолжает
заполнять dynamic cache, hot seeds защищены от eviction. Каждое использование
слота защищено до завершения GPU event, исполнение Engine сериализовано.

Чтобы сохранить M1, нужны отдельные per-rank F32 contributions. Нельзя
сложить все CPU outputs отдельно и все GPU outputs отдельно: это изменит
порядок F32 сложения. После CPU/GPU вычислений надо умножить исходные route
weights, слить slots в исходном top-k порядке0→1→2→3, затем shared/residual
как в CPU oracle. NVFP4 external scales и OAI alpha/clamp/up+1 прежние.

Для overlap GPU запускается перед CPU misses, возврат GPU output — после
CPU работы. Текущий диагностический exact-dot helper синхронный и использует
CPU OAI между матрицами; ему нужен отдельный async per-rank interface прежде,
чем обещать перекрытие. Уменьшение used при packing hits тоже требует probe
для0/1/2/3/4 hits и каждой quant схемы: менять dispatch/округление нельзя.

Приёмка: no-upload counter на decode miss, per-rank outputs и все logits
побитно с CPU oracle для любых hit masks; затем настоящее окно TG4/16/32K
с CUDA/CPU overlap и холодным/тёплым кешем. Replay23%hits не доказывает выгоду:
latency GPU launch, CPU activation quantization, host OAI и синхронизации
могут стоить дороже сэкономленных CPU dot operations.

## Port4: runtime seed API применён, GPU проверка ожидает очередь

По следующему заданию координатора обе исходные SHA256 из
`cache_seed.integration.sources.json` совпали, patch применён. Правки GPU
ограничены init/API/recovery/pinned eviction, `execute` и арифметика не менялись.
Default: seed отключён; исходные router IDs, веса и порядок суммы прежние.

`GpuExperts::seed_hot` подключён также через opt-in environment:

| Переменная | Значение |
|---|---|
|`M31_GPU_SEED_HOT`|Unset/0 отключает seed; положительное число включает|
|`M31_GPU_SEED_PATH`|Путь к исходному rank `.bin`|
|`M31_GPU_SEED_IDENTITY`|Ожидаемый identity SHA256 из preflight|
|`M31_GPU_SEED_SHA256`|Ожидаемый SHA256 rank binary|
|`M31_GPU_SEED_RECEIPT`|Путь к sealed preflight `.seal`|
|`M31_GPU_SEED_RECEIPT_SHA256`|Ожидаемый SHA256 sealed receipt|

`.seal` создаёт `cache_seed_preflight.py --seal NEW_PATH`; это небольшая
структура с identity/bin hashes, geometry, slots/hot и записями model shards:
canonical path, size, mtime_ns, header size и SHA256. В init проверяются
seal hash, соответствие env/receipt/actual slots и все текущие GGUF
header hashes/stat. При смене файла между stat/header-read будет отказ.
Metadata-only первый шард8.2MB дополнительно сравнивается с reference content
SHA, поскольку весь этот маленький файл является header;150+GB payload
остальных шардов заново не сканируется.

Seal и checksums — integrity guards собственных файлов, не механизм
авторизации пользователя. Ожидаемые значения должны приходить из свежего
preflight собственных артефактов. Для этого `cache_seed_launch.py` внутри
aictl worker выполняет verify, пишет новые JSON/seal, выставляет env и exec
явно заданного argv. Без `M31_CARD_SESSION=1` launcher не запускает команду.
Это не заменяет внешнее управление GPU через aictl/card lock.

`cache_seed_build_gpu.sh` делает отдельный link-only
`build-cache-ranking/seed-runtime`, копирует исходники для согласованной
сборки и не меняет vendor. `codex-m31-port4-cache-seed-runtime-build-v1`
rc0. CPU receipt/reader tests17/17, включая изменения metadata header при
сохранённых stat, path/mtime, seal hash/version/trailing/truncation и rank
permutation после preflight. Настоящие5 GGUF headers прошли C++ проверку
из `cache-seed-preflight-runtime-v1.{json,seal}`; model payload hash при этом
не пересчитывался.

Подготовлены:

- `job-cache-seed-probe-v1.json`: direct `GpuExperts`, F32 tiny,6 slots/2hot,
  холодный API/env seed, reset_stats, all-layer eviction, пережившие scan hot
  hits, bitwise output seed0/N, injected partial seed failure и forward DMA
  recovery со сбросом pinned metadata. Без attention/FA, GPU0 через card_launch.
  Controller `codex-m31-port4-cache-seed-probe-v1` поставлен в очередь после
  full gate20 под card lock; результата GPU на этой контрольной точке ещё нет.
- `job-cache-seed-code128-v1.json`: real rank20hot, fresh runtime preflight,
  CPU-compatible GPU math + async, compare с исходным CPU oracle. Нужна новая
  сборка `build-m31-cuda/m31_generate` после seed API; frozen gate20 binary
  намеренно не содержит seed wiring. Запуск только по согласованию координатора.

Лог сборки/CPU проверок: `/mnt/ai/logs/m31-port/cache-seed-runtime-build.log`.
До GPU результатов побитное совпадение seeded пути не заявляется.
