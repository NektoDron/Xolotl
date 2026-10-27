# UD-Q2_K_XL: совместимость и выбранные GPU-oracle гейты

Волна 27.10.2026 завершена **отрицательной приёмкой Q2 в Xolotl**.
Основной fast CPU профиль получил GSM97/100 и tools17/23; порог tools19
не выполнен. Reference llama.cpp E3 получил97/100 и19/23.

Primary matched oracle с CPU-хранением экспертов/GPU op-offload совпал
побитно на224 выбранных строках. Это отдельный контракт от E3 resident
oracle. Cached E3 исправил короткий code128 до побитного совпадения, но
чистые ordinary2048 semantic гейты3000/16K/32K не прошли пороги.
Новой полной cached quality или performance приёмки нет; IQ3_S v8
сохраняет прежнюю самостоятельную приёмку.

GGUF: четыре шарда UD-Q2_K_XL, ревизия
`b17b7f5459c0b8f127ea12ee774409b282521d6a`; SHA скачивания подтверждены
координатором. Заголовки и модельные preflight проверялись повторно,
полный134GiB payload заново не хешировался.

## Исходная проверка заголовков

Существующий CPU `build-port7-canonical/v2/bin-cpu/m31_inspect` успешно
выполнил GgufModel::open/read_geometry для shard1, включая поиск всех шардов,
проверку форм/типов/границ тензоров. Это чтение заголовков, не backend load.
Receipt: `/mnt/ai/logs/m31-port/q2-compat-stock-v1/header-inspect.json`.

Получены 60 слоёв, embd6144, vocab200064, NFC=true. Routed experts:
133857017856bytes; GPU fixed weights без embedding:9199661568bytes;
KV F16 при32K:4529848320bytes. Архитектура и reader принимают Q2 без
доработок. Старый рейтинг экспертов не используется.

## План numeric diagnostic

`tools/m31/q2_compat_prepare.py` только готовит jobs, explicit oracle replay
и проверяет результаты; самостоятельно GPU/модели не запускает. Запуск jobs
сериализует общий card dispatcher через aictl/card_launch. Профиль исходного
stock-плеча — resolved port7 baseline с CPU_DIRECT/FUSED/STOCK_EXPERTS=0.
Все cached/exact/fastFFN/DSpark режимы выключены. На префилле остаётся
штатная CUDA MMQ для IQ2_XS; обычный decode использует CPU RoutedGraph.
Никаких новых арифметических ядер в этом исходном плече нет.

| Вход | Context | Chunk cap | Выходные строки | Дополнительная геометрия |
|---|---:|---:|---|---|
| code_02_128 |4096|2048|все128|96-token PP, затем32 singleton teacher-forced входа|
| code_18_3000 |4096|8192|последние32|один большой3000-token PP, head32|
| corpus_16384 |40000|8192|последние32|adaptive PP, затем32 singleton входа, head32|
| corpus_32768 |40000|8192|последние32|adaptive PP, затем32 singleton входа, head32|

Это четыре выбранных teacher-forced случая, а не full gate20 и не greedy
rollout. Они дополняют GSM/tools, но не заменяют их. Correctness-инструменты
используют F16 KV; скоростной llama.cpp-сервер — Q8 KV.

На первом шаге frozen ELF `b31953cd7fcc656dc5955abe4f3d2f4f9eae982836908ae1ba2b301943f27c24`
(port7 canonical v2). Oracle — PR31192, ELF
`45e3f31277242f761932713cfcd351e5aafe385b9d3e3d8838d7dc4d81b1eea0`
с поддержкой actual chunk replay и fullFFN59 до crop головы. В oracle все
expert weights хранятся CPU, `op_offload=1` разрешает штатный GPU compute;
это прежний принятый numeric GPU-offload контракт. Скоростной ref baseline
оставляет три полных экспертных слоя10/11/12 резидентными на GPU — это
отдельная раскладка хранения.

Fusion отключён в обеих сторонах: `GGML_CUDA_DISABLE_FUSION=1` и native
per-backend overrides=0. Сначала выполняется native; затем oracle обязан
повторить фактические contiguous PP/decode boundaries native. Сравнение
проверяет model path, байты входных IDs, позиции строк, vocab, head width,
полную FFN59 до crop, фактические chunks, физический context, fusion policy
и numeric пороги top1≥.99/meanKL≤.001/maxKL≤.05. Все ELF/receipts/профили/IDs
заморожены с SHA. Квантовые payload повторно не хешируются этим скриптом.

При OOM oracle разрешённый root отдельный capacity-equivalence diagnostic
должен явно зафиксировать requested/actual context, active padded n/nk,
contiguous KV и trained linear RoPE. По умолчанию такой fallback отсутствует.

## Границы исходного плана

Jobs в `/mnt/ai/logs/m31-port/q2-compat-stock-v1/` передавались единому
диспетчеру; подготовка файлов сама по себе не была runtime/math приёмкой.
Новые opt-in CPU wrappers проверены отдельным финальным ELF gate против
свежего Q2 oracle, как описано ниже. IQ3_S logits не использовались.

## Фактический stock результат

Все четыре исходных stock case завершились. Свежие Q2 logits побитно
совпали с отдельным PR31192 GPU-offload oracle:224 строки, top1=100%,
mean/maxKL=0, maxabs=0. Oracle использовал тот же физический context, что
native:4096 для двух коротких и40000 для16K/32K. Отчёты `CASE/compare.json`
в `/mnt/ai/logs/m31-port/q2-compat-stock-v1/`.

Stock native/компоненты были выполнены одним настоящим combined card job;
stock oracle и final native — следующим combined job. Per-case `.result.json`
являются явно производными command receipts с exact argv/log/rc и SHA
настоящего combined job/result. Отдельные per-case units не запускались.

## Финальная сборка и производственная геометрия

Финальный native ELF из `build-port7-canonical/q2-v1`:
`2b51de98bbae67d6f9d14a83da1d530d6cf04fa47cdbe7ab917bd99c466e9a8c`.
Все четыре случая повторены с производственными ctx40000, maxchunk32768,
arena140GiB, cache7GiB, threads80. Все76 ENV и CLI движка проверены против
замороженного HTTP profile с SHA
`23c9b526db78afe9076cff152401cd280046a7daf433d42c7ff251c714c6d59f`.
Отдельны лишь correctness-вход/выход, head/rows и teacher-forced tail.
Context40000 — запрошенный размер; обе реализации округляют KV до40192
ячеек. Для short oracle это наблюдается в runtime `llama_context: n_ctx`;
для long capacity receipts requested=actual16384/32768.

CPU_STOCK_EXPERTS=1/CPU_FUSED=1, DIRECT=0; tapsnone сохраняет fused CPU
путь, и реальные startup logs всех четырёх случаев содержат
`M31 fused CPU MoE: enabled=1`. Cached GPU/DSpark/fastFFN/fusion overrides
остаются0. Полные FFN всех строк считаются до bounded head crop.

| Case | Фактический PP | Реальных n1 входов | Проверенные позиции | Строк | Native / oracle requested ctx |
|---|---|---:|---|---:|---|
|code128|96|32|0–127|128|40000 /40000|
|code3000|3000|0|2968–2999|32|40000 /40000|
|16K|11904 +4448|32|16352–16383|32|40000 /16384|
|32K|11904 +9088 +7424 +4320|32|32736–32767|32|40000 /32768|

Итог — **224 строки побитно равны**, top1=100%, mean/maxKL=0, maxabs=0.
Независимый raw SHA256 seal всех224 F32 rows подтверждает побитность:
`q2-final-selected-byte-seal-v3.json`.
Всего96 реально выполненных singleton входов в трёх случаях; code3000
проверяет один большой PP и не содержит decode tail. Каждый row teacher-forced;
это не свободный greedy rollout. Summary:
`/mnt/ai/logs/m31-port/q2-compat-final-v1/selected-gates-summary-v2.json`.

Для16K/32K исходный oraclectx40000 не смог зарезервировать
21138389504bytes compute и завершился OOM. Обе неудачи сохранены.
После разрешённого координатором уменьшения только physical KV allocation
oracle завершился с physical requested=actual16384/32768. Chunk boundaries,
head width, входные IDs, позиции, активные padded keys, contiguous KV и
trained linear RoPE сохранены. Каждый oracle runtime trace проверяет
seq_min=0/seq_max=end−1, nk=PAD(end,256), RoPE origctx1048576,
frequency_scale1/ext_factor0.

Native runtime planner печатает n/nk только для PP n>1. Для32 n1 каждого
long case nk следует из hash-связанного исходника канонической сборки:
`kv_length(P+1)=PAD(P+1,max(256,block128))` и фактических contiguous chunk
positions. Это источник инварианта n1, а не выдуманный runtime nk counter.
Первый checker ошибочно ожидал planner line также для n1; сохранённый отказ
не был numeric failure. Исправленный guard явно различает эти свидетельства.
Результаты long — `CASE/compare-capacity-v1.json`, с отдельными capacity
receipts и runtime traces. Независимое ревью dense-агента PASS: `q2-capacity-independent-audit-v1.json`.

Четыре выбранных случая не являются full gate20/GSM100/tools23. Проверки
качества и скорости финального HTTP-профиля учитываются отдельно в общем
отчёте волны.

## Воспроизведение

Новые каталоги обязательны; инструменты не перезаписывают готовые результаты.
Сначала `q2_compat_prepare.py prepare` готовит stock diagnostics; последующие
`replay`/`compare` формируют oracle jobs и проверяют actual geometry.
`q2_compat_final_prepare.py` — отдельная оболочка для финальных
ctx40000/chunk32768/arena140/env76. Она сохраняет исходный план и явное
производственное преобразование, не меняя существующие jobs или runtime.
`q2_compat_derive_results.py` извлекает честные per-command receipts из
реально завершённых combined jobs. `q2_compat_capacity.py` допускает long
physical-capacity diagnostic только после реального исходного allocation
failure и с явным текстом разрешения; GPU она не запускает.

Все jobs передаются `card_launch.py` через единого диспетчера. Упаковка
новых воспроизводящих tools после канонической сборки не меняла runtime,
зафиксированные ELF, source snapshots, принятые IQ3_S v8 или готовые jobs.

## Дополнительное размещение экспертов oracle

По просьбе root отдельно подготовлен opt-in overlay PR oracle с e3/e4
резидентными экспертными слоями. Он переиспользует read-only PR static
archives, не меняет каноническую Xolotl сборку и по умолчанию сохраняет
старое all-CPU storage + GPU op-offload поведение. Разрешены только
`M31_ORACLE_GPU_EXPERT_LAYERS=10,11,12` либо `10,11,12,13`; первый matching
buffer override направляет эти экспертные матрицы вCUDA0, остальные вCPU.

В `build-q2-oracle-placement/v2` build PASS; v1 сохранил setup failure при
попытке изменить скопированный readonly файл, исправлено только chmod
новой собственной копии. Инструмент `q2_compat_oracle_placement_build.py`
воспроизводит тот же C++ overlay; literal-render SHA совпадает с построенным
v2 (`q2-placement-builder-reproduction-audit-v1.json`). После reference speed выбран E3: он быстрее E4 на всех трёх глубинах.
E3 code128 job завершился; невыбранный E4 остался prepared-only.
Runtime loader подтвердил9 GPU expert roles на слоях10/11/12 и162 CPU
roles; requestedctx40000/actual40192, F16 KV, MMQauto, fusionoff и
фактические PP96+32singleton inputs совпадают с native.

**Дополнительный E3 diagnostic не проходит допуски KL**: top1=127/128
(99,21875%), meanKL=0,01059634, maxKL=0,36021724, maxabs=3,90809.
Первые96 PP rows имеют нулевое расхождение. Различие начинается ровно
с первого singleton входа96: на32 decode rows top1=31/32,
meanKL=0,04238537, worstKL position101; argmax отличается в позиции104.
Это наблюдение отдельного expert placement, согласующееся с переходом
CPU vecdot↔GPU MMVQ на этих слоях, а не доказательство ошибки всей
GPU математики. Полный layer-wise causal probe не выполнялся.

Артефакты — `q2-best-placement-code128-e3-v1/compare.json` и
`phase-analysis.json`. `q2_compat_placement_score.py` воспроизводит
проверку фактических171 loader overrides, CLI/геометрии и численных
метрик. Общий machine summary различает два numeric контракта:
`/mnt/ai/logs/m31-port/q2-numeric-contracts-summary-v4.json`.
Побитность primary all-CPU-storage/GPU-op-offload на224 строках
не означает совпадения с выбранным E3 размещением. GSM/tools обеих
финальных конфигураций остаются отдельной проверкой.

Даже этот дополнительный выбранный code128 диагностирует F16 KV,
ctx40000 и fusionoff. HTTP reference использует Q8 KV,ctx65536 и обычный
fusiondefault, поэтому универсальное побитное совпадение с его свободной
генерацией не заявляется.

## Однократный cached E3 diagnostic

После обеих baseline GSM100/tools23 сессий root разрешил ровно один
дополнительный code128 trial без изменения runtime/ELF. Подготовка:
`/mnt/ai/logs/m31-port/q2-cached-e3-code128-v1/`. После завершения обеих
baseline сессий и cleanup выполнена ровно одна попытка; command rc0.

Используется тот же gen2b51 и existing native cached decode; cache9,88GiB
даёт ровно384 fixed slots по27615616bytes. Decode-only seed384 содержит
все128 экспертов каждого из слоёв10/11/12; ordinary seed0. Rank — явно
синтетический порядок размещения, все usage counts0, calibrationnone;
это не рейтинг наблюдённой частоты. Заголовки/статы текущих четырёх GGUF,
полная перестановка rank и binary seed receipt независимо проверены.

Сравнение — со свежим ранее выполненным E3 F16 gold, ctx40000, fusionoff,
PP96+32 actual n1. `comparison-plan-v2.json` и SHA seal readonly;
замороженный `comparison-v2/q2_cached_e3_compare.py` проверит aggregate128
по прежним порогам и дополнительно decode32 отдельно по тем же порогам.
Оба должны пройти; это дополнительное условие данного кандидата, не
изменение baseline критериев задним числом.

Проверки исполнения требуют однократный успешный seed384, pool384,
seed6964641792bytes,7296 lookup/384 resident routes/96 hybrid forwards;
на каждом n1 ожидается приращение228/12/3 и отсутствие новых weight uploads.
Все четыре expert routes в E3 слоях и ноль в остальных следуют из
наблюдаемого полного seed/pool, полного множества ключей E3 и исходника
cached-only пути без допуска misses. Отдельного per-layer runtime счётчика
в этой сборке нет — это явно source/full-pool inference.

`launch_once.py` сначала проверяет обе завершённые baseline сессии по123
уникальным eval rows и cleanup, затем допускает единственное предъявление
job. Компаратор дополнительно связывает actual submission receipt с этими
завершениями. KL-failure завершает эту попытку; PASS сам по себе не запускает
long/perf/quality и не переносит качество baseline на новый профиль.

Её официальный замороженный scorer v2 прошёл: aggregate128 и decode32
отдельно имеют top1=100%, mean/maxKL=0, maxabs=0. Весь F32 payload
побитно равен существующему E3 gold, SHA256
`a9f613046f807cf3981693c6c264c83f4b4351ef446ad0f73a72b469e78a0f54`.
Артефакт — `q2-cached-e3-code128-v1/strict-compare.json`.

Все33 profile snapshots прошли ожидаемые приращения: финальные
7296 lookup/384 resident routes/96 hybrid forwards, pool10604396544bytes,
один seed6964641792bytes. После primer weight uploads не росли.
Внутренние тайминги экспертов были выключены (`EXPERT_PROFILE=0`),
счётчики печатались через `cfg.profile=1`; это не скоростной замер.

Это дополнительный short numeric PASS выбранного cached-профиля.
Он не переносит baseline GSM/tools на изменившуюся арифметику; baseline
native97/100 и tools17/23 сохранён как отдельный quality fail по tools.
Long/perf/новая полная quality запускаются только отдельным решением root.

## Квалификация cached E3 с maxchunk6656 и ordinary oracle2048

Root отдельно разрешил следующий профиль с тем же gen2b51 и теми же
cached/seed flags, но maxchunk6656/adaptive. Профиль
`q2-cached-e3-final-profile-v1/runtime-profile.json`, SHA
`fd78776df17eae2ee1c53ca545e722b7d48b4b0aa4b238e2a81f66d7f99b8acb`.
Четыре native процесса завершились rc0; actual16K PP6656/6656/3040,
32K PP6656/6656/6144/5376/4736/3168, затем32n1 в обоих long cases.
Seed/coverage audit PASS; code3000 остался PP-only без primer.

Matched-shape resident oracle подтвердил short128/3000 побитно, но его
fullctx40K long16 не зарезервировал CUDA compute12988499968bytes.
После разрешённого physical16K allocation инициализация прошла, однако
первая PP завершилась SIGABRT−6 в MMQ activation-Q8 VMM allocation.
Все исходные ошибки сохранены, long32 этого каскада не запускался.
Независимый journal timeline подтвердил пересечение именно этой попытки
с внешним miner/watch collision. Это загрязнённый resource attempt;
его нельзя считать доказательством чистого workspace-дефицита oracle.
Это resource failure, а не numeric mismatch.

Новый явно разрешённый контракт использовал **ordinary PR E3 oracle**:
F16, ctx40000/actual40192, fusionOFF, PP≤2048, stock last-layer pruning
без принудительного fullFFN59 tap. Новый oracle-only ELF
`8bc774f8d826aa14030ccc73939ba8f65db63b31d976cc193614fef948896b0a`.
Native6656 не менялся. Это semantic cross-batch comparison, а не
matched-shape proof. Проверены полные ID bytes, выбранные позиции,
непрерывный KV0..end−1, causal attention, trained linear RoPE,
параметры F16, E3 storage9roles и actual ordinary2048 boundaries.
PP n/nk и full-last-layer policies сознательно могут различаться.

Контроль code128 (actual96PP+32n1) побайтно совпал со старым E3 gold.
Первый rest3 был остановлен при обнаружении внешнего miner наGPU0;
его частичные данные не использовались для новой численной приёмки.
После ROOTREADY три случая выполнены заново в новых каталогах под NVML
монитором; старые файлы и контроль сохранены.

| Случай | Top1 | MeanKL | MaxKL | Решение |
|---|---:|---:|---:|---|
|code128 control|100%|0|0|PASS; byte-equal oldE3gold|
|code3000 last32|87,5%|0,226658|1,152634|FAIL|
|16K last32n1|84,375%|0,213821|2,298108|FAIL|
|32K last32n1|68,75%|1,980432|13,538279|FAIL|

Long decode32 phase проверена отдельно и также не проходит допуски.
Метрики не подменены и критерии baseline задним числом не изменены.
Ошибка выявлена в чистом повторе; отличие batching/pruning включено в
сам контракт. Без дополнительных layer-wise экспериментов нельзя
приписать всё расхождение одному оператору или объявить его GPU bug.
Новых perf/quality/gold/native вариантов этот helper не запускает.

Артефакты: `q2-cached-e3-crossbatch2048-retry-v2/CASE/compare.json`,
контроль из `q2-cached-e3-crossbatch2048-v1/code128/compare.json`,
`q2-cached-e3-crossbatch-final-summary-v1.json`.
`q2_crossbatch_e3.py` сохраняет отдельный semantic guard и не вызывает
старый compare_case с замалчиванием различий chunk/fullFFN policy.
Native tools17/23 baseline и отсутствие новой полной cached quality
приёмки остаются самостоятельными ограничениями.

## Завершение волны

Root остановил дальнейшую квалификацию после clean numeric FAIL.
Cached om_bench и новая cached GSM100/tools23 не запускались; подготовленные
HTTP jobs не являются результатами. Никаких последующих gold/profile
вариантов, runtime изменений или переносов старой quality приёмки нет.

Финальная machine сводка: `q2-compat-wave-final-summary-v2.json`.
Кураторский список полезных standalone files:
`q2-compat-curated-commit-list-v3.json`. Все свои build units завершены;
модельными jobs и окончательной изоляцией/cleanup управлял validation.

Возможное дальнейшее направление — semantic microbatch2048 внутри
большого слоевого expert-batch: dense/MSA/FFN подгруппы с тем же activeKV
окном, expert weights передаются один раз на слой. Это только план;
реализация, скорость и точность не заявляются.
