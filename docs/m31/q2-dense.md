# UD-Q2_K_XL: плотные проекции и dispatch

Обычный `forward(1)` использует штатные `ggml_mul_mat` и не требует нового
ядра для Q5_K-проекций или Q4_K-head. В зафиксированном vendor
`ggml-cuda/mmvq.cu` имеются Q4_K/Q5_K traits, dot-функции и launch dispatch.
На RTX 5090 штатный `ggml_cuda_should_use_mmvq()` выбирает MMVQ для Q4_K
до пяти столбцов и Q5_K до шести; один токен попадает в этот диапазон.
Это вывод по исходникам; модельную загрузку/логиты/скорость проверяет
общая compatibility/validation очередь.

Для первоначальной полной приёмки Q2 достаточно `M31_DSPARK_BATCH_DENSE=0`.
DSpark и эксперименты dense batch не требуются обычной генерации.

## Изменение API exact-MMV

`ggml-exact-mmv.patch` добавляет Q4_K и Q5_K к разрешённым форматам
индивидуально помеченной проекции. F32/Q6_K/Q8_0 сохраняются. Ограничения
K1–4, двумерной contiguous геометрии, F32-активаций, leaf NONE weights в
буфере WEIGHTS, hint NONE и output fence сохранены. Непомеченные узлы
исполняют прежний dispatcher; код dot-ядер не меняется. MSA score tensors
не становятся проекциями. Существующий opt-in `M31_DSPARK_BATCH_DENSE=1`
остаётся единственным production-потребителем этого marker API.

Текущий metadata API contract обновлён: Q4/Q5 принимаются, Q2_K/F16
по-прежнему отвергаются. Исторические receipts port7 описывают прежнюю
версию, их содержимое не переписано.

## Воспроизводимая проверка

Новый `tools/m31/q2_dense_mmv_build.py OUT_NEW_DIR` создаёт изолированную
копию canonical-v2 ggml и заменяет только dispatch TU в собственной
копии CUDA static archive. Защищённые v8 и canonical-v2 не изменяются.
Скрипт проверяет соответствие текущему patch через `git apply --reverse
--check`, записывает SHA источников/ELF, compile/link argv и metadata
contract. Компонент использует все встреченные подписи role/type/shape
реальных Q2 dense weights, два детерминированных семейства активаций,
K1/2/3/4 и контроль K8. Эталон — отдельные штатные вызовы n1; сравниваются
штатный batch, прямой MMV и marker batch. Отдельно проверяется, что
пометка другой проекции не изменяет повторно захваченный MSA score.

## Результат компонента

Сборка `build-q2-dense/mmv-v1` завершилась успешно. Metadata contract:
**189 checks PASS**, CUDA в нём не инициализируется. Build receipt фиксирует
неизменность исходного canonical-v2 archive и SHA новых ELF. Изменён только
dispatch TU; MMVQ/MMVF kernels взяты неизменными из canonical-v2.
Независимый разбор всех 146 archive members подтвердил, что изменился
только `ggml-cuda.cu.o`; MMVQ/MMVF и MMQ Q4_K/Q5_K objects побайтно прежние.
Отдельный source audit нового canonical Q2 подтвердил равенство десяти
критических MMVQ/MMVF/vecdotq/quantize/common/MMQ файлов с port7-v2 и
проверенным overlay. Его dispatch TU и API header равны overlay. После завершения canonical
Q2 отдельный metadata fixture перелинкован с **конечными Q2 CUDA/CPU/base
libraries** и снова прошёл 189 checks без CUDA initialization. Symbols
setter/getter присутствуют в конечных generate/serve/DSpark ELF; SHA и
дизассемблирование сохранены. GPU component на конечном archive заново
не запускался; модельный final-ELF gate проверяется общей очередью.

GPU0 через aictl проверила **26 role/type/shape signatures, 728 cases**.
Все **208 marker cases K1–4, 6 609 920 F32 outputs** побитно равны
отдельным штатным вызовам n1:

| Формат | Marker cases | F32 outputs | Различия |
|---|---:|---:|---:|
| F32 |32|28 160|0|
| Q4_K, output head |8|4 001 280|0|
| Q5_K |64|1 515 520|0|
| Q6_K |72|798 720|0|
| Q8_0 |32|266 240|0|

MSA isolation после пометки другой проекции и повторного CUDA capture:
4096 outputs, различий 0. Контроль K8 намеренно не помечается; прямой
MMV там меняет F32 reduction order, штатный batch может менять ещё и
алгоритм/точность. Штатный F32 batch K4 также отличается от scalar n1.
Эти контрольные случаи не входят в точный контракт K1–4.

Компонент собирает данные и возвращает rc0 после полного обхода, поэтому
приёмка сделана отдельным явным scorer по `marked_mmv`, `columns<=4`,
`different==0`, `max_abs==0`, `rms==0` и MSA isolation. rc0 сам по себе
не объявляется доказательством совпадения.

Receipts:

- `/mnt/ai/logs/m31-port/q2-dense-source-audit-v1.json`.
- `/mnt/ai/logs/m31-port/q2-dense-overlay-object-proof-v1.json`.
- `/mnt/ai/logs/m31-port/q2-dense-canonical-source-proof-v1.json`.
- `/mnt/ai/logs/m31-port/q2-dense-final-api-proof-v1.json` и одноимённый
  `.py`: конечный archive API и runtime link symbols.
- `build-q2-dense/mmv-v1/build-receipt.json` и `cpu-contract.json`.
- `/mnt/ai/logs/m31-port/q2-dense-mmv-component-v1.json`.
- `/mnt/ai/logs/m31-port/q2-dense-mmv-component-v1.summary.json`.
- `tools/m31/q2_dense_mmv_score.py` (копия сохранённого
  `/mnt/ai/logs/m31-port/q2_dense_mmv_score.py`) и
  `q2-dense-mmv-component-v1.explicit-score.json`: 10 независимых checks PASS,
  включая полный перебор каждой подписи × K1–4 × двух семейств активаций.

GPU proof шёл в correctness-окне одновременно с общей canonical-сборкой:
внутренние времена не используются как speed acceptance. Проверены CUDA
13.1/sm120a/GCC14; HIP/SYCL backend не менялись и не собирались этой задачей.
Это компонентное доказательство dispatch, а не DSpark model gate, логиты
полной модели, качество или ускорение. Первоначальный Q2 acceptance
по-прежнему использует обычный decode и `M31_DSPARK_BATCH_DENSE=0`.

Standalone scorer добавлен в tools после canonical build с разрешения root.
Он не включён в runtime targets; собранные ELF и runtime source не менялись.

## Независимый аудит HTTP-профиля и сэмплера

Подготовленные native-speed-v2/native-quality-v1 используют тот же конечный
canonical Q2 serve ELF (SHA `2e2a4de354ad3b4fb1e12a343f42425a08401b8b20d3a2edfc423a5bd58a7a90`),
тот же tokenizer/adapter и идентичный resolved profile. Сверены фактические
SHA frozen files, SHA canonical receipt, полный набор environment/unset,
Q2 model path/revision, ctx40000/chunk32768/80 потоков. DSpark, prefix cache,
GPU cached decode, broad/RMS fusion и batch-dense выключены. CPU IQ2 stock
traits + fused wrapper включены явно. Числовые значения argv проверяются
отдельно от исторического IQ3 provenance внутри базового профиля.

Обычный HTTP worker вызывает `engine.logits(last_n-1,1,...)` для окончания
префилла и `engine.forward(...,1)` + `engine.logits(0,1,...)` для следующих
токенов. `Engine::logits` строит обычный `ggml_mul_mat` для `output.weight`;
exact-MMV marker в этом пути не ставится. Поэтому Q4_K head использует
штатный n1 MMVQ, а новая marker allowlist не меняет обычную HTTP-генерацию.
Профиль/manifest audit — доказательство конфигурации; runtime/gates и
quality должны дополнить его после окончания запросов.

Обе quality-обвязки запускают неизменённый `~/eval_duty_1023.py`, первые
100 GSM8K + 20 tools + 3 продолжения, `temperature=1.0`, `top_p=0.95`,
`max_tokens=6000`, размышления по умолчанию шаблона. `EVAL_TOPK`, `EVAL_MINP`,
`EVAL_MAX` убраны. Фактические умолчания обеих реализаций — **top_k=40,
min_p=0.05**, repeat_penalty=1, frequency/presence penalties=0. Дополнительные
llama DRY/XTC/typical/sigma sampler стадии выключены. Активный порядок —
top-k → top-p → min-p → temperature → distribution. В reference speed log
40/.95/.05 подтверждены реальным sampler params; quality log проверяется
отдельно по мере появления, а не подменяется speed log.

Ограничения сравнения:

- Native использует `mt19937_64` и double probabilities, llama — `mt19937`
  и float probabilities. Seed не задан; это два стохастических прогона.
  Одинаковые controls не означают одинаковый rollout или статистически
  значимое отличие на несколько задач.
- Reference production использует q8_0 KV и ctx64K, native — F16 KV,
  ctx40000 и другой chunk. Таблица сравнивает настроенные конфигурации
  целиком. Она не изолирует один только квант весов.
- Tools23 проверяет также chat template, grammar и HTTP/parser stack.
  Численная сверка с GPU-offload oracle проводится отдельно на выбранных
  одинаковых inputs/геометриях исполнения.

Машинный read-only аудит:
`/mnt/ai/logs/m31-port/q2_http_sampler_audit.py`, snapshots
`q2-http-sampler-audit-v*.json`. Он не объявляет качество по наличию
manifest: требуются все 123 уникальные записи, rc0 HTTP-сессии, GSM≥95
и tools≥19. Финальный baseline snapshot `q2-http-sampler-audit-v5.final.json`:
reference **97/100 и 19/23 PASS**, native **97/100 и 17/23 FAIL**.
Оба содержат 123 уникальные записи, server/card/HTTP/client rc0,
length/leak0. Параметры sampler и SHA соответствуют плану.
`q2-tools-quality-comparison-v1.json` разбирает отличия без пересчёта:
у native дополнительные ошибки tool10/tool13/follow1, исправлен reference
fail tool5; общие9/11/18. В follow1 ответ103,38 семантически верен,
но исходный grader требует103.38. Официальный score остаётся17/23.
В JSONL нет зарегистрированных JSON decode/check исключений; причинность
разницы одному kernel/KV/placement не доказана.

## Независимое review длинного capacity-equivalent oracle

Снимок `q2-capacity-independent-audit-v1.json` в logs PASS: native requested
ctx40000 означает физические 40192 после PAD256. Исходный oracle с теми же
40000/40192 исчерпал VRAM при резервировании 20159,14 MiB compute buffers.
Оба исходных failed result и реальные OOM logs сохранены с SHA.

Повтор меняет только физический oracle ctx на 16384/32768 и output path,
добавляя явный capacity-equivalence guard. Тот же oracle ELF, F16 KV,
GPU op-offload, auto MMQ precision, fusion off и input IDs сохранены.
Native всё ещё requested40000/physical40192. Actual chunk replay совпадает:
16K — 11904+4448+32 singleton; 32K — 11904+9088+7424+4320+32 singleton.
PP active nk: 12032/16384 и 12032/20992/28416/32768 соответственно.

Trained RoPE остаётся orig_context1048576, NEOX, dim64, base10000000,
scale1/ext0. Оба варианта используют тот же полный последний FFN до
head crop; сравнивают по 32 последних n1 logits rows. По compare receipts
64 выбранные long rows побитно равны; это выборочная длинная проверка,
а не все позиции и не gate20.

Ограничение наблюдения обозначено явно: PP native nk взят из runtime
planner log; singleton nk и oracle n_kv/n_ps восстановлены из связанных с
сборкой исходников и runtime-проверки contiguous sequence positions без
wrap/shift. Это не прямые device tensor-shape taps. Независимый audit во
время speed-окна читает только небольшие JSON/source/log файлы; повторный
сырой logits hash проведён позднее в shared correctness/quality окне.
`q2-final-selected-byte-seal-v2.json` PASS: оба frozen ELF совпали с
manifest SHA, обе пары last32 long logits F32 побайтно равны по SHA256
(25 608 192 bytes на файл), IDs/positions совпадают. Повторный расчёт KL
не запускался; основной comparator уже проверяет finite logits.

После reference e4 speed выбран e3: 10/11/12, поскольку e3 быстрее
на всех трёх TG. Финальный sampler audit должен ссылаться на
`q2-reference-quality-e3-v1`, а не на подготовленный, но не выбранный e4.

Дополнительный seal `q2-final-selected-byte-seal-v3.json` подтвердил
**все 224 выбранные primary rows** (code128, last32 code3000, last32
16K/32K) побайтным F32 SHA256, включая signed-zero представление. Эталон
primary — GPU op-offload при CPU storage всех routed weights.

## Отдельная проверка resident E3: допуски не пройдены

Дополнительный oracle с GPU-resident экспертами слоёв 10/11/12 сверялся
на code128 при F16 KV/ctx40000/fusionoff. Фактическое размещение девяти
weight roles подтверждено PR loader; остальные routed weights — CUDA_Host.
Он ближе к размещению выбранного reference HTTP e3, однако его KV/context/
fusion контракт всё ещё отличается от Q8/64K/default-fusion HTTP.

PP96 rows побайтно равны native. В последних 32 teacher-forced n1 rows:
top1=31/32, mean KL=0,0423854, max KL=0,360217; первое ненулевое KL на
позиции 96, единственный top1 mismatch — 104. На всех 128 rows top1=127/128,
mean KL=0,0105963, max KL=0,360217: **KL-допуски провалены**. Нельзя
объявлять pass по одному лишь aggregate top1 99,22%.

Независимый v3 seal подтвердил PP96 raw-byte equality, различие tail32,
и правильность агрегирования метрик фаз. Изменение expert storage связано
с переключением n1 CPU/GPU quant/dot dispatch: CPU IQ2_XS использует Q8_K,
GPU — Q8_1. Это наблюдение вариации от размещения; конкретный ошибочный
оператор, ошибка GPU-ядра или универсальное нарушение GPU-математики не
доказаны. Усиление через дискретный MSA выбор блоков остаётся гипотезой
без per-layer taps. Равенство PP logits также не подменяет сравнение
каждого внутреннего KV tensor. Operational GSM/tools оцениваются отдельно.


## Cached-native E3: один отдельный выбранный diagnostic

Без новой runtime сборки после обоих baseline quality выполнен один
предварительно зафиксированный code128 trial: 384 slots, decode-only seed
всех 128 экспертов слоёв 10/11/12, `--gpu-experts-gib 9.88`, `M31_GPU_CACHED_DECODE_NATIVE=1`,
`M31_GPU_CACHED_DECODE_EXACT=0`. В остальных 54 routed слоях остаётся CPU; отсутствие
incidental hits следует из полного384-key pool и no-admission-on-miss
исходника, а не из отдельного per-layer hit counter.

Независимые source/guard/post-execution audits:
`q2-cached-native-e3-source-review-v1.json`,
`q2-cached-e3-comparator-review-v2.json`,
`q2-cached-e3-post-execution-review-v1.json`. Aggregate 128 и tail 32 gates
прошли: top1=100%, KL/maxabs=0, raw F32 SHA обеих сторон
`a9f613046f807cf3981693c6c264c83f4b4351ef446ad0f73a72b469e78a0f54`.
Реальны 7296 lookups, 384 hits, 96 hybrid forwards, один seed primer
6964641792 bytes; последующих admissions/uploads нет.

Это совпадение с F16 E3 oracle только выбранных 128 teacher-forced rows.
Baseline native 97/17 не переносится на этот режим; новых om/long/GSM/tools
для него этот audit не заявляет. Больший pool меняет long adaptive chunk
budget, а primer должен оставаться внутри TG при дальнейших замерах.
Решение о продолжении — отдельно у координатора; автоматических повторов
или запусков из scorer нет.

## Новая квалификация cached E3 с cap6656 — в работе

После короткого совпадения координатор отдельно разрешил четыре свежих
selected gates, два чистых om-прогона и, при прохождении численного и
скоростного порога, один новый GSM8K100/tools23. Это новое испытание,
а не автоматический повтор неудачного baseline.

Профиль `q2-cached-e3-final-profile-v1/runtime-profile.json`, SHA
`fd78776df17eae2ee1c53ca545e722b7d48b4b0aa4b238e2a81f66d7f99b8acb`:
тот же canonical ELF и ENV/seed, maxchunk6656 вместо32768, cache9.88 GiB,
ctx40000/arena140/F16/adaptive. Нельзя переносить старый short PASS или
baseline97/17 на завершённую приёмку этого профиля.

Prelaunch audit `q2-cached-http-sampler-prelaunch-audit-v1.json` PASS
проверяет profile/ENV/unset, фактические frozen SHA, seed384 и rank/receipt
SHA, sampler controls. Новые numeric artifacts:
`q2-cached-e3-6656-gates-v1/`; helper связывает фактические native chunks
с новым resident E3 oracle. Полный physical ctx40000 пробуется первым;
physical16K/32K fallback допустим только после сохранённого OOM и проверки
active positions/NK/contiguous KV/trained RoPE. Source review helper без
блокеров; это ещё не результат самого испытания.

### Блокировка long16 нового профиля: динамический OOM

Свежие code128 и code3000 при cap6656 прошли побайтно. Long16 не дал
численного результата: сначала полный oracle ctx40000 не разместил
статический compute, затем physical16384 разместил начальный граф, но
завершился `-6` на первой PP6656 в `cuMemCreate`. Original32 после этого
не запускался; новые speed/quality ещё не принимались. Это отказ ёмкости,
а не отрицательный численный gate.

Независимый source/disassembly review сохранён в
`/mnt/ai/logs/m31-port/q2-dense-oom-source-review-v1.json`, воспроизводитель
`q2_dense_oom_source_audit_v1.py`. Четыре frame-диапазона точного ELF
`0e634e…` однозначно дают ASLR bias `0x584ec3ebc000`; frame8 соответствует
`ggml_cuda_mul_mat_q+0x775`, сразу после alloc `0x642012`. Предыдущий
`mm_ids_helper` и формула размера локализуют **ID src1_q8_1**, `mmq.cu:333–334`.
Это не failing alloc `tmp_fixup` и не head. Для n6656/used4 обычные размеры
основной Q8 части gate/up — 175.5 MiB, down — 87.75 MiB, плюс J-padding.
Роль, регистры, фактический запрос и состояние pool в стеке отсутствуют;
эти примеры нельзя выдавать за восстановленный exact request.

Важная поправка бюджета: `6108.69 MiB` — **начальный** `sched_reserve`.
Далее есть `sched_graph_realloc nodes=7949` против reserve7007.
В `ggml-backend.cpp` callback печатается перед новым `reserve_n`; его
итоговый compute не логируется. Поэтому вычитать начальные 6108.69 MiB
из VRAM и объявлять остаток доступным перед MMQ нельзя. VMM дополнительно
держит физический high-water по pool/stream; точное состояние неизвестно.
Runtime, chunk, точность, MMQ-путь и oracle этим review не менялись.

### Отдельный semantic cross-batch контракт ordinary oracle2048

Координатор разрешил один новый oracle-вариант после OOM: штатный PR
resident E3/F16/ctx40000/fusionOFF, PP≤2048, без принудительного полного
FFN59. Native6656 и его профиль `fd7877…` не меняются. Допуски остаются
`top1>=.99, meanKL<=.001, maxKL<=.05`; aggregate и фактические32 n1
проверяются отдельно. Равенство внутренних batch-shape не требуется и
побитность внутренних KV не утверждается.

Это штатная семантика PR: `minimax-m3.cpp:573` выбирает востребованные
строки после attention59 и до FFN59. KV/индексер всех токенов уже обновлены;
результаты отброшенного finalFFN не нужны следующему слою. Смена формы
FFN59 может менять арифметику, поэтому это самостоятельный численный
контракт. Controlcode128 `head_tail=0`: все96 PP outputs востребованы,
прежний E3 gold должен совпасть побайтно до допуска followups.

Source/build review `q2-dense-stock-head-source-review-v1.json` PASS:
отдельный oracle-only ELF `8bc774f8…`, opt-in head-pruning, честные
`force_full_ffn59`/`full_last_layer_before_crop`, causal/contiguous KV
trace, прежние F16/FA/op-offload и trained RoPE; kernels/PR archives/native
не менялись. Неудачная подготовка overlayv1 остановилась на строгом
patch-anchor до компиляции, сохранена; reviewed/compiled overlayv2.

Frozen comparator расположен в `q2-cached-e3-crossbatch2048-v1/`.
Независимый `q2-dense-crossbatch-independent-review-v1.json` дополнительно
сверяет exactENV/CLI, отсутствие FORCE_MMQ/CUBLAS/graph/precision override,
profile/seed/compiled source SHA, inputIDs, head selection и teacher32n1.
Это review подготовки и законченных runtime receipts, не самостоятельный
численный PASS или итоговая quality-приёмка. Его воспроизводитель можно
обновить после появления всех четырёх официальных compare.json.

### Поправка: внешнее занятие GPU и остановка ordinary2048 followups

После source review координатор/validation сообщили, что
`aictl-hold-watch` запустил майнер keryx на GPU0 вопреки hold, заняв ещё
около7.4 GiB. Followups ordinary2048 остановлены; частичные результаты
помечаются aborted/resource-contaminated и не дают завершённого gate.
Численный controlcode128 уже имел официальный byte-equal oldE3 result,
но чистоту соответствующего ресурсного окна должна подтвердить timeline.

Это также меняет трактовку cap16OOM: локализация failing ID-Q8 alloc по
стеку сохраняется, но нельзя называть отказ чистым пределом самого oracle
или доказанным следствием перераспределения графа. Начальный static reserve,
runtime realloc, pool high-water и внешняя нагрузка вместе могли исчерпать
VRAM. Точное время внешнего запуска относительно OOM здесь независимо не
проверено. Исходные артефакты сохранены; дополнения находятся в
`q2-dense-oom-source-review-v2.json` и
`q2-dense-crossbatch-independent-review-v2.json`.

## Закрытие волны: Q2 не принят

Чистый ordinary2048 повтор завершён. Контрольcode128 остаётся побайтным
PASS прежнего E3 gold. Остальные три выбранных сравнения численно FAIL:

| Вход | Top1 | Mean KL | Max KL |
|---|---:|---:|---:|
|около3000,last32|0.875|0.2266581023|1.1526340162|
|16K,tail32|0.84375|0.2138207056|2.2981083579|
|32K,tail32|0.6875|1.9804316195|13.5382793315|

Допуски .99/.001/.05 нарушены; у длинных входов отдельные32 n1 также FAIL.
Чистые outputs находятся в `q2-cached-e3-crossbatch2048-retry-v2/`, аудит
ресурсного окна — в `q2-cached-e3-crossbatch2048-rest-retry-v2/`.
Это разные каталоги по назначению, не замена старых загрязнённых данных.

Независимый `q2-dense-crossbatch-independent-final-audit-v1.json` PASS
подтверждает exact ENV/CLI/profile/seed/source, одинаковые полные IDs и
выбранные позиции, F16/fusionOFF/MMQauto, actual171 expert roles с9 E3 CUDA,
causal contiguous KV/RoPE traces, SHA всех logits/headers/logs. Прямое
чтение всех38,412,288 F32 значений трёх native/oracle пар подтвердило
конечность. Повторно прочитаны214 NVML samples/209 active: queryrc0,
только3 собственных oraclePID, майнеров/посторонних приложений нет.

Менялись PP-разбиение и finalFFN pruning; накопленные KV и пакетная
арифметика тоже могут отличаться. Причина не локализована до одного
оператора или общего «CPU/GPU бага». Source/counter/component PASS не
отменяет численный FAIL. Cached speed и новая GSM/tools не запускались.
Baseline GSM97/tools17 также FAIL по заданному качеству. IQ3_S v8 и
умолчания сохранены; нового принятого Q2 профиля нет.

Scoped review `REPORT-q2kxl.final.draft.md` по numerical contracts,
capacity/head pruning и границам CPU/GPU утверждений конкретных ошибок
не выявил. Новых GPU запусков, вариантов, runtime правок или тестовых
экспериментов этот финальный аудит не создавал.
