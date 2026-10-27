# HTTP DSpark MVP: дизайн и проверка

Финальный speed/quality клиент должен обращаться к тому же HTTP серверу.
Отдельный research CLI не заменяет om_bench/GSM/tools приёмку.
Новый путь opt-in: Python launcher передаёт worker `--dspark-model PATH`,
`--dspark-depth K` и `--dspark-p-min P`. Без модели поведение v8 сохраняется.

MVP использует DSpark только при temperature0, grammar_check=false, CUDA,
постоянном CPU pool и GPU expert threshold>8. max_tokens<=1 не требует draft.
Остальные условия явно возвращают ordinary target fallback с reason вtimings.
Статические suppress_ids и ignore_eos не являются ограничением: выбор токена
остаётся существующим choose(), а helper проверяет фактически выбранный token.
Ready сообщает greedy-only режим; temp1 GSM/tools работают ordinary path.

## Граница запроса и префилл

Каждый cold request сначала уничтожает draft weights/graphs, обнуляет external
resident budget и сбрасывает target. CPU arena/registration/expert pool остаются.
`Engine::set_draft_feature_layers(layers,maxrows,error)` разрешён только сразу
после reset приready/!poisoned/pos0/last_n0. Для eligible запроса включаются
[4,18,32,46,60], для fallback —empty. Поэтому temp1 не платит feature readbacks.
M3 prefix reuse отключён; cache_prompt=false всегда означает cold target.

После каждого prefill chunk helper получает captured feature segment. Хранятся
последние4096 rows; желательно сегментами сoffset, без повторного копирования
половиныгигабайта на каждомchunk. Исходная математика target/chunk planner
не меняется. Дополнительный D2H всё равно должен учитываться реальным PP.

Worker как раньше вычисляет prompthead и делает первый choose() вprefill.
Draft init, feature append и release_prefill_workspace происходят только
после первогоemit/ACK, в прежнем месте первого Engine.forward. Их стоимость
целиком попадает вdecode_ms. Если maximum0/1 или первыйsampleEOS, draft не грузится.

## Verified raw row очередь

Helper не выбирает токены и не вызывает RNG/grammar. Он принимает previous_token
ровно там, где прежний loop подавал его вEngine.forward перед следующимchoose.
Track logical_fed отдельно отphysical speculative target position.

При отсутствии подходящей очереди helper flush-ит только уже consumed feature
rows вdraft ring, rewind target кlogical_fed, получаетdraft candidates после
anchor и вызывает exact Engine.verify([anchor,candidates]). Queue хранит input
IDs иraw rows. Advance потребляет ровноодин input ивозвращает соответствующую
raw distribution. Следующий choose() вызывается только перед реальнымsample.
Если фактически выбранный previous_token совпадает сследующимcached input,
можно использовать уже вычисленный row; иначе tail отбрасывается и начинается
новаяround среальнымtoken. Features дляring накапливаются только длялогически
поданных input rows и append-ятся пачкой перед следующим draft.

Это поддерживает suppress/stop/ignore_eos без отдельного sampler. Вбудущем тотже
механизм может поддержать stochastic choose иgrammar без дополнительных RNG draws,
но MVP eligibility оставляет ихordinary fallback до собственной приёмки.

## Отмена, конец и метрики

После predicted samples логическая позиция равна
`prompt_n + max(predicted_n-1,0)`: последний sample ещёнеподан вtarget.
Это верно дляlength, EOS доemit иinteractive ACKfalse послеemit.
При любом завершении physical speculative tail откатывается кlogical_fed;
queue/features очищаются, draft evict-ится. При ошибке target reset как прежде.
Нельзя сохранять невидимые будущие токены какcache или учитывать их вKV counters.

Prompt timing заканчивается после первого choose(), decode timing —на последнем
sample доtransport, как v8/om_bench. Draft init/append/verify входят вdecode.
Финальный rollback/eviction после sample раскрывается отдельными counters и
не маскируется подвычисленнуюгенерацию. PP использует только cold вычисления;
cache_false никогда не получает искусственный cached rate.

API helper, owner DSpark agent: begin_prefill, observe_prefill,
prepare_after_first_sample (лениво), advance(previous_token,remaining,raw_logits),
finish/abort иstats. Worker/Python/link patches применены root после review и входят в frozen V5.
CPU fixture использует настоящий helper/worker control flow сmock Engine/DSpark,
без GPU/model weights: 62 сценария, 552 проверки PASS; launcher — 8 PASS.

Реальная проверка V5: `port7-dspark-worker-smoke-v2` — 100 PASS, независимый
аудит — 72 PASS; `port7-dspark-http-smoke-v1` — 32 PASS. Ordinary target и M3
получили одинаковые 16 raw IDs, включая suppress; проверены temp1 fallback,
grammar fallback, max0/1, EOS, cancellation с новым cold запросом, точная
граница fed KV и ленивый draft load. HTTP проверен через completions, SSE,
chat и отмену. Результаты лежат в одноимённых каталогах `/mnt/ai/logs/m31-port/`.
Это короткий correctness smoke, а не приёмка скорости/GSM/tools.

В том же диагностическом запуске warm PP M3 был ~3644–3703 мс против
~3067–3100 мс ordinary. Поэтому последующая опция
`M31_DSPARK_CAPTURE_REUSE=1` вводит persistent pinned scratch и host backing,
без повторного zero fill, и capture только последних 4096 prompt rows.
Её математическая проверка и замер выполняются отдельно; default0 сохраняет
legacy capture. Первое выделение и page touch находятся до PP timer,
sidecar load остаётся внутри decode timer.


## Итог representative actual HTTP, V10

`/mnt/ai/logs/m31-port/port7-dspark-http-representative-v10-v1/`
содержит exploration-report.json (PASS) и phase-counter-audit.json (PASS).
Шесть запросов используют actual om corpus4K/16K/32K, fixed UUID, temperature0,
ignoreEOS/max256, cache_prompt=false. По одному server load на depth, без warmup.
Capture REUSE1/VIEW1, MSA1; численно проверенный профиль COALESCE0/CACHE0/LOOKUP0.

| depth | контекст | PP t/s | TG t/s | accepted/proposed | fed/round | lazy prepare ms | verify ms |
|---|---:|---:|---:|---:|---:|---:|---:|
|1|4K|1276.50|24.54|113/142|1.796|150.60|9702.99|
|1|16K|1534.04|25.76|115/140|1.821|154.42|9243.34|
|1|32K|1245.93|23.16|112/142|1.783|155.91|10356.04|
|3|4K|1267.99|22.08|160/285|2.684|151.54|10970.91|
|3|16K|1529.11|21.72|159/286|2.656|156.69|11173.37|
|3|32K|1241.50|20.97|159/285|2.656|166.14|11572.18|

Для каждого контекста depth1/3 получили одинаковые actual prompt IDs,
полный PP recipe/nk, native256 generated IDs и parsed choices. Sampling calls,
ACKs и predicted_n равны256; fed_generated_n255; KV=P+255. Дополнительно
проверены verified_rows=rounds+proposed, fed=rounds+accepted и cached_rows=accepted.
Это cross-depth consistency и ранее выполненная math-проверка профиля, а не
новая ordinary-arm/oracle или GSM/tools приёмка.

TG считается255000/predicted_ms. Lazy prepare входит в этот таймер и включает
load/init/initial seed append; отдельного seed_ms нет. Prepare занимает1.31–1.56%
TG, verify —93.36–95.17%. Finish/rollback/eviction11.83–21.37ms идёт после
последнего sample и раскрыт отдельно; client wall включает его. Cold4K содержит
первый decode graph/capture setup. Host capture backing подготовлен до PP timer;
без одинакового warmup ordinary-контроля4K PP regression не утверждается.

K1 быстрее K3 во всех трёх контекстах, но ни один не достиг35t/s и не доказал
ускорения над ordinary. M3 остаётся research opt-in; default не меняется.
Paired V9 VIEW PP32K и actual HTTP VIEW cancel/next отложены координатором.
Новые источники speed tuning DSpark в этой волне закрыты.
