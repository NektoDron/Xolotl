# CPU decode: port4, первый этап

Работа отделена от GPU cache. Цель 18.13 → ≥23.33 ток/с пока **не измерена**.
Один NUMA узел на машине; interleave между узлами не даст резерва.

Добавлен opt-in `Config.cpu_decode_cache` / CLI `--cpu-decode-cache 1`:
однотокенные CPU routed graphs сохраняются отдельно для каждого слоя. Их
исходные ggml ops, порядок F32 суммирования, типы весов, IDs/weights сохраняются;
при каждом вызове обновляются input/ids/weights. Смена prompt через reset()
сохраняет графы, но не активации. Batch n>1 и taps идут прежним путём. Default false.
Один cached graph имеет 1MiB metadata и собственный gallocr; отсутствует повторная
постройка графа/alias-plan на следующих single-token шагах. Нет предположения, что
разные transient graphs с одинаковым числом nodes имеют одинаковый alias-plan.

`--profile 1` теперь выводит stage timings prepare, dense, routed setup/compute/read,
residual и отдельные head setup/compute/read. Дальше нужен реальный профиль: сам
статический аудит **не доказывает**, что graph construction определяет потерю TG.
Общий forward timer не включает prepare и head; они печатаются отдельно.

Добавлен `Config.route_observer(layer,n,ids,weights)` для калибровки рейтинга.
IDs/weights token-major `[n,used]`; default callback пустой.

Отдельный link-only build `build-cpu-decode` через `tools/m31/cpu_build.sh`
и `cpu_build/CMakeLists.txt`: использует готовые static vendor libs из
`build-m31-cuda`, vendor не пересобирает. Shared `.build.lock`, `-j24`.

Проверка v2 rc0: старые CPU tiny/NumPy/arena tests и новый `cpu_test_decode.py`.
Новый тест: chunk1, chunk7+decode_tail11, taps fallback; два разных prompts с reset
на одном Engine; все logits cache0/cache1 побитно одинаковы. Затем добавлены head
и prepare timers; v3 пересобирает финальный source и повторяет те же tests.

Активные units:
- `codex-m31-port4-cpu-decode-build-v3` — финальная build/tiny; лог
  `/mnt/ai/logs/m31-port/cpu-decode-build-*.log`.
- `codex-m31-port4-cpu-decode-correct-v1` — ждёт v3 rc0, затем card_session
  `/mnt/ai/logs/m31-port/job-cpu-decode-correct-v1.json` (aictl/lock/trap).
  Code128 и code3000+decode_tail32 against исходные M1 oracle datasets.
  Результаты `cpu-decode-code128-compare-v1.json` и
  `cpu-decode-long-compare-v1.json` в logs/m31-port.

После correctness: отдельное окно без сборок/CPU нагрузки, CPU stage/perf profile,
затем om_bench TG4K на live server с cache0/cache1. Серверному направлению передан
запрос пробросить CLI флаг (serve_m31.py и m31_serve.cpp принадлежат ему).
Без замера нельзя заявлять ускорение или M2 acceptance.

## Второй этап: реальные замеры и dense/head cache

Предыдущий CPU MoE cache проверен на настоящей модели: code128 и code3000 с
последними32 однотокенными шагами **побитно** совпали с M1 oracle.

Окно без сборок/тяжёлой CPU работы, одинаковый om_bench.py, CUDA0/F16 ctx8192,
chunk2048,64threads, два4K запроса после warmup:

| Путь | TG4K, два прогона | PP4K, два прогона |
|---|---|---|
| cpu_decode_cache=0 |17.81 /17.85|117.9 /120.3|
| cpu_decode_cache=1 |18.96 /19.11|117.7 /118.6|
| llama.cpp лучший1GPU ref |23.33|425|

Средняя TG выросла на6.76%, но всё ещё ниже ref на18.4%. Случайные bench prompts
отличались на несколько токенов; это два диагностических4K прогона, не полный M2.
Источники: `/mnt/ai/logs/m31-port/port4-cpu-cache-speed{0,1}/speed.tsv`.

Средние stage times decode по255 шагам на каждый4K запрос:

| Этап, мс/шаг |cache0, два прогона|cache1, два прогона|
|---|---|---|
| Dense |22.103 /22.111|22.268 /21.948|
| CPU MoE setup |1.417 /1.430|0.257 /0.240|
| CPU MoE compute |27.728 /28.515|25.401 /25.970|
| CPU MoE readback |0.136 /0.141|0.144 /0.143|
| CPU residual |0.141 /0.140|0.141 /0.136|

Prepare≈0.055мс; head≈0.7мс отдельно. Stable CPU graph сократил не только setup,
но и measured compute на≈2.4мс. Причинность cache locality/scheduling без perf
не доказана. Разница API времени и суммы engine stages составляет ещё несколько
мс на токен: sampling/interactive IPC/tokenizer parsing нужно измерять отдельно.

Новый **opt-in** `--dense-decode-cache 1` (требует reuse_allocations+cpu_residual):

- Отдельный persistent dense graph каждого слоя для n=1 в текущем nk bucket.
- K/V/index F16 cache записывается `ggml_set_rows` по positions input: не меняются
  data pointers/op params; CUDA capture может пройти warmup и переиспользоваться.
- Head n=1 также persistent: transient head destructor иначе очищал бы все
  backend CUDA captures после каждого токена.
- До замены общих AttentionInputs при смене n/nk очищаются все cached graphs;
  reset также их очищает. CPU MoE graph cache сохраняется как ранее.
- Default CPY-путь и taps используют прежние вычисления.

CPU tiny/NumPy/arena и новые cache/reset tests прошли. GPU synthetic fixture
с HD64/block128/top2/context640 проверяет520+259tokens, bucket256/512 и reset:
все779logits **побитно**. Воспроизводимый generator:
`tools/m31/cpu_make_gpu_fixture.py --out NEW-DIR`. Первый попытанный GPU fixture
с исходными HD16/block4 падал на baseline FlashAttention unsupported shape ещё
до включения кеша; это ограничение fixture, исправлено отдельной CUDA geometry.

До следующего патча pack IDs реальные code128 и code3000/tail32 прошли побитно:
`cpu-dense-code128-compare-v1.json`, `cpu-dense-long-compare-v1.json`.
Смена реального nk bucket во время MSA decode (code3000/tail300) проверена на194
выбранных строках, native default/cache logits побитно:
`cpu-dense-transition-compare-v1.json`. Это дополнительный diagnostic,
не полная suite20 acceptance.

Раздельные dense timers (`build/allocate/upload/compute/read/destroy`) выявили
префилл bottleneck: chunk128 тратил≈310мс на readback против≈40мсcompute.
`top_k` возвращал strided view, и старый loop делал128синхронных CUDA memcpy на
каждом из57слоёв. Под opt-in reuse теперь добавлен `ggml_cont` для I32 IDs и
один readback всехIDs; маршруты/weights не меняются. Default graph не изменён.
Этот patch меняет alias topology, поэтому повторный real gate обязателен.

Текущее состояние:
- `codex-m31-port4-dense-decode-build-v2` — финальная сборка с packedIDs,
  актуальными server grammar counters и CPU tests; source после её старта не менять.
- `codex-m31-port4-cpu-dense-pack-correct-v1` ждёт еёrc0, затем code128 и3000tail32
  против M1 oracle и GPU boundary fixture заново.
- job: `/mnt/ai/logs/m31-port/job-cpu-dense-pack-correct-v1.json`.
- reports: `cpu-dense-code128-pack-compare-v1.json`,
  `cpu-dense-long-pack-compare-v1.json`, `cpu-dense-tiny-pack-compare-v1.json`.

Dense cache/packedIDs скорости в отдельном окне ещё **нет**; диагностические
stage времена из correctness не являются M2 benchmark.

## Третье окно: обе graph caches + packed IDs + server greedy fast scan

Все повторные gates после packed IDs завершились rc0: code128 и3000/tail32
против M1 oracle побитно, CUDA synthetic779rows тоже побитно. Движок зафиксирован
координатором; последующая скорость использует также серверное улучшение greedy
sampling. Поэтому этот замер не изолирует вклад одного dense cache.

Окно без сборок/других тяжёлых задач, тот же om_bench.py, F16 ctx8192/chunk2048,
CPU routed64threads, GPU0 dense, без VRAM экспертов:

| Путь | TG4K | PP4K |
|---|---|---|
| Оба кеша + packed IDs + greedy scan, прогон1 |19.49|124.8|
| То же, прогон2 |20.17|124.3|
| Лучший1GPU llama.cpp ref |23.33|425|

Среднее TG19.83: +11.2% к окну безCPU cache17.83, +4.2% кCPU-cache19.035;
**ниже llama.cpp на15.0%**. PP124.55 ниже425 на70.7%.
Это по-прежнему два4K прогона; 16K/32K/GSM8K100/tools23 не измерены этим окном.
Ref использует Q8 KV64K и три GPU expert слоя, native — F16/8K, все routed CPU.

Средние времена на однотокенном decode (255 шагов каждого4K):

| Этап, мс |Прогон1|Прогон2|
|---|---:|---:|
| Prepare |0.086|0.085|
| Dense всего |18.396|18.339|
| Внутри dense compute |15.654|15.628|
| Внутри dense D2H read |2.126|2.077|
| Внутри dense upload |0.520|0.518|
| Внутри dense build+allocate |0.072|0.092|
| CPU MoE setup |0.257|0.252|
| CPU MoE compute |28.246|26.074|
| CPU MoE read |0.141|0.135|
| CPU residual |0.130|0.133|
| Head setup+compute+read |0.712|0.711|
| Greedy sampling/call |0.347|0.339|
| Interactive ACK/wait |2.847|3.431|
| API время/шаг |51.300|49.588|

Engine+sampling+interactive объясняют API время с остатком0.138/0.088мс/шаг.
Counters sampling включают первый prefill draw, interactive — последний ACK
после последнего sample; средние per-call/wait являются приближением decode
accounting. Engine means и API predicted denominator используют255шагов.
CPU compute между запросами отличается на2.17мс; предыдущие UUID/outputs/routes
также не идентичны. Нельзя приписать эту вариацию регрессии graph cache без
контролируемого perf/thread-affinity эксперимента.

Полные данные и provenance:
`/mnt/ai/logs/m31-port/port4-dense-window-analysis.json` — API timings, средние
engine stages, SHA256 исходного лога, оговорки accounting.
Сырые файлы находятся в `port4-dense-cache-speed1/`.

До23.33ток/с нужно сократить среднее время примерно на7.6мс/шаг. Полное устранение
измеренного interactive ожидания3.14мс дало бы лишь≈21.1ток/с. Серверное направление
готовит более безопасное сокращение PEG parsing RPC без смены stop/cancel
протокола; скорость этого последующего изменения здесь не измерялась.

### Cached-only GPU decode: пока только оценка

Статический cache248 с held-out hit16–20% дал бы идеальный линейный потолок
экономии4.3–5.4мс от CPU compute27.16мс, **до** GPU работы, запусков, transfers и
неизменной CPU barrier цены. Сам по себе даже идеальный20% вариант дал бы около
22.2ток/с на текущем API времени. Превышение23.33 требует сочетания нескольких
улучшений и нового измерения; hit rate не является доказательством speedup.

Корректный будущий контракт: заранее посеянные resident experts, на decode нет
uploads весов и eviction; misses считаются CPU. CPU и GPU возвращают отдельные
уже взвешенные contributions по исходному top-k rank. Сумма собирается строго
rank0→1→2→3; складывать две предварительные частичные суммы нельзя — меняются
F32 скобки. Для n1 CPU tiled threshold8 не включается, но нужны уже проверенные
CPU-compatible Q8_K/Q8_0 CUDA arithmetic, не штатный Q8_1 путь.

Текущий GpuCpuQuants делает три синхронных multiply с host quantization и CPU OAI
между ними. При hit≈18% около30слоёв/токен могут потребовать отдельного GPU пути;
launch/roundtrip цена способна съесть4–5мс экономии. Поэтому сначала нужен
one-token resident-expert microbench с нулевым upload_bytes, затем exact
per-rank gate и лишь после этого integration/speed окно. Код этого режима не
реализовывался; engine source оставлен frozen для общей сборки.
