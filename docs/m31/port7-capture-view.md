# Bounded feature view: устранение export allocation

На strict paired PP32K V7 одинаковые actual prompt IDs/chunk/nk и256 greedy
parsed choices подтверждены. Native generated IDs тогда не записывались. Capture reuse дал1229.09 против1246.20t/s (−1.373%). Из logs
`port7-capture-pp32-v7-v3/timing-decomposition.json`:

| запрос | PP delta | inline capture/прочий forward | prepare delta | вне forward/prepare/head |
|---|---:|---:|---:|---:|
|33025 tokens|304.34ms|55.01ms|−1.43ms|250.71ms|
|33022 tokens|433.47ms|82.11ms|101.87ms|249.77ms|

Capture внутри forward занимает около55.5ms только на последних1407+2689
строках; ранние chunks имеют обычные~0.04ms unattributed overhead. Две прочие
составляющие второго запроса содержат drift. Стабильные~250ms находятся в
`draft_features(values.assign)` с новым vector на каждом capture chunk: новая
страничная память и копирование суммарно503MB. Последующий push в deque —move,
он матрицу ещё раз не копирует.

Новая отдельная опция `M31_DSPARK_CAPTURE_VIEW=1` требует REUSE1. Старый
`set_draft_feature_window(min)` и его min0/longdump смысл сохраняются. Worker
задаёт новый reset-only bounded [begin,end) с end=prompt_n,
begin=max(0,end−4096). Capture пишет текущие куски в непересекающиеся offsets
одного прежнего host backing. GPU kernels, planner VRAM и F32 операции
не меняются. Старый vector getter продолжает возвращать текущий chunk.

Session.observe получает временный view только для проверки metadata и сразу
его отпускает. В ней сохраняются boundaries прежних seed segments без матрицы.
Это сохраняет append/max_append GEMM geometry, включая1024/383/641 хвосты;
слияние в append4096 намеренно не используется. Перед lazy init берётся один
финальный lease; release_prefill_workspace освобождает GPU workspace, сохраняя
plain-host backing и epoch lease. Blocking draft append читает прежние segments
из общей памяти, после чего lease удаляется до verify/следующего forward.

Lease владеет только обычной aligned host памятью через shared owner, не GGML
buffer/backend/pinned scratch. Epoch делает stale data() ошибкой после reset,
rewind, verify и успешного нового forward; Engine destruction также делает view
недействительным. Новый forward/policy change при ещё живом lease отклоняется
до изменения состояния. Cancel/max/EOS до lazy init не держат lease; после
prepare он уже отпущен. Нельзя экспортировать spare capacity/неполные slots.

CPUv2: unit `codex-m31-port7-capture-view-cpu-v2` rc0, actual ggml host/view958
PASS; controller9720cases/55728rows PASS включает view0/view1, cancel/stop/max,
skipped window, suppression, lazy load и stale-tail branch. V3 rc0 повторил его
после добавления equality прежних append recipes (view0↔view1):958 host checks,
9720 controller cases/55728 generated rows PASS. Он исполнился после CPU perf
под shared build lock.
Первыйv1 остановился на missing-Iinclude при сборке controller после958PASS;
вv2 исправлен только include path. Общий Engine CPU/CUDA syntax PASS сообщил
владелец dense lookup; session-template проверяется CPU fixture отдельно.

Расширенный raw model probe создаёт whole-window F32, повторяет чтение через
lease после workspace release, проверяет запрет нового forward при живом lease,
инвалидацию getter/epoch после rewind. Сравнение сохраняет прежние chunk F32
artifacts и дополнительно требует whole-window/after-release bytes equality.
Ниже приведён завершённый GPU raw proof; paired PP и actual HTTP cancel/next
для VIEW остаются отдельными проверками.

Реальный V9 GPU modelproof завершён: `port7-capture-view-raw-v9-v1`, unitrc0,
6commands. Один ELF e2c98535… при REUSE1 и VIEW0/1, остальные новые math flags0.
Два повтора split4096+256 дают whole4096=3840+256 features503316480 bytes,
head800256 bytes: chunk export, accumulated view и view после release workspace
побитно равны, finite. Full-window SHA256
`60d03ed8e28e7aaefb159b0c526e4b33120e4ca591e0616b7bef47e652c8a61e`.

Вторая геометрия min4335/4096+256 подтверждает actual rows0→17;
features2088960 bytes SHA256a3db7aba… и headaffa3bad… совпадают с V7.
Raw producer проверил реальный live-lease forward preflight reject без изменения
позиции, getter invalid после workspace release, lease invalid после rewind
и исключение при data() stale epoch. Оба warm repeats bitexact. Дополнительный
`independent-audit.json` —53 PASS controls/ELF/ENV/lifetime поверх frozen raw
finite/byte comparer. Сам probe включает feature export и IO; его seconds
не используются как PP benchmark.

Подготовлены `port7-capture-view-pp32-v9-v1` и
`port7-capture-view-http-v9-v1`. PP writer дополнительно сохраняет реальные
native event.token IDs после native timers; grammar candidate samples и скрытый
terminal/EOS sample не приравниваются к emitted IDs. Сверяются IDs, indexes,
ACKs, sampling_calls и interactive_waits. CPU protocol smoke на фактическом
frozen RPC observer дал10 PASS (включая скрытыйEOS и cancelACKfalse). Никакая
ретокенизация текста не используется. Оба job остались READY и по решению координатора отложены: M3 не дал
ускорения, дополнительную загрузку модели в финальном окне не расходуем.
Paired PP/no-regression и actual HTTP lease cancel/next не заявлены выполненными.
CPU controller покрывает отмену, а GPU split probe — реальную host lifetime
через release и rewind; эти области не подменяют отсутствующий HTTP smoke.


В последующем V10 exploratory actual HTTP (`port7-dspark-http-representative-v10-v1`)
VIEW1/REUSE1 использованы в шести полных length256 запросах. На depth1/3
совпали все actual prompt IDs, PP recipes/nk и native emitted256 IDs для4K/16K/32K.
PP32K1245.93/1241.50t/s, но ordinary-arm в том же job отсутствует: это не
paired доказательство no-regression. Cold4K не имеет одинакового warmup-контроля.
Нативные token IDs получены из event.token с ACK accounting, без ретокенизации.
Короткий actual HTTP cancel/next именно с VIEW1 по-прежнему pending.

Опции REUSE и VIEW по умолчанию0; без --dspark-model worker не включает
feature layers и не выделяет capture backing. Поэтому принятая v8 остаётся
отдельной базовой конфигурацией. Последняя v8 quality-приёмка не переносится
автоматически на opt-in DSpark: полного GSM/tools на этом пути не было.
