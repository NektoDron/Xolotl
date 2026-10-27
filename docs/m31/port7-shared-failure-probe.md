# Port7: подготовленный shared-submit failure probe

`tools/m31/port7_verify_failure_probe.cpp` — новый отдельный opt-in инструмент. Он не изменяет acceptance dumper, Engine/scheduler, CMake или frozen V6. Сборка и GPU запуск пока не выполнялись. Связывать только с согласованным frozen Engine/API snapshot, не с произвольной старой библиотекой.

Публичный hook уже существует: `Config::route_observer`. В новом scheduler `m31_dspark_verify.inc` observer вызывается после установки `SharedDrain.pending=true` и успешного `ggml_backend_graph_compute_async(shared_graph)`. Throw observer поэтому проходит через RAII SharedDrain до outercatch. Наличие реально ещё исполняющихся CUDA команд в момент throw инструмент не измеряет; он проверяет именно ветку асинхронной submit/exception/reset и последующее состояние.

Аргументы как у `m31_generate`; дополнительно `--probe-k 1|2|4|8` (по умолчанию8). Требуются CUDA через aictl runner, `M31_CARD_SESSION=1`, persistent CPU pool, expert threshold>8 и ENV `M31_DSPARK_SHARED_OVERLAP=1`, `M31_DSPARK_BATCH_DENSE=1`. Для первого пилота владелец scheduler предложил sharedinputs1/batchops1/fusion0/cache0; cache1 — отдельный будущий случай. Сам tool не меняет профиль или замок карты.

Для каждого prompt:

1. При disarmed observer выполняется обычный prefill с записью фактических chunk boundaries, затем K жадных n1 шагов с копированием исходных logits и features.
2. Reset и replay тех же chunk. Сохранённые prompt features не нужны; prefill workspace освобождается до verify.
3. Observer armed на один вызов `Engine.verify`, бросает уникальное исключение на первом routed layer (берётся из фактической geometry, ожидается layer3 у M3). Verify обязан вернутьfalse с этим marker. Hook затем disarmed.
4. `forward`, `verify`, `logits`, `draft_features` доreset должны быть отвергнуты — poisoned state нельзя использовать.
5. Reset, replay тех же prefill chunks и тот же K-вход. Verify теперь должен успешно вернуть позиции/формы и побитно те же finite logits/features, что независимые ordinary n1 вызовы.

Результат — небольшой JSON на случай плюс summary.json. F32 массивы на диск не сохраняются. `nk_padding_derived` явно вычислен из geometry, это не runtime trace. Успех не доказывает качество, скорость, отсутствие утечки VRAM за множество запросов или переносимость остальных backend. Это отдельный lifecycle proof, который нужно атрибутировать к frozen source/ELF/profile и выполнять только по разрешённому окну root/validation.
