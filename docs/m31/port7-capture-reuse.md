# DSpark residual capture: reuse и окно

`M31_DSPARK_CAPTURE_REUSE=1` меняет только владение F32 host-буферами capture.
Математика target, CUDA graph и порядок residual additions сохраняются.
Default0 продолжает legacy capture.

До PP timer `set_draft_feature_layers` подготавливает persistent host backing
[4096, layers*embd] и scratch [4096, embd]. CUDA scratch использует backend host
buffer type; pageable fallback считается ошибкой. Первое zero/first-touch
выполняется только при выделении. Повторный запрос той же геометрии переиспользует
адреса и не zero-ит backing. Один scratch обслуживает все selected layers.
Каждый валидный slot/row перезаписывается F32 и помечается завершённым;
`draft_features` не возвращает incomplete/stale rows или запасную capacity.
Pinned owner уничтожается до backend.

Reset-only `set_draft_feature_window(min_position)` ограничивает prompt capture
снизу. Worker при eligible M3 и flag1 ставит `max(0,prompt_tokens-4096)` до PP.
Целиком пропущенный chunk даёт успешный getter с rows0/empty values и first=end;
session/standalone consumer сохраняет предыдущие seed segments. Частичный chunk
возвращает только валидный суффикс. Decode verify инвалидирует host logical view
и пишет свой прежний owned vector; после rewind/reset/release old capture
недоступен. Остатки future KV не используются для feature getter.

Проверка: `codex-m31-port7-capture-check-v1` rc0, 565 PASS на настоящем CPU ggml
buffer. Проверены точное копирование token-major F32, reuse адреса, неполные slots,
shrink, zero rows, invalidate, oversized input и новая геометрия. Receipt:
`/mnt/ai/logs/m31-port/port7-capture-check-v1.json`.
Общий CPU/CUDA syntax owner DSpark сообщил rc0; controller 4860 сценариев PASS
содержит skipped-window rows0. Это не доказательство raw GPU parity.

`port7_capture_raw_check.cpp MODEL INPUT.i32 OUT_DIR MIN_POSITION CHUNK REPEAT`
создаёт F32 artifacts captured features/head для независимых flag0/flag1 arms.
В обеих руках сравниваются одинаковые input IDs, chunk/nk, min_position и
выгруженный суффикс. Нужны byte equality finite features и head, отдельно warm
повторный запрос и partial/whole skipped chunks. Таймер probe включает export
и файловый IO, поэтому не используется как PP benchmark. Реальный PP и HTTP
smoke выполняются отдельно на frozen candidate после raw proof.

Реальный GPU proof V7 завершён: `port7-capture-raw-v7-v1`, 6/6 commands rc0.
Один frozen probe ELF SHA256
`70c4c1b6a29be7f275aa6a2eac2c6e65d5787d9b91c01698c965d1fda2e10172`
сравнивает legacy0/reuse1 при остальных равных ENV/IDs/chunk/nk. На4352 токенах
с min256 каждый из двух повторов сохраняет 503316480 bytes features и800256
bytes head: обе руки и warm repeats побитно равны и finite. Features SHA256
`372982e98f9025bdfc400e382de245e3e5fcf50bc8bf4409588790e264f3f949`.

Отдельная геометрия4096+256 с min4335 проверяет реальный GPU getter rows0 после
первого4096 и17rows first4335/end4352 после хвоста. По2088960 bytes features
и800256 bytes head на повтор побитно равны legacy suffix и reuse, включая warm
repeat. Features SHA256
`a3db7abaef7b806b1d1f1003612a93c6f7ddbac3f413259a3a555a349a7e1b76`.

CUDA log подтверждает pinned1 и единичный prepare owner. Frozen guards связывают
source/ELF/input/actual math ENV и SHA256 полного ENV; nonmath значения не
экспортируются. `independent-audit.json` —34 PASS метаданных/controls поверх
frozen byte/finite comparer. PP32K ordinary-vsM3capture1 подготовлен отдельно
как `port7-capture-pp32-v7-v2` с exclusive build lock. Результат этой подготовки и последующего V3 приведён ниже.
Повторный HTTP cancel/next именно с bounded VIEW1 остался отложен.

PP32K V2 завершён rc0 как стандартный om_bench diagnostic: ordinary1250.0/1245.9,
M3reuse1233.6/1227.1 t/s. UUID из штатного om_bench был случайным для каждого
запроса: ordinary33025/33022 tokens, M333022/33020. Это ~−1.4% diagnostic,
а не identical-input A/B или доказательство совпадения greedy256. В подготовке
не был зафиксирован UUID; ошибка обнаружена при проверке response traces.

NEW `port7-capture-pp32-v7-v3` использует UUID sequence ordinaryV2 и обратный
порядок M3reuse→ordinary, по2 запроса32K. Frozen adapter observation после chat
render сохраняет actual token IDs/параметры перед native write. Guarded scorer
проверяет actual IDs/params и prefill-plan chunk/nk recipes до speed score,
затем требует одинаковые256 greedy choices и sampler counts. При различии
не выдаётся PP acceptance. Engine, native sampler и математика не меняются.
На момент подготовки V3 имел статус READY; последующий завершённый результат
приведён ниже.

PPv3 strict проверка завершена rc0. Actual rendered IDs/parameters и все native
chunk/nk recipes совпали. По256 greedy choices в обоих32K запросах и32warmup
совпали, sampling_calls=predicted_n. Prompt33025/33022, native chunks
10240/7936/6528/5632/2689 либо2686. Reverse M3→ordinary дал1233.19/1225.02
против1247.37/1245.04t/s; weighted1229.09 против1246.20, −1.373%. Поэтому
literal no-regression ещё не достигнут, хотя≥1000t/s выполняется. Capture-only
DSpark профиль сOPS0/overlap0 также медленнее в TG18.04/19.39 против24.80/24.05;
этот замер не является доказательством35+t/s. Следующий обязательный фикс
copy export описан в `port7-capture-view.md`.
