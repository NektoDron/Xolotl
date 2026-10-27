# Port7: независимый source review default OFF

База — HEAD `faeb1332c5b245ab0b8a76f59b350b692600b795`
(`m31: document accepted port6 speed and quality results`). Проверен live diff
Engine/header, worker/Python, native GPU expert entrypoints и обязательные ggml
patches. SHA читавшихся файлов сохранены в
`/mnt/ai/logs/m31-port/port7-default-off-review.source.json` (2026-10-27 13:58 UTC).
Это readonly review, без новых сборок, GPU/CPU performance измерений и правок runtime.
Параллельные будущие изменения других агентов в этот вывод автоматически не входят.

Область OFF: сохранён исходный v8 профиль старых ENV, новые ENV выключены/не заданы,
`Config.draft_feature_layers.empty()`, без `--prefix-cache` и `--dspark-model`.
Не следует понимать OFF как отсутствие старых v8 флагов GPU/MMQ/chain/adaptive.

Вывод: в проверенной области не найден новый обязательный GPU compute/copy,
изменение tensor arithmetic/order или явный VRAM resident, который сам по себе
менял бы PP recipe. Добавлен небольшой host control flow. Единственное явное
изменение общего vendor hot path — mutex/map lookup в fusion matcher даже при
неиспользованном override. Поэтому буквальный «нулевой overhead/byte-identical
binary» не подтверждается. Отдельный frozen OFF model/performance результат
нужен для численной/скоростной приёмки; source review его не заменяет.

## Engine init и MSA

`m31_engine.cpp:1424`: обход/копирование/сортировка пустого layers vector,
проверка maxrows4096, создание vector из60 пустых unique_ptr для capture graphs.
Это порядка480B элементов наhost плюс поля/metadata Impl, не GPU graph allocation.
Новые ENV читаются на init, bool fields остаютсяfalse/null. `prepare_capture_host`
немедленно выходит при empty layers или REUSE0: ни96MiB pinned scratch, ни480MiB
host backing в обычном новом worker не выделяются. DSpark library не имеет
статического GPU-load constructor; sidecar появляется только из session init.

`M31_MSA_SORTED_LOOKUP=0/unset` оставляет `msa_sorted_lookup==nullptr`.
На n1 ветка else в `attention` содержит прежние CAST/SCALE/REPEAT/ARANGE/ADD и
следующие gather/FA ops в прежнем порядке. Maps не создаются, их bytes не
попадают в device free snapshot. Для n>1 lookup ветка вообще не используется.

Новый `ProjectedAttention*` по умолчаниюnullptr. Обычный `build_dense` передаёт
прежние аргументы без projected. Все добавленные условия encoded/kv_written/
selected_idx/defer_output проходят старую ветку: RMS/RoPE, KV CPY/SET_ROWS,
indexer, block selection, attention projection сохраняют AST и порядок.
Проекционные MMV markers выставляются только verifier helper-ами; обычный
Engine не вызывает setter. Unmarked vendor matmul идёт по старому dispatch.

## Forward/reset/logits и planner

В начале обычного forward добавлены: live-view guard (short-circuitfalse),
`set_dense_fusion` (возврат при flagfalse), clear пустого verified_logits и
reset пустого verify_state. Последние не вычисляют/не синхронизируют CUDA.

Перед layer loop captured_rows=0, invalidate меняет два bool и проверяет null
owner/epoch, captured_features.resize(0). На каждом из60 слоёв есть новая
проверка captured_rows, тело capture не вызывается. Нет residual graph, D2H,
scatter/export или copy feature matrix. В dense-read дополнительно проверяется
pinned pointer и trace bool, wrapper io_time вызывает прежний operation; при
TRACE0 он не читает clock. Порядок enqueue/read/synchronize на OFF тот же.

`clear_dense_decode_cache` дополнительно обходит60 capture pointers и reset-ит
null. Reset дополнительно сбрасывает null verify_state, пустые vectors, capture
metadata; fusion setter возвращается без backend API. Новых CUDA cache-clear
вызовов из пустых capture/verify owners нет. Старые dense/shared cleanup остались.

`verified_logits` записывает только успешный `verify` ([n,vocab]). Reset,
rewind, release workspace и обычный forward после успешного preflight очищают
его. Logits проверяет empty и дальше строит/читает старую голову. Неудачный
preflight сохраняет предыдущий valid result; ошибка после начала compute
оставляет poisoned и блокирует чтение. Контрпример stale verifier row после
обычного forward/reset в проверенном lifecycle не найден.

Planner добавил external resident subtraction и verify scratch credit:
OFF external_resident_bytes=0 и verify_state=null, поэтому free budget/retained
scratch равны старым. Capture host не VRAM. Native GPU experts добавляют
`leave_decode_seed()` на prefill (пустой order — немедленный return), и ещё одну
проверку пустого decode_seeded_keys при eviction. Порядок выбора victim остаётся
прежним, native `forward_device` FFN операции не меняются. Новые cached workers,
streams и seed uploads не создаются. Дополнительные profile log строки есть
даже при нулевых cached counters; их IO cost не измерялся.

Все новые guard `require(bool,const std::string&)` используют существующую
сигнатуру: длинный literal может создавать temporary std::string даже при
true. Например live-view guard добавляет такую операцию на forward, external
budget guard — на planning. Это host allocation overhead, не tensor math;
величина без измерения не объявляется существенной или нулевой.

## Worker и Python adapter

С пустым dspark_model policy возвращает disabled, session unique_ptr остаётся
null. DSpark constructor/init/observer/advance/finish не вызываются. На каждом
prefill chunk и decode step есть проверка optional pointer; дальше выполняется
прежняя Engine.forward/logits последовательность. Sampling draw, RNG seed,
choose/grammar/ACK timing boundaries не изменены этим refactor.

Без prefix flag ExactPromptCache хранит пустые vectors и clear — no-op по данным;
actual_chunks остаётся пуст. Cache lookup/planner_at/rewind/raw-head copy не
вызываются. Cold start timer повторно ставится после reset, как в v8. Начальный
лишний Clock::now до reset находится вне измеренного prompt interval, но входит
в небольшой request wall overhead. Сборка done JSON вместо temporary — те же
поля при OFF, без новых cached/KV/DSpark fields.

Python добавляет пустой prefill_cache dict на запрос, извлекает до3 optional
ключей после prefill; на done empty guard. usage_counts возвращает прежние
usage поля без cached_n. Новые аргументы не добавляются к worker argv без
флагов — frozen v8 ELF остаётся совместимым.

Единственное намеренное OFF-visible изменение HTTP contract: если клиент
прислал cache_prompt, теперь требуется именно bool (строка/число отклоняются),
и значение передаётся worker. При prefix OFF worker его не читает из-за
short-circuit, а обычный bool запрос всё равно cold. Это не math regression,
но утверждение «все неправильные запросы ведут себя побайтно как раньше» неверно.

## Общие vendor/build изменения и конкретные ограничения

1. `ggml-decode-fusion.patch` обязательно применяется при STRATA_M31 даже если
   M31_GPU_DECODE_FUSION0. `ggml_cuda_fusion_disabled(ctx)` теперь берёт global
   mutex и ищет backend в unordered_map на каждом try_fuse и graph_optimize.
   При отсутствии override результат равен старому static ENV bool. Сами
   steady CUDA graph replays не выполняют fusion matcher; transient/rebuild
   graphs платят этот host lookup. Источника math change здесь нет, но это
   реальный не-opt-in overhead, не измеренный данным review.
2. `ggml-exact-mmv.patch` добавляет проверку magic op_params[15] в matmul dispatch
   и matcher для unmarked узлов; отсутствие marker возвращает старый путь.
   Дополнительной синхронизации или GPU allocation это условие не делает.
3. STRATA_M31_DSPARK=OFF отключает research executables, но runtime library всё
   равно компилируется/линкуется worker. Это build/binary size change, не draft
   load. External STRATA_GGML_DIR теперь обязан иметь два новых patches, иначе
   configure завершится FATAL_ERROR. Старый неизменённый vendor tree не подходит.
4. Config layout/API изменились; старый archive нельзя смешивать с новым header
   или worker. Требуется coherent rebuild Engine+GpuExperts+worker+patched ggml.
   Source/math preservation не означает ABI или binary identity.
5. После хотя бы одного opt-in M3 запроса в том же worker capture host backing
   может оставаться выделенным при последующем sampling fallback с empty layers;
   feature readback отключён, но это уже не never-enabled OFF memory profile.
   Verify scratch уничтожается reset, verified_logits.clear оставляет host capacity.
   Обычный новый worker без sidecar никогда этот путь не проходит.
6. Неправильно унаследованные новые ENV не равны OFF: VIEW1/REUSE0 даже с empty
   layers вызовет init error; fusion/cached/seed flags могут менять policy без
   sidecar. Канонический профиль должен явно обнулять/снимать новые ENV.

Рекомендация для отчёта: говорить «новые алгоритмы и residents выключены;
исходный ordinary tensor path сохранён по source review», а не обещать полное
отсутствие любого overhead. Реальной default-OFF math/lifetime ошибки review
не выявил. Runtime не менялся; новые tests/jobs не запускались.


Обновление после этого source snapshot: новый `ggml-rms-only-fusion.patch`
добавил global empty-map atomic fastpath и убрал mutex lookup при count0.
Его отдельный readonly review — [port7-rms-fusion-review.md](port7-rms-fusion-review.md).
Первоначальное предупреждение выше относится к двум API patches БЕЗ нового RMS
incremental patch. Canonical builder теперь применяет все четыре patches;
данный документ сохраняет исходные review/hash границы и не выдаёт изменение
за новый измеренный performance результат.


Canonical v2 compile/provenance завершены:CPU/CUDA targets и CPU-backend
fixtures49 stages rc0, source/vendor/protected-v8 before==after. Resolved
profile сохраняет48 исходных controls и27 явных zeros. Это не real-GPU default
model/performance gate; он остаётся отдельным от source/build проверки.
Audit: `port7-canonical-v2-end-provenance-audit-v1.json` в logs/m31-port.
