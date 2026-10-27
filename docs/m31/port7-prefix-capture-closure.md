# Port7: итог prefix/capture/server

Работа закрыта как набор отдельных opt-in возможностей. Default prefix cache
выключен; DSpark sidecar пуст; CAPTURE_REUSE и CAPTURE_VIEW по умолчанию0.
Без --dspark-model worker не включает capture layers, не грузит sidecar и
не выделяет host feature backing. Новый launcher не добавляет новые worker
аргументы без соответствующих флагов, поэтому может запускать frozen v8 ELF.
Source review default OFF: [port7-default-off-review.md](port7-default-off-review.md).
Ordinary tensor path сохранён, но общий vendor fusion matcher добавляет host
mutex/map lookup даже OFF в исходном API snapshot. Следующий RMS patch
заменил его atomic fastpath при глобально пустой map — отдельный
[review](port7-rms-fusion-review.md); нулевой overhead не заявляется.
Принятая v8 остаётся исходным performance/quality baseline; эту приёмку нельзя
выдавать за полную приёмку нового DSpark пути.

| область | результат | граница доказательства |
|---|---|---|
|Exact full-prompt single-chunk prefix cache|GPU raw cold/reuse/cold:9rows finite byte-exact; worker16PASS, independent29PASS|только полный одинаковый prompt с actual recipe; partial/multi-chunk reuse выключены|
|Capture reuse/window|V7 legacy↔reuse raw bitexact на4352 и хвосте17|старый vector export остаётся; pairedPPV7 −1.373% выявил лишнюю503MB CPU копию|
|Bounded capture VIEW|V9 GPU split3840+256 whole4096, rows0→17, after-release lease byte-exact; independent53PASS|forward live-lease reject, rewind stale epoch проверены; actual HTTP VIEW cancel/next отложен|
|VIEW controller/host|9720cases/55728rows и958 actual ggml host checks PASS|CPU protocol/lifetime; не GPU math/perf gate|
|HTTP DSpark ordinary controller|V5 native100PASS, audit72PASS, HTTP32PASS|короткий16-token smoke; temp1/grammar/max0/1/EOS/suppress/cancel/next, не GSM/tools|
|V10 representative HTTP|6 запросов, depth1/3, три контекста: native256IDs/actual prompt IDs/PP recipes equal|без ordinary-arm и warmup, не финальная quality/throughput приёмка|

Результат generation отрицательный: depth1 24.54/25.76/23.16t/s,
depth3 22.08/21.72/20.97t/s для4K/16K/32K. Verify занимает93.36–95.17%
TG; lazy prepare150.6–166.1ms —1.31–1.56%. PP32K1245.93/1241.50t/s
удовлетворяет абсолютной1000t/s цели на этих запросах, но paired no-regression
не доказан. Координатор отложил READY V9 pairedPP и actualHTTPlease jobs,
чтобы не расходовать финальное окно на медленный M3. Полного GSM/tools с этим
opt-in не было. DSpark не рекомендован как speed default.

Подробности: [prefix](port7-prefix.md), [capture reuse](port7-capture-reuse.md),
[bounded view](port7-capture-view.md), [budget review](port7-capture-budget-review.md),
[HTTP integration/results](port7-dspark-http-design.md).

## Полезные файлы для выборочного коммита

Список относится к prefix/capture/server области; Engine/worker/CMake общие
с другими агентами и должны коммититься согласованным полным изменением.
Коммитов и push этот агент не делал. CPU/CUDA сборки и runtime проверки
перечислены в receipts; HIP/SYCL в этой сессии не собирались.

Runtime:

- `tools/m31/server_prefix_cache.hpp`, `tools/m31/server_dspark_policy.hpp`.
- `tools/m31/capture_host.hpp`, `tools/m31/draft_feature_view.hpp`.
- Совместные изменения `tools/m31/m31_engine.hpp`, `m31_engine.cpp`,
  `m31_serve.cpp`, `serve_m31.py`, `cmake/m31.cmake`.
- `tools/m31/m31_dspark_session.hpp` совместно с DSpark owner и его runtime
  `m31_dspark.hpp/.cpp`, verifier/helper dependencies; отдельно worker без этих
  зависимостей коммитить нельзя.

Meaningful checks/research targets:

- `tools/m31/tests/prefix_cache_check.cpp`, `prefix_engine_check.cpp`,
  `test_prefix_cache.py`, `test_prefix_api.py`.
- `tools/m31/tests/capture_host_check.cpp`, `dspark_session_check.cpp`.
- `tools/m31/tests/test_dspark_worker_protocol.py`, `test_dspark_launcher.py`,
  `tools/m31/tests/serve_dspark_fixture/`.
- `tools/m31/port7_prefix_raw_check.cpp`, `port7_prefix_native_smoke.py`,
  `port7_capture_raw_check.cpp`, `port7_capture_compare.py`,
  `port7_capture_pp_compare.py`.
- `tools/m31/port7_dspark_worker_smoke.py`, `port7_dspark_http_smoke.py`,
  `port7_dspark_http_explore_score.py`, `port7_capture_view_http_smoke.py`.
- Build helpers `port7-prefix-build.sh`, `port7-capture-check-build.sh`,
  `port7-capture-view-check-build.sh`, `port7-dspark-http-fixture-build.sh`.

Документация этой области: данный файл, `port7-prefix.md`,
`port7-capture-reuse.md`, `port7-capture-view.md`, `port7-capture-budget-review.md`,
`port7-dspark-http-design.md`, `port7-default-off-review.md`. Readonly reviews и offline ngram script/docs
можно оставить исследовательскими артефактами; они не нужны worker runtime.
Старые `.patch`, `.orig`, build-output каталоги и разовые logs не являются
необходимыми runtime файлами; они оставлены на диске, ничего не удалено.

## Receipts для итогового отчёта

Все пути относительно `/mnt/ai/logs/m31-port/`:

- `port7-prefix-gpu-v1/{raw/result.json,raw/independent-byte-audit.json,native-worker.json,independent-audit.json}`.
- `port7-capture-view-raw-v9-v1/independent-audit.json`
  (точные raw artifacts перечислены в frozen manifest/result).
- `port7-capture-view-{host,session}-v3.json`, `port7-capture-view-cpu-v3.manifest.json`.
- `port7-dspark-worker-smoke-v2/`, `port7-dspark-http-smoke-v1/`.
- `port7-dspark-http-representative-v10-v1/{exploration-report.json,phase-counter-audit.json}`.
- READY, не выполнены: `port7-capture-view-pp32-v9-v1/`,
  `port7-capture-view-http-v9-v1/`.


Последующий canonical v2 PASS:49 stages, по7 CPU/CUDA targets, final-ELF
CPU-backend fixtures. Source/vendor/protected-v8 fingerprints before/after
равны. Resolved profile75=48v8+27zero. Это не GPU model или quality acceptance;
оба соответствующих flags в receiptfalse. Audit
`/mnt/ai/logs/m31-port/port7-canonical-v2-end-provenance-audit-v1.json`.
Новые default equivalence/om и registered-arena THP trials учитываются root
отдельно; их результаты не предполагаются этим срезом.
