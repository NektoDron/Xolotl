# Port7: минимальное замыкание общих runtime зависимостей

Это список для выбора source commit root, не команда git add и не
перечень прошедших performance gates. Файлы не коммитились/не удалялись.
Список получен рекурсивным разбором локальных #include из шести runtime
TU: m31_engine.cpp, m31_serve.cpp, m31_dspark.cpp, m31_gpu_experts.cpp,
gpu_cpu_quants.cu, gpu_cached_exact.cpp. Условные include включены в список:
выключенный runtime flag не отменяет compile-time dependency. Vendor
headers/старые tracked headers не повторяются. Этот граф дополняет, но не
заменяет CMake target_sources и backend patch/build receipts.

Новые (untracked в момент аудита) общие runtime зависимости:

- `tools/m31/cached_decode_merge.hpp`
- `tools/m31/cached_decode_merge_rank_outer.hpp`
- `tools/m31/capture_host.hpp`
- `tools/m31/cpu_batched_dot.hpp`
- `tools/m31/cpu_batched_verify.hpp`
- `tools/m31/dense_pinned_io.hpp`
- `tools/m31/draft_feature_view.hpp`
- `tools/m31/gpu_cached_batched_impl.hpp`
- `tools/m31/gpu_cached_decode_impl.hpp`
- `tools/m31/gpu_cached_exact.cpp`
- `tools/m31/gpu_cached_exact.hpp`
- `tools/m31/gpu_iq3_lane_exact.cuh`
- `tools/m31/gpu_iq3_lane_exact.hpp`
- `tools/m31/m31_dspark.cpp`
- `tools/m31/m31_dspark.hpp`
- `tools/m31/m31_dspark_batched_dense.hpp`
- `tools/m31/m31_dspark_batched_graph.inc`
- `tools/m31/m31_dspark_batched_msa.hpp`
- `tools/m31/m31_dspark_batched_ops.hpp`
- `tools/m31/m31_dspark_coalesced_rows.hpp`
- `tools/m31/m31_dspark_packed_routes.hpp`
- `tools/m31/m31_dspark_session.hpp`
- `tools/m31/m31_dspark_verify.inc`
- `tools/m31/m31_dspark_verify_state.hpp`
- `tools/m31/msa_sorted_lookup.hpp`
- `tools/m31/server_dspark_policy.hpp`
- `tools/m31/server_prefix_cache.hpp`

Общие уже tracked изменения должны идти согласованно: CMakeLists.txt,
tools/m31/cmake/m31.cmake, m31_engine.hpp/.cpp, m31_gpu_experts.hpp/.cpp,
gpu_resident_dot.hpp, gpu_cpu_quants.cu, cpu_fused_decode.hpp,
gpu_device_forward_impl.hpp, m31_run.hpp, m31_serve.cpp и serve_m31.py.
Часть принадлежит другим владельцам; этот список не разрешает перезаписать
их правки. Добавленные Stats fields требуют rebuild GPU helper + Gpe +
worker, Config API — согласованный Engine/caller, не relink старого ABI.

Backend: `ggml-decode-fusion.patch`, `ggml-exact-mmv.patch`,
`ggml-rms-only-fusion.patch` и `cmake/ggml_decode_fusion.cmake` нужны для
воспроизводимого CUDA snapshot с marker/mode2 API и empty-map atomic fastpath.
Canonical применяет общий Xolotl patch, затем эти три API patches по порядку. Не следует вручную коммитить generated vendor/build файлы.
Источником точной версии backend служит final frozen build receipt.

Отдельно при включённом STRATA_M31_DSPARK собираются research executables
m31_dspark_generate.cpp, m31_dspark_probe.cpp и port7_verify_dump.cpp.
Они не обязательны для ordinary worker runtime, но CMake optionON требует
все три исходника. Controller proof tests/dspark_session_check.cpp и capture
fixtures нужны для воспроизводимых проверок. Чистый native RPC adapter
port7_dspark_native_mini.py полезен как benchmark harness, не библиотека.

Не нужны для runtime: промежуточные *feature.patch/*verifier.patch/*worker.patch,
*.orig, standalone component executables/build scripts, unused
index_f32_shadow_proposal.hpp, исследования hidden finish/gatherFA/multiK.
Их можно сохранить отдельным research/docs commit по решению root;
не подключать к runtime автоматически. Ничего из этого не удалялось.

После изменения include/target_sources граф надо обновить. В частности
GPU IQ3 два production headers уже в этом списке. Последующие IQ3/GU
компоненты прошли exact proof; selected ordinary V11 OFF/ON/OFF128IDs равны,
но conservative speed ratio0.9989477 не доказывает ускорения. Сам список
не включает будущие неприменённые предложения.


Canonical v2 завершён PASS: `/mnt/ai/Xolotl-m31/build-port7-canonical/v2/`.
49 stages rc0, по7 CPU/CUDA targets, CPU-backend final-ELF math/worker/server/
session fixtures. Source/vendor/protected-v8 before/after равны; resolved
baseline75 controls = исходные48 +27 явных zeros. Независимый metadata audit:
`/mnt/ai/logs/m31-port/port7-canonical-v2-end-provenance-audit-v1.json`.
Это CUDA compile и CPU-backend fixtures, не real-GPU model или quality приёмка:
GPUacceptance/modelquality в build receipt остаютсяfalse. Первый v1 сохранил
CPU compile PASS и import failure из-за пропущенных collector gguf_reader/writer;
v2 добавил только поддержку fixture/builder, runtime не изменялся.

Curated lists сейчас: runtime core60, dense/cache recommended31/21, отдельные
CPU/DSpark inventories. Независимый literal closure всех пяти inventories:
union176 local files,218 edges; `port7-curated-tooling-dependency-audit-v2.json`,
missing files/dependencies0. Первый v1 охватывал только runtime/CPU/DSpark131.
Dynamic/generated paths не считаются полностью доказанными static parser-ом;
canonical реальные fixtures закрывают отдельную execution область.
