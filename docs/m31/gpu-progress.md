# GPU correctness, port4

2026-10-26. Fixed VRAM slots + layer `mul_mat_id` проверены на code128.

- `gpu-slot-code128-compare-v2.json`: против GPU-offload oracle **128/128 top1, KL0, maxabs0, побитно**.
- `gpu-slot-vs-cpu-code128-v2.json`: против исходного M1 CPU oracle top1=.78125, meanKL=.7785709318, maxKL8.68356237. Это в точности oracle GPU-vs-CPU, т.е. дополнительная ошибка grouped/MMVQ устранена.
- Штатные IQ3/IQ4 CUDA операции квантуют активации Q8_1 (32 элемента), CPU Q8_K (256); NVFP4 CPU Q8_0. `--gpu-q8-activations` не заменяет Q8_1 на Q8_K. Принятый M1 CPU oracle не меняем.
- Полная synthetic tiny CUDA модель не поддержана: head geometry fixture вызывает abort в CUDA FA (`fattn.cu:761`) до MoE. CPU tiny ранее проходит. Для маленькой CUDA проверки нужен изолированный expert probe.
- Real code128: arena pack72.975с; forward11.137с (диагностика, не speed benchmark); slots248×25909632B, pool6425588736B, uploads133020050688B, misses5134, scratch22025216B. Cache6GiB, chunk128, threads64, nofusion.

Следующий шаг: opt-in CUDA dot с CPU Q8_K/Q8_0 activation quantization и совместимым AVX2 FMA порядком накопления. Сначала single expert, затем full code128 против неизменённого CPU oracle. M2 GPU acceptance ещё нет.

## CPU-quant compatible CUDA proof

`gpu-cpu-quants-probe-v1/report.json`: original expert0 слоёв3/10/52, n7,
все gate/up/down и full OAI FFN промежуточные стадии **побитно равны CPU**.
Проверены IQ3_XXS/IQ4_XS/NVFP4 и external NVFP4 scale.
`gpu_cpu_quants.cu` использует штатный CPU activation quantizer Q8_K/Q8_0 и
CUDA integer dots с тем же AVX2 8-lane/FMA/reduction порядком. OAI пока CPU.
Новый CMake option `STRATA_M31_CPU_QUANTS_CUDA` OFF default; исполнение отдельно
включается `M31_GPU_CPU_QUANTS=1`. Это прототип корректности, ещё не скорость.
Сборка link-only `build-gpu-correctness`, vendor archives не менялись.

## Осторожно: n7 не покрывает CPU tiled dispatch

Первый full-model CPU-compatible prototype gate128 **не прошёл**:
top1=.7578125, meanKL=.76127884. Он ещё не пригоден для M1/M2.
`gpu-cpu-quants-code128-compare-v1.json`.

Найдено ещё одно CPU отличие: IQ3/IQ4 CPU `MUL_MAT_ID` при >=8 назначениях одному
эксперту включает `ggml-cpu/tiled/tiled.cpp`. Там целочисленная сумма по256
элементам преобразуется в F32, затем `sum*weight_d` умножается на activation_d
с FMA в накопитель (подтверждено linked object disassembly); для IQ3 weight_d уже включает0.25. При<8 используется
AVX2 vec_dot с8 FMA накопителями. Выбор делается per-expert, не по общему n.

`gpu-cpu-quants-probe-n128-v1/report.json`: при n128 отличие CUDA AVX2-compatible
от CPU tiled составляет relL2≈1.1–1.4e-7 для matmul,1.69e-7 full FFN.
Это доказывает различие накопления, но пока не доказывает, что оно объясняет
всю full-model ошибку. Source исправления dispatch по histogram IDs написан,
сборка/повтор отложены на окно CPU speed координатора.

Tiled source учитывает отдельный weight mul + FMA с activation_d.
Хотя исходник VNNI содержит отдельные mul+add intrinsics, GCC14 -O3 сворачивает
их в vfmadd213ps; проверен objdump linked tiled-kernel.cpp.o.
На aiserver подтверждены AVX512VNNI/VL/DQ (gcc14 -march=native predefines).
Опциональный `M31_GPU_CPU_QUANTS_PROFILE=1` печатает только диагностическое
время host quantization и общего H2D/kernel/D2H; это не speed benchmark.
`M31_GPU_QUANTS_DEBUG_DIR` сохраняет реальные input/IDs/gate/up/down/result
MoE3 для per-role диагностики. CPU OAI и staging roundtrips ещё присутствуют.

Ограничение прототипа: точная FP-совместимость проверяется с данным
`ggml-cpu` aiserver (gcc14 -O3 -march=native, EPYC9654, AVX2+FMA,
AVX512VNNI). Kernel требует AVX2+FMA CPU build. Переносимость на иные
компиляторы/`-ffp-contract` не заявляется; общий штатный ggml CUDA default
не меняется. CPU OAI — диагностический компромисс; perf acceptance отдельно.

## v2: vecdot/tiled arithmetic проверена

`gpu-cpu-quants-probe-v2/report.json`: **36 cases**, слои3/10/52,
n1/7/128, отдельные gate/up/down и full FFN. Все промежуточные
`cuda_cpu_quants` результаты **побитно CPU** (bad0), включая tiled n128
и NVFP4 down scale. Доказывает совместимость как vecdot, так и tiled.
Полный code128v2 с реальными MoE3 dumps запущен отдельно; результата пока нет.

## Code128v2 принят против неизменённого M1 CPU oracle

`gpu-cpu-quants-code128-compare-v2.json`: **top1=1, meanKL=maxKL=maxabs=0,
все128 позиций/25,608,192 logits побитно**. Юнит
`codex-m31-port4-gpu-cpu-quants-code128-v2` rc0, down+rollback выполнены.
M1 oracle и исходные веса не менялись, GPU0 через aictl/card lock.

`gpu-quants-l3-compare-v2.json`: реальные MoE3 input, исходные512 expert IDs,
gate/up и сумма routed output побитны CPU oracle. В слое3 одновременно
29 tiled экспертов (>=8 назначений) и60 vecdot экспертов.

Диагностический один forward128:13.719с (не M2 benchmark).
171 matrix calls: host activation quantization228.957мс;
H2D+kernel+D2H1001.056мс; scratch14381056B. Cache248 slots /6425588736B,
misses5142, uploads133227327744B. CPU OAI, pageable small activation packing,
синхронные roundtrips ещё присутствуют. Массовая загрузка экспертных весов
остаётся главным видимым расходом этого конкретного пути; полное профилирование
и speed window ещё нужны перед количественными выводами.

Бинарник `build-gpu-correctness/m31_generate`, source snapshot в
`build-gpu-correctness/source` (до добавления dense_decode_cache), совместно
собраны engine/header/CLI и GPU modules. `gpu-build-engine-quants-v2.log`.
Build option STRATA_M31_CPU_QUANTS_CUDA=ON и env M31_GPU_CPU_QUANTS=1 обязательны;
default остаётся штатный ggml CUDA, который побитен GPU-offload oracle, но
не соответствует исходному CPU oracle. Полный gate20 пока не запускался.

## Pipeline integration начата

Opt-in `M31_GPU_PREFILL_ASYNC=1` подключает PrefillUploads: два pinned staging
буфера, исходные три slices, zero padding, begin/finish вокруг полного слоя.
Для raw CPU-quants dot после finish вызывается synchronize: его default stream
не получает event wait ggml compute backend автоматически. При исключении
uploader сначала дренируется, затем все cache entries инвалидируются и free
slots восстанавливаются; uploader пересоздаётся, при неудаче recovery forward
запрещён. Default sync path и проверенная арифметика не менялись.

Сборка `codex-m31-port4-gpu-build-pipeline-v1` rc0, отдельный бинарник
`build-gpu-correctness/m31_generate-pipeline`. Engine/header/generate взяты из
прежнего самосогласованного source snapshot, обновлены только GPU cpp и новый
header транспорта. Код нового dense decode engine не смешивается.
`job-gpu-pipeline-code128-v1.json` запущен на отдельный CPU-oracle gate.

Подготовлен (НЕ запущен) `job-gpu-pipeline-suite20-c2048-v1.json`:
новый CPU-routed oracle chunk2048 → GPU pipeline chunk2048 → require-suite.
Старый golden chunk128 для сравнения с chunk2048 не используется: per-expert
CPU tiled threshold зависит от размера исходного батча. Golden128 отдельный.

### Async transport code128 PASS

`gpu-pipeline-code128-compare-v1.json`:128/128, top1=1, KL0/maxabs0,
побитно с неизменённым CPU oracle. Worker `job-gpu-pipeline-code128-v1.result.json`
rc0. Pinned async ring:57батчей,5142передачи,133227327744байт,
51819264байт staging,5140reuses. Диагностический forward18.383с, не benchmark;
ускорение не заявляется. Arithmetic/source bytes/padding/IDs сохранены.

Session unit отдельно rc127: `card_session.sh: line58: ual: command not found`.
Во время нашего активного shell coordinator обновил этот файл, что сместило
чтение его продолжающегося исполнения. Cleanup down+rollback всё же выполнен;
aictl status подтверждает обоих майнеров и дежурную8084. Следующие GPU jobs
используют immutable snapshot `gpu-card-session-port4-immutable-v1.sh` в logs.
Не смешивать успешный worker parity gate с ошибкой lifecycle controller.

Подготовлен recovery probe с compile-only `M31_GPU_PREFILL_TESTING`:
искусственная ошибка после2го queued upload, отсутствие stale entries, затем
повторные батчи/evictions против синхронного baseline. В обычной сборке fault
injection полностью исключён препроцессором. Результат ниже после выполнения.

### Recovery контракт проверен на GPU

`gpu-uploader-recovery-v1.json`:PASS. Compile-only fault injection выбрасывает
исключение после2гоqueuedcopy; после drain/recovery occupied_slots=0. Тот же
GpuExperts объект затем выполняет9батчей по3слоям,2016F32 значений побитно
с синхронным baseline;32evictions проверяют повторное использование слотов.
Два staging буфера24576B,36успешныхuploads/442368B,34ring reuses.
`codex-m31-port4-gpu-uploader-recovery-v1` и build unit rc0, cleanup завершён.
Probe обходит attention, поэтому ограничение tiny CUDA FA к нему не относится.

Все новые карточные jobs далее должны запускаться через `card_launch.py`,
создающий read-only runner snapshot и SHA manifest. Recovery был запущен до
появления launcher через собственный immutable snapshot, результаты не затронуты.
GPU source freeze для следующего окна скорости; full gate20 ещё не запускался.

## Интегрированный CMake build: gate подготовлен

После общей сборки `codex-m31-port4-build-integrated-v1` используется настоящий
`build-m31-cuda/m31_generate` (CPU-compatible CUDA option ON), включая новые
CPU packed IDs/dense decode caches, uploader и остальные интегрированные изменения.
Новый code128 `job-gpu-integrated-code128-v1.json` выполняется против исходного
CPU golden128, оба decode cache включены, measurement=false. Результат ждать.

Полный gate изменён на matched **chunk2048/decode_tail32 в обоих процессах**:
`job-gpu-integrated-suite20-c2048-d32-v1.json`. Новый CPU oracle создаётся отдельно,
все20/28723logits сохраняются, compare требует полный suite. Это одновременно
проверяет целевой batch и decode/MSA после2048; отдельный long128 пока оставлен
только для диагностики. Gate не запускать до разрешения координатора после
ACK/speed окна.

Самосогласованные источники и manifests сохранены в
`/mnt/ai/logs/m31-port/gpu-integrated-gate-sources-v1/`.
Для полного gate argv уже используют отдельные frozen binary copies с mode0555,
отдельными inode и SHA256 в `frozen-manifest.json`; последующие сборки их не меняют.
Исторические hardlinks сохранены дополнительно, но gate их не исполняет.

### Интегрированный code128 PASS

`gpu-integrated-code128-compare-v1.json`:128/128 top1, KL0/maxabs0,
побитно исходному CPU golden128. Источник — интегрированная CMake сборка,
CPU_QUANTS=1, PREFILL_ASYNC=1, cpu/dense decode caches=1.
Один диагностический forward11.742с, sourceuploads133227327744B,
hostquant234.123мс, H2D/kernel/D2H1090.579мс; это не скоростной замер.
Полный suite2048tail32 ждёт завершения окна координатора.

## Полный integrated GPU gate20 ПРИНЯТ

`gpu-integrated-suite20-c2048-d32-compare-v1.json`: scope=M1-complete-logits,
**20/20,28723позиции,top1=1,meanKL=maxKL=maxabs=0 по каждому case**.
Все logits побитно fresh CPU-routed oracle с теми же chunk2048/decode_tail32.
Юнит `codex-m31-port4-gpu-integrated-suite20-c2048-d32-v1` rc0; cleanup выполнен.
Golden128 остаётся отдельно и тоже принят. Это correctness, не speed benchmark.

## Следующий candidate: hostpool + route schedule + seed + hostreads

В текущие main sources применены проверенные opt-in host/route prototypes.
Host quantization и OAI используют один persistent GpuHostExecutor. Runtime
без M31_GPU_HOST_THREADS остаётсяserial1; явное32 создаётparallelpool. Worker
FP environment/MXCSR наследуется и sticky flags возвращаются вызывающему потоку.
GPU route worklist включается только M31_GPU_EXPERT_SORT_ROUTES=1,n>=32;
меняется расписание, input IDs/full histogram/output indices не меняются.
Seed API/env и protected hot region сохранены. THPheader новый,self-consistent,
но M31_ARENA_THP в GPU candidate gates выключен.

Общая CMakeсборка v1 rc0: CPUtiny/comparatorPASS,22server/vocabtestsPASS,
385hostCPUchecksPASS,48route-layoutchecksPASS. После CPU HOSTREADSOURCEFREEZE
инкрементальная v2 rc0, tiny с M31_REUSE_HOST_READS=1 побитно; engine/frontends
пересобраны согласованно. Сохранены отдельные executable copies и source hashes
в `gpu-candidate-gates-v1/manifest.json`, включая latest m31_serve fdb source.

`gpu-candidate-probe-{default,host32,route,combined}-accept-v1.json`:
все4варианта PASS, каждый36cases/117стадий/21307392F32 побитно CPU.
Probe теперь проводит full FFN OAI через тот же host executor.
Новый корректный routeproofv2 (отдельные процессы envрежимов) PASS36cases;
v1 имел8ожидаемых CPU-reference ошибок из-за staticonce_flag env caching,
при этом sorting сам побитен во всех36. Существовавший frozen v1 ELF не менялся.
Seed F32 API/env probe тоже PASS, включая recovery и partial seed terminal.

Запущены только два новых real128 gates: combined host32+route+hostreads+async
с seed0, затем combined+seed20 через fresh sealed preflight receipt.
`job-gpu-candidate-combined-code128-v1.json`, `job-gpu-candidate-seed20-code128-v1.json`.
Четыре отдельных полных startup/pack не запускаются. Численных/скоростных
результатов этих двух gates пока нет; runtime performance окно у координатора.
