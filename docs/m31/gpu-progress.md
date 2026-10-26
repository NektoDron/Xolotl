# GPU correctness, port4

**Итог сессии:** два раздельных полных численных контракта проверены (frozen
CPU-exact и stock GgmlGPUoracle), но выигрыша M2 нет. Best native CPU-overlap:
PP183.3/TG22.155; stock GPU:PP95.35/TG20.75; exact GPU:PP46.2/TG21.375.
Референс llama.cpp1GPU:PP4K425 (32K433),TG4K23.33. Новые модели/сборки/замеры
после этих окон не запускаются; M3 не продолжаем. Финальные scope/ограничения
и разбор stock window находятся в последнем разделе ниже.


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

## Итог GPU speed4K: M2 отрицательный, остановка

Строгое отдельное окно без сборок/heavyCPU, одно принятое frozen candidate плечо:
`port4-gpu-candidate-speed-v1/manifest.json`. Порядок: generic HTTP109 PASS,
затем исходный `~/om_bench.py`, OM_OFF={"reasoning_effort":"low"}, warmup/4K/4K.
Session и внешний unit rc0, cleanup выполнен. Никаких дополнительных экспериментов
для объяснения результата не запускалось.

| Повтор | Prompt | PP,t/s | TG,t/s | Gen | Prompt,s | Wall,s |
|---|---:|---:|---:|---:|---:|---:|
|4K №1|4350|46.2|21.57|256|94.2|106.1|
|4K №2|4352|46.2|21.18|256|94.1|106.2|

Медианы TSV:PP46.2,TG21.375. Против лучшего llama.cpp1GPU PP425/TG23.33:
10.87%/91.62% скорости соответственно. PP также ниже принятого CPU-плеча124.5
(37.11% его скорости). Выигрыша нет; полный M2 benchmark16K/32K и M3 не продолжаем
по исходному критерию Андрея. F16KV8192 здесь отличается от refQ8_0KV65536;
точное клиентское тело low, скрипт и корпус сохранены/хешированы.

Telemetry:peakGPU0=19475MiB; peak суммы RSS engine+adapter+helper317.02GiB,
включая file mappings и отдельную точную RAM-арену. Это RSS, не private RAM.

### Что измерено именно в этом speed window

`port4-gpu-candidate-speed-v1/profile-analysis-final.json` содержит расчёт по
native stage counters, отдельно для каждого из двух4K запросов:

- Prompt94.229/94.099с. Forward-сумма94.151/94.058с.
- Dense stage2.117/2.046с; внутри GPU compute0.869/0.869с,
  dense read0.950/0.880с. CPU residual0.605/0.594с.
- Остаток routed-MoE+control91.350/91.379с, около97%prompt времени.
  Это разность forward и известных stage counters, а не измерение PCIe/kernel.
- Каждый полный2048chunk занимает39.58–40.05с; хвост254/256 —14.35/14.43с.
- Generation11.820/12.039с; суммарный interactive ACK wait58.9/49.8мс,
  sampling92.6/91.6мс. Grammar checks/rejections/wait=0. IPC/sampling по этим
  таймерам не объясняют ни94с префилла, ни основную задержку генерации.

**Отдельных CPUquant/OAI/H2D/kernel/D2H чисел именно для этого HTTP speed run нет.**
GpuCpuQuants печатает host_quant и H2D_kernel_D2H только в destructor, а текущий
RPC.close завершает C++engine SIGTERM. В server.log0таких строк и0финальных
uploader summaries. Числа одношаговых code128 correctness runs выше не переносим
на этот benchmark. Остаток91.4с включает packing/zeroing/upload/sync, quant/OAI,
CUDA dot, activation/output transfers, allocations/route metadata и ordered reduce;
разложить его на эти компоненты данными этого окна нельзя.

### Что следует из кода, но не является отдельным измерением

Проверенный CUDA dot — точный скалярный прототип:8lanes на одну пару
(output row,route). Каждая такая пара снова декодирует исходные IQ3/IQ4/NVFP4
blocks. Stable sort сближает маршруты одного эксперта, но не создаёт совместный
M×N tile, shared weight decode или гарантированное переиспользование между
CUDA blocks; возможные L2 hits не устраняют повторные integer decode инструкции.
Загрузка всех выбранных слотов заканчивается до execute, так что нынешний uploader
перекрывает CPU packing с H2D, но не H2D с MoE compute. При следующем token chunk
последовательный обход60слоёв снова вытесняет большую часть рабочего набора.
Эти свойства объясняют направления оптимизации; профиль не доказывает, какая
доля91.4с приходится на каждый из них. Утверждать, что всё время —PCIe, нельзя.

Следующий порт: CPU-compatible batched MMQ/DP4A/IMMA для IQ3/IQ4, с tile по
output rows и токенам одного эксперта, unpack исходных весов один раз на tile,
shared staging и повторным использованием quantized activations. Сохранить
исходные full-batch histograms (>=8CPUtiled), exact integer subblock scales,
накопление по256 и CPU FP/FMA порядок. Для редких<8 и NVFP4 оставить проверенный
fallback, пока отдельный быстрый kernel не докажет тот же контракт. Тот же GGUF,
без requant/prune. Сначала component probes и code128, затем полный20/28723 и
только после них новое окно скорости. Следующий ресурс следует тратить на
матричное переиспользование, а не очередные hostpool настройки.

### Граница численных доказательств и source fingerprint

Полный20/28723 относится к **предыдущей frozen integrated версии**, chunk2048/
decode_tail32 в обоих процессах. Дополнительно `gpu-fullgate-bytecmp-v1.json`
проверил ОС `cmp`:20/20rc0,5746438272F32 значений — реально побитно, не только KL0.

Измеренный более новый candidate включает hostpool32, route sort, host reads,
seed20 и новый совместимый arenaheader (THPoff). Для него проверены4×117стадий
standalone и два realcode128 (seed0/20), top1=1/KL0/maxabs0. **Полный20 для этой
новой комбинации не повторялся**; не подменять его старым fullgate. HTTP109 пройден.

`gpu-candidate-gates-v1/manifest.json` хранит исходники/ELF SHA и независимые copies:

- generate SHA256 `744f710ef55c3ed33cb141448464dbc20f9c36f3eee4e63638565901a84c4432`;
- serve SHA256 `10644555845043ae8e2ff3c35a24bbb34d97d276aaea47fbfbcb1ae22c0d3b36`.

Speed manifest дополнительно фиксирует PythonACK/helper/client/corpus/ref/rank
fingerprints, всю CLI/env, sealed seed preflight и telemetry/profile paths.
Default host threads1, sort/seed/hostreads/THP opt-in; production не менялся.

## Дополнительный диагностический потолок штатного CUDA MMQ

Координатор отдельно разрешил исключить цену CPU-compatible scalar kernel из
M2 решения: те же frozen ELF, но M31_GPU_CPU_QUANTS=0. Это штатный ggml CUDA
`mul_mat_id`, async/hostreads/seed20 сохранены; hostpool/route-worklist exact-kernel
здесь не используются. **Этот контракт назван GgmlGPUoracle и не заменяет M1CPU.**
Его CUDA activation quantization отличается от принятого CPU oracle: исторический
128 comparison даёт top1=.78125,meanKL=.77857093. Смена oracle для этой диагностики
не делает эти CPU расхождения пройденным M1 gate.

Fresh `gpu-ggml-gpuoracle-code128-compare-v1.json`:PASS128/128,KL0/maxabs0 против
сохранённого GPU-offload oracle128. Session rc0. После этого запущен отдельный
`job-gpu-ggml-gpuoracle-suite20-c2048-d32-v1.json`: новый oracle-op-offload1 и native
CPU_QUANTS0,chunk2048/tail32,все20/28723. Итоговый scope специальный
GgmlGPUoracle-complete-logits, не M1-complete-logits. Пока результат ожидается.
Завершённый code3000 промежуточно совпал численно на всех3000позициях, но это
не заменяет полный suite. Stock speed не запускался; только после PASS и нового
окна координатора. Источники/библиотеки не менялись и не пересобирались.

Read-only sidecar `gpu-ggml-gpuoracle-policy-v1.json` фиксирует SHA job/runner/ELF,
разные CPU/GPU контракты, env, seed, исходные CPU proof paths. Числа ранее
измеренного exact-mode46.2/21.375 остаются действительными для своего candidate.

### GgmlGPUoracle full20 PASS, скорость ещё не измерена

`gpu-ggml-gpuoracle-suite20-c2048-d32-compare-v1.json`:20/20,28723строки,
top1=1,mean/maxKL=0,maxabs=0; scope=GgmlGPUoracle-complete-logits.
Native CPU_QUANTS0 совпадает с fresh oracle-op-offload1 при chunk2048/tail32.
Внешний unit rc0, cleanup выполнен. Это численное совпадение по полным logits;
дополнительный файловый bytecmp этого нового режима не выполнялся.
CPU-exact fullgate/bytecmp остаются отдельным доказательством прежнего режима.

Подготовлен один reviewable stock speed job (не запускался при подготовке):
`port4-ggml-gpuoracle-speed-v1/m31-port4-ggml-gpuoracle-v1-card-job.json`.
Физически те же frozen ELF/runtime, меняется только envCPU_QUANTS0 и отключаются
неиспользуемые hostpool/route flags. Seed20/arena170/cache6,ctx8192/chunk2048/t64,
F16,bothcaches,async,hostreads сохранены. ExactOM_OFFlow,generic109→warmup/4K/4K.
Stock speed пока ждёт нового отдельного окна координатора. Его manifest явно
называет альтернативный контракт и содержит известное отличие от CPU oracle.

## Заключительный stock MMQ benchmark и отрицательный M2

Официальный новый speed window завершился rc0, generic HTTP109 PASS,
исходный om_bench warmup/4K/4K,256tokens,length в обоих измерениях.
Результат `port4-ggml-gpuoracle-speed-v1/stock-summary.json`;
разбор `port4-ggml-gpuoracle-speed-v1/profile-analysis-final.json`.

| Путь | PP4K,t/s (медиана) | TG4K,t/s (медиана) | Численная граница |
|---|---:|---:|---|
| Best native CPU-overlap |183.3|22.155|Отдельный CPU-overlap gate3000/d32; не frozenfull20|
| Stock ggml CUDA, GgmlGPUoracle |95.35|20.75|Full20/28723 с matching GPU-placement oracle|
| Exact CUDA + host32/route/reads/seed20 |46.2|21.375|4×117 component stages и code128 seed0/20; не новыйfull20|
| llama.cpp best1GPU ref |425|23.33|Конфигурация отчёта ref|

Stock повторы PP96.5/94.2,TG20.92/20.58; prompt4348/4347,
prompt45.0/46.1с,wall57.3/58.6с. Медиана stock PP в2.064раза выше scalar exact
варианта, но составляет22.44%отref425 и52.02%отbest native CPU183.3.
Stock TG составляет88.94%отref. Это наблюдаемое сравнение **целых путей**:
меняются не только CUDA kernel, но activation quantization, GLU placement и
обмен промежуточными данными; выделять из него причинный эффект одной операции
нельзя. Ни stock путь, ни exact путь M2 выигрыша не дали; дальнейшие feature
loops/full M2 bench/M3 прекращаются по условию Андрея.

### Измеренные части stock profile

Для двух официальных4K запросов сумма prefill forward44.966/46.077с.
Dense stage2.136/2.066с (внутри GPUcompute0.862/0.862с,
dense read0.976/0.908с),CPUresidual0.604/0.599с.
Неатрибутированный routed+control остаток42.151/43.358с —93.58%/93.99%
полного prompt времени. Для exact варианта аналогичный остаток91.350/91.379с
(~97%). Эти остатки измерены вычитанием известных stage counters, но не являются
таймером PCIe. В stock mode CPUquant/OAI из GpuCpuQuants вообще не исполняются;
GPU quantization/GLU, MMQ, packing/upload/sync, allocator и служебные операции
содержатся внутри общей MoE части. Раздельного PCIe/kernel split здесь нет.
Данные не позволяют назвать весь остаток загрузкой весов или всем CUDA compute.

Медианы показывают, что scalar exact путь действительно имеет существенную
совокупную цену относительно штатного MMQ, однако и штатный backend не достигает
скорости референса. По коду остаются serial staging/zero+copy, ожидание всех
uploads до execute и отсутствие слоевого переиспользования на весь prompt;
их отдельная доля не измерена. Следующий отдельный порт должен рассматривать
CPU-compatible tiled MMQ/DP4A/IMMA и матричное/слоевое переиспользование, сохраняя
проверенный integer-scale/FMA контракт, а не объявлять hostpool исправлением
основного ограничения. Новых экспериментов для этих выводов не делалось.

Stock telemetry:peakGPU0=19621MiB,peak суммы RSS316.55GiB (с file mappings,
engine+adapter+helper). Source/env/ELF/runtime fingerprints записаны в speed
manifest; используется тот же новый frozen candidate ELF, что в candidate128
и stock full20. Прежний CPU-exact full20 использовал ДРУГОЙ старый frozen ELF;
не переносить его доказательство на новую exact-mode комбинацию автоматически.

### Контракты и ограничения сравнения

1. Frozen CPU-exact20/d32 — top1=1,KL0,maxabs0 и отдельный bytecmp20/20,
   5746438272F32. Это прежняя frozen integrated версия.
2. Stock GgmlGPUoracle20/d32 — top1=1,KL0,maxabs0 против oracle-op-offload1;
   это отдельная standard-CUDA activation policy. Дополнительный bytecmp stock
   файлов не запускался. Известный128 CPU-vs-stock diff .78125top1/.77857meanKL
   остаётся непройденным **для CPU контракта**, он не скрыт сменой oracle.
3. Более новый exact host32/sort/reads/seed candidate проверен component probes
   и code128, а CPU-overlap — своим3000/d32 gate. Старый frozenfull20 не выдаём
   за доказательство полного20 для этих новых комбинаций.

Общими в speed comparison являются исходные GGUF, GPU0/host hardware,
клиент/корпус и exact OM_OFF low добавка к запросу. **Это не полностью matched
optimization arm:** все native варианты требуют GGML_CUDA_DISABLE_FUSION=1;
reference scripts не показывают явного DISABLE_FUSION, тождественность этой
настройки не установлена. NativeF16KV/8192 также отличается отrefQ8_0KV/65536.
Нельзя приписывать весь разрыв одному движку при идентичных настройках.

Финальная read-only проверка: своих активных GPU/build units нет, aictl active=null,
`m31-xolotl`/`m31-xolotl-check` отсутствуют, исходные `m31`/`hxq3k` сохранены.
GPU0майнер, GPU1Pearl и дежурная8084 работают. Никаких дополнительных моделей,
сборок, численных/скоростных тестов или больших логитных сравнений после окна
не запускалось.
