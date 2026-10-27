# Port7: фактический flat CPU profile n1 decode

Выполнен один clean profiler run `codex-m31-port7-cpu-flat-perf-job-v3`, frozen canonical V8 Engine, corpus3324 +128 ordinary continuation IDs (teacher forcing). Сборок/других тяжёлых CPU работ не было. Инференс шёл через aictl/card lock. Отдельный collector `User=nekto` имел толькоCAP_PERFMON и target PID2158628; sysctl/THP/callgraph/общесистемного сбора не было. Collector и native завершились0,34228samples,lost0; samples принадлежат только этому PID,87TIDs из91, существовавших приattach.

Perf: cycles:u,99Hz, flat self symbols. Init/PP исключены: первый n1 marker3324 запускает disabled collector, enable ACK получен при last observed3339; disable ACK — при3451. Attach/control задержка отражена в receipt, не предполагается точное покрытие каждого forward. Python monotonic и perf event timestamps имеют разные абсолютные clock bases; их нельзя напрямую совмещать в GPU frames.

| Категория self user cycles | Доля |
|---|---:|
| secondary thread wrapper | 39.29% |
| IQ3_XXS dot | 38.04% |
| ggml_barrier | 16.00% |
| IQ4_XS dot | 3.73% |
| CpuFusedDecode::compute glue/OAI/scales | 1.24% |
| NVFP4 dot | 0.93% |
| named activation quantizers | ≈0.08% |
| named expf/libm | ≈0.05% |

Весовой unpack/dequant находится ВНУТРИ vec_dot и входит в42.70% dot cycles. Named activation quantizer percentage не означает, что вся quant-format arithmetic стоит0.08%. malloc/free/memcpy/mutex self symbols малы, по округлённому flat report порядка0.00–0.01% каждый. Метаданные не выглядят главным CPU throughput bottleneck этого n1 run.

Сделана read-only `perf annotate` двух спорных symbols. В secondary_thread≈98.34% егоlocalperiod попадает в конкретную петлю pause/work_pending/generation checks (`0xa274d0..0xa27504` с исключением dispatch path). В barrier≈98.38% localperiod — pause/load generation/cmp/branch (`0xa20bb0..0xa20bb6`). Итого≈54.4% всех sampledUSERcycles действительно в подтверждённых spin-loop instructions, а не полезном dot.

Это НЕ54.4% wall и НЕ готовая оценка потенциального TPS. Secondary polling идёт также пока GPU/dense работает; убрать его — прежде всего снизить бесполезную CPU активность, а не удалить соответствующий wall interval. cycles:u не видит blocked kernel waits/off-CPU duration. Barrier spin не показывает, кто задержал фазу: полезная работа последнего worker, стартовая неравномерность/миграция, малое число quant workers или собственный fan-in. Dot cycles включают memory/TLB stalls; из этого события нельзя отделить arithmetic от DDR/TLB.

Следующий шаг обоснован: приватная phase instrumentation frozen CpuFusedDecode, per-thread GU/down work time и barrier wait с rowcounts. Только если реальный work tail существенен, рассматривать dynamic/cyclic scheduling. Poll0 уже проигрывал component BW, поэтому отключать spin по этим процентам нельзя. Отдельный THP reserve допустим по своему плану; реальный smaps подтверждает4Kpages/AnonHugePages0, но эффекта на throughput ещё нет.

Артефакты:

- `/mnt/ai/logs/m31-port/port7-cpu-flat-perf-job-v3/result/receipt.json` — gate/capability/PID/argv.
- `report.txt`, `full-report.txt`, `script.txt`, `flat-analysis.json` в том же каталоге.
- `secondary-annotate.txt`, `barrier-annotate.txt` — инструкция-level self evidence без callgraph.

Это диагностический CPU profile, не om, не качество/новая модельная конфигурация и не новая backend-portability приёмка.
