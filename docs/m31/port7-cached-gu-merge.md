# Port7: contiguous GU и CPU rank-outer merge

Все новые режимы opt-in. Принятый v8 остаётся default. Native cached GPU путь ранее отклонён по точности; здесь используется точная CPU-совместимая математика.

## Исходная ordinary точная кешированная генерация

`/mnt/ai/logs/m31-port/port7-p7v9-ordinary-exactcache290-v1/compare.json`: один frozen P7-v9 ELF, prompt3324 + greedy128, cachedOFF25,668 т/с → exactcachedON23,736 т/с (−7,53%). 128 IDs идентичны, primer108,171мс включён в TG. Actual hit12,222%, join1,286мс/токен включает ожидание GPU и host copies; это не цена только CPU merge.

## GU без новых CUDA ядер

Слоты содержат gate/up одинакового типа и строго соседние полные матрицы. Два original матричных вызова можно заменить одним multiply с rows=2F. Исходные строки весов, reduction tree, scale/OAI и down сохраняются. Неподходящий type/offset/padding возвращает старый путь. `M31_GPU_CACHED_GATE_UP=1`, default0; после измерения разрешён только nongroup. Grouped1 сохраняет прежнюю реализацию.

GPU компонент `/mnt/ai/logs/m31-port/port7-cached-gate-up-component-v1/component-result.json`: **60 cases / 1 474 560 F32 exact**, original IQ3/IQ4/NVFP4, real cached assignments, GROUP0/1, padding-gap fallback. Восемь физических реплик и reuse distance > actual L2=96MiB; это capacity thrash, фактические HBM/L2 транзакции не измерены.

Nongroup n1 helper median: L3 422,01→301,23мкс (1,401×), L10 249,50→193,90 (1,287×), L52 257,27→201,71 (1,275×). K8 nongroup1,30–1,36×. Grouped K8 неоднороден: L3 553→857мкс, L10 923→895, L52 587→507. Поэтому GU не включается в grouped. Эти числа не являются TG или оценкой итоговой генерации.

## CPU merge

Новый неиспользуемый pure helper `cached_decode_merge_rank_outer.hpp`: pointers original ranks выбираются один раз, rank0 memcpy сохраняет signed zero; остальные ranks прибавляются к contiguous output по порядку. FP reassociation/FMA не применяется.

`build-port7-cached-merge-rank/v1/check-result.json`: **1281 cases / 1 648 391 F32 exact**, четыре FP rounding mode, cancellation/signedzero/subnormal, E1/7/31/256/6144, used1/2/4/8 + fallback129, masks, invalid inputs unchanged. Реальное тело ASM содержит addps SIMD.

Чистый CPU timing `/mnt/ai/logs/m31-port/port7-cached-merge-rank-timing-v1.json`: tasksetCPU0, E6144/used4, все14 masks с1–3hits, alternating blocks. Old11,77–18,39мкс → rankouter1,81–1,95мкс (6,43–9,63×). Экономия лишь10–16мкс на hit-layer; она не удаляет GPU wait/full contribution copies из join. `M31_GPU_CACHED_RANK_OUTER=1` принят parent для следующего combined кандидата, default0.

## IQ3 интеграция, компонент завершён

После отдельного IQ3 IMMA component PASS ядра перенесены дословно с переименованием namespace/include в `gpu_iq3_lane_exact.cuh/.hpp`. Existing CUDA TU `gpu_cpu_quants.cu` включает их один раз; CMake и новый CUDA TU не нужны. `GpuDeviceDots::multiply_iq3_n1` выполняет прежнюю GPU CPU-compatible Q8_K quant и два exact stages. Старый multiply ABI сохранён. `M31_GPU_CACHED_IQ3_IMMA=1` действует только на IQ3/count1/!tiled; остальные вызовы старые. Raw Q8−128 исключён из доказанной области, finite CPU quant выдаёт[-127,127].

Private scratch pre-reserved4 718 592B для geometry6144/3072, входит в device_bytes и lives с worker. Lease/reset/error drains actual worker stream до переиспользования слота. Combined helper GPU component завершён: `/mnt/ai/logs/m31-port/port7-cached-imma-gu-component-v1/component-GU0.json` и `component-GU1.json`, тот же ELF33b8afd12ae5c1c9b382135c790d929cce58ec738662d2885fe92d2ecfa2655a. **Обе руки по60cases/1 474 560F32 точны** (итого120/2 949 120). GROUP1 joinedGUcalls0, padding-gap fallback проверен, actualIMMAcalls>0. Wholehelper nongroupn1: GU0 L3 456→297мкс, L10 245→148, L52 253→170; GU1 L3 461→267, L10 266→166, L52 221→169. Между GU0/GU1 отдельные последовательные руки с различными baseline times, прямой causal GU1vsGU0 speedclaim не делается. Матричный3× не переносится на wholehelper; joins/OAI остаются. Root разрешил freezev11. Ordinary128 IDs/TG и default preservation пока ожидаются.

## Ordinary dense shared FFN

Shared gate/up уже объединены в v8: native-gpu profile `M31_GPU_PROJECTION_CONCAT=1`, Weights::load пакует m31_ffn_gu_shexp, n1 ffn запускает один matmul и два views. Повторное объединение не добавляет ускорения. F32 indexQK также packed640×6144;60слоёв=.9437GB, общий F32 MMV1.123GB/.976мс≈1.15TB/s. Existing F32 kernel256threads,640CTAs,12float2 итераций; изменение thread/reduction geometry потребует новой точности. Даже идеал2TB/s даёт менее0.42мс/токен для всех F32 MMV; это не самостоятельный рычаг+10% TG.

## Ограничения объяснения CPU starvation (source review)

Старый ordinary P7-v9 лог сообщает threads80/poll50/strict0/affinity_cpus0;
в P7-v11 profile M31_CPU_AFFINITY также не задан. В ggml пустая cpumask не
вызывает sched_setaffinity, поэтому гипотеза «default worker наследует один
CPU, закреплённый pool» не подтверждается этим исходником. При explicit
M31_CPU_AFFINITY действует ранее проверенное восстановление worker mask.
Default worker наследует process mask и остаётся под управлением scheduler.
79 secondary workers могут spin до6 553 600 relax iterations при poll50;
специальный CPU/priority для cached worker не резервируется. На96 физических /
192 logical CPU суммарных cores формально достаточно, но SMT/scheduler delay
в отдельных слоях возможен. Точного sched trace пока нет.

Host expf/scales выполняются worker между двумя GPU joins, его wall OAI
включает возможное дескедулирование. Старый модельный профиль: OAI122,200мс /
3567 experts =34,26мкс/expert (0,955мс/token), что сравнимо с idle standalone
component (~15–39мкс/expert на выбранных input rows). Поэтому aggregate не
показывает gross starvation; tails остаются непроверенными.

CPU miss8,594мс/token в старом hybrid footer относится только2806 hit-jobs,
а routed26,971мс/token охватывает все57layers плюс seed/submit/CPU/GPUjoin и
host merge. Эти числа нельзя трактовать как два измерения одного CPU участка.
Worker6,378мс/token перекрывается с CPU misses и не складывается с ним напрямую.
