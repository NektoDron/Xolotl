# Port7: exact cached same-expert countM — изолированный компонент

После warm/cold bitproof и component gain root разрешил live P7-v9 перенос
с default0. Модельная проверка нового режима ещё ожидается; frozen P7-v8
не менялся. Исходный проверенный patch/source:
`/mnt/ai/logs/m31-port/port7-cached-grouped-proposal-v3/`.
Build `build-port7-cached-grouped/v1` rc0, ELF SHA256
`148cd6f1f315373c7ac3b709c33999cf2a2d4726fe3796cfe84bf93a00bed417`.
GPU job подготовлен `port7-cached-grouped-component-v1`, measurement=true,
передан validation и прошёл чистое окно: 27cases/712 704 weightedF32 побитно. Default `M31_GPU_CACHED_GROUPED=0`.
Native hot cached math по-прежнему FAIL и не принимается этим proposal.

## Измеренный исходный профиль и изменение

В `port7-verify-trace-v6/trace.json` worker stream14 kernel union:
257,744ms/9=28,638ms/K8, из них multiply_kernel27,164ms/K8; quant+metadata
вместе1,474ms.5970 multiply/9=663,3 вызова, то есть221,1 cached assignments/K8
и около12,1% actual verifier routes. Это GPU kernel duration из профайлера,
не om_bench и не standalone latency. CUDA launch API на private worker также
занимал112,191ms/9≈12,466ms/K8; суммы host/GPU времени нельзя складывать.

Existing `GpuDeviceDots::multiply(type,weights,columns,rows,count,...,false)`
уже обрабатывает несколько входов одного эксперта. Предлагается stable group
по slot: GU/down запускать с count=число токенов этого эксперта. Вместо
metadata+quant+dot ×3roles×H получается то же ×3roles×G (unique experts).
Повторение внутри одного token запрещено входным контрактом; G group count≤8.

Для IQ3/IQ4 `original_tiled=false` принудительно оставляет scalar lane8 AVX2
contract независимо отcount≥8. ДляNVFP4 tiled всегдаfalse; тот же Q8_0 packed
input [columns,count], тот же FMA/shuffle tree каждого output. Count меняет
только grid/input row. Никаких новых CUDA kernels/expf/другой arithmetic.

CPU OAI/scales/fenv сохранены дословно. Gather использует свободные до down
`host_down` и `device down`; после OAI свободный `host_gate_up` хранит output
в исходном assignment/rank порядке. ОграничениеE≤2F (уM31 E=2F), иначе
proposal mode отвергается. Дополнительные pinned/device matrices не нужны.
Сама quant/metadata scratch заранее reserve count_max8, иначе cudaFree/realloc
случился бы в горячем пути. Device bytes2 169 720→2 218 860, pinned неизменны
2 162 688. Если повторений slot нет, используется старый baseline run.

## Компонент A/B

Fixture `/mnt/ai/logs/m31-port/port7-cached-grouped-fixture-v1.json` содержит
27 реальных held greedy hot290 assignments: N1..8 дляL3/L10/L52 и по одному
K8 maxreuse. Никаких synthetic all-hot routes. Активации берутся из ранее
записанного реального L3 x, а не из hidden именно выбранного layer/position:
это компонент арифметики/latency, не модельная проверка.

Baseline/grouped находятся в одном ELF/процессе, оба сравнивают weighted terms
побитно с CpuFusedDecode, включая original full rank route weights/NVFP4 scales.
Далее4warmups и20alternating A/B repeats; отчёт содержит main wall median/mean,
worker/GU/OAI/down mean и фактические jobs/groups/multiply calls. Два join
сохраняются. Worker time включает stable grouping; wall включает submit/wake,
копии metadata и finish. Это настоящее время helper, но не TG/Engine latency.

Physical layout зафиксирован отдельно:
`port7-cached-grouped-component-v1/physical-layout.json`. Запрошенные GPU pools
90 243 072 /108 380 160 /95 551 488B дляL3/L10/L52, один слой одновременно.
Пик pools +обаhelper112 768 740B; это requested cudaMalloc bytes, а не новое
NVML измерение driver residency. В каждом повторяемом case raw weight working
set21,676–65,028MB, меньше96MiB L2. Поэтому первый timing warm-cache. Если
получится≥15–20% gain, следующий обязательный этап — отдельный cold/L2-thrashing
A/B перед live integration. Второй ELF до этого результата не создаём.

## Warm result и обязательный cold этап

`port7-cached-grouped-component-v1/component-result.json`: K8 median wall
L3 H8/G2:1150,13→530,49us (2,168×), L10 H9/G2:1453,83→863,16us (1,684×),
L52 NVFP4 H4/G2:721,66→532,98us (1,354×). Maxreuse L10 H8/G1:1298,55→711,41us
(1,825×); L52 H4/G1:738,11→436,93us (1,689×). No-duplicate cases около1,00×,
отклонение median wall примерно±1%. Все27кейсов совпали сCPU побитно.

Порог15–20% пройден; root разрешил обязательный cold capacity-thrash A/B.
`build-port7-cached-grouped-cold/v1` собран одним CXX TU, helper object точно
тот же. ELF SHA256 `d9e160fc57908f18931b4a2680895e1b42fe80eea0b2011bb1f75bcb7d0dfa53`.
Новый main циклически выбирает8 физических копий неизменённых весов; runtime
проверяет `(8-1)*active_payload > cudaDeviceProp.l2CacheSize`. Это capacity
thrash, не прямое измерение L2 miss counters. Peak requested pools+обаworker
871 429 860B. Никаких новых CUDA kernels. GPU measurement job
`port7-cached-grouped-cold-component-v1` передан validation, результата ещё нет.
Cold результат и разрешённый перенос описаны ниже.

## Cold result и разрешённый перенос P7-v9

`port7-cached-grouped-cold-component-v1/component-result.json`: все27 cases и
712 704 weighted F32 побитно совпали с CPU. Actual L2=100 663 296B; reuse
distance151 732 224–455 196 672B превышает его. Peak requested GPU pools
867 041 280B плюс оба helper4 388 580B. K8 median wall:

| Case | Baseline, us | Grouped, us | Ratio |
|:---|---:|---:|---:|
|L3 H8/G2|1439,67|789,12|1,824×|
|L10 H9/G2|1552,99|929,19|1,671×|
|L52 NVFP4 H4/G2|800,18|637,29|1,256×|
|L10 maxreuse H8/G1|1364,69|738,73|1,847×|
|L52 maxreuse H4/G1|774,41|489,10|1,583×|

Сумма трёх K8 median cases3792,84→2355,61us,1,610×. Это три выбранных
компонентных кейса, не взвешенная модель и не TG. No-duplicate cold controls
имеют разброс ratio0,955–1,180; там ускорение не заявляется.

После этого root разрешил live перенос ровно tested helper.hpp/.cpp: candidate
SHA из proposal совпали. P7-v8 frozen не менялся. Default0 сохраняет старый
run/reserve1, group1 резервирует count8 только для batched worker. Новый
GpuExperts::print_cached_batch_profile вызывается владельцем Engine после
verify-cache при profiling; при нуле grouped jobs он молчит. Поля jobs/H/G,
multiply calls, фазы и bytes подтверждают actual execution. G — сумма distinct
expert groups по отдельным job, а не число разных expert IDs за всё время.

Config и GpuExperts::Stats layout не менялись. Tested GpuCachedExact::Stats
имеет4 добавленных counter в хвосте: нужен matching rebuild helper+GpuExperts;
старый precompiled caller с новым helper возвращаемым struct несовместим.
Root candidate builder пересобирает согласованные единицы. CMake/CUDA kernels
не менялись. Следующий обязательный этап — P7-v9 same-ELF flag0/1 modelgate,
greedy и end-to-end cpu_ms/hybrid/join/TG, включая primer. Component gain не
доказывает TG gain: прежний worker мог частично скрываться CPU misses.
