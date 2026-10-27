# UD-Q2_K_XL: opt-in штатных CPU dot в прямом FFN

Волна Q2 закрыта как **NOT_ACCEPTED**. CPU components прошли exact проверки,
но baseline full quality97/100 и 17/23 не достигла tools≥19. Cached-E3
не прошёл clean ordinary long numeric gates; новых cached bench/quality
не было. Полный scope и отдельные receipts — `q2-quality-audit.md`.

Новый флаг `M31_CPU_STOCK_EXPERTS=1` разрешает IQ2_XS в существующих
`CpuDirectDecode`, `CpuFusedDecode`, `CpuBatchedVerify`. По умолчанию IQ2
по-прежнему идёт в штатный routed ggml graph; прежние IQ3_XXS/IQ4_XS/NVFP4
возвращают поддержку до чтения нового флага. Флаг разрешает только IQ2_XS,
не произвольные форматы. Скалярная математика и порядок суммирования не менялись.

Gate/up используют `ggml_vec_dot_iq2_xs_q8_K` через CPU traits и штатную
Q8_K-квантизацию активаций. `from_float` для **весов** не требуется: оригинальные
байты GGUF передаются напрямую; у IQ2_XS callback отсутствует штатно. Pair mode
для IQ2 возвращается к двум независимым stock nrc=1 dot; IQ3/IQ4 down может
использовать прежние точные pair helpers при явном включении этого режима.
Это поддержка общего FFN wrapper, не новое IQ2 unpack-once ядро.

IQ2 tensor eligibility проверяет положительные и int-sized columns, делимость
на блок веса и activation, rows до 65536, experts до 128, ne3=1, упакованные
nb0/nb1 и nb2 не меньше размера эксперта. Паддинг между экспертами разрешён,
поэтому исходные GGUF и padded arena views имеют одну арифметику.

Engine init использует `CpuDirectDecode::supported_type(ggml_type)`;
`CpuDirectDecode::supported_weight` и `supported(wg,wu,wd)` применяют
дополнительные проверки геометрии перед вызовами.

## Проверка

`tools/m31/q2_cpu_check.cpp` строит независимый per-token stock ggml graph:
`mul_mat_id(gate/up) → swiglu_oai → mul_mat_id(down) → route weights →`
последовательная сумма в исходном порядке ranks. Он не вызывает wrappers
для получения ожидаемых значений. Оригинальные mmap GGUF веса слоёв 3/52/59
охватывают IQ2/IQ3, IQ2/IQ4 и protected IQ3/IQ4; активации синтетические конечные.
Никакого переквантования production IQ2 весов нет.

Проверяются N1/K2/4/8, 4 routes, одинаковые/пересекающиеся/раздельные expert IDs,
малые/обычные/большие активации и отрицательный route coefficient. Сравниваются
полные F32-векторы и weighted contributions: direct, fused modes0/1/2,
batched modes0/1/2, all-miss/all-hit/mixed masks и возврат к unmasked после них.
Явные eligibility guards проверяют unset/0/01/1, сохранение старых 3 типов,
8 испорченных геометрий и допустимый padded expert stride.

Запуск CPU без карты через systemd, сборка shared и proof exclusive под общим
`/mnt/ai/logs/m31-port/.build.lock`:

```bash
bash tools/m31/q2_cpu_build.sh /mnt/ai/Xolotl-m31/build-q2-cpu/v1
bash tools/m31/q2_cpu_run.sh /mnt/ai/Xolotl-m31/build-q2-cpu/v1 /mnt/ai/logs/m31-port/q2-cpu-component-v1
```

Оба каталога должны быть новыми. Header/source snapshot сохраняется до сборки;
`build.sha256` включает ELF, snapshot и stock ggml libraries. Runner сохраняет
commands, SHA256, JSONL и результат в `receipt.json`. Job
`codex-m31-q2-cpu-component-v1` завершён с rc0; receipt:
`/mnt/ai/logs/m31-port/q2-cpu-component-v1/receipt.json`.

Результат **PASS**: 36 конфигураций,135 эталонных token rows,1008 полных
побитных сравнений векторов (53 084 160 значений F32),324masked cases,
8negative geometry guards. Все режимы совпали с независимым stock graph,
без допусков. Threads32, elapsed2.220 с; GPU не использовался.
ELF SHA256 `0c7173d9b74753bd75f7eb57a83e47ea6022b07f8d1fa30ce139f51bdbf6f42c`. Snapshot header/proof исходников
совпадает с текущим source на 15:34UTC27.10.2026.

Компонент не подтверждает model gates, качество
GSM/tools, GPU prefill или прирост t/s. Они проверяются отдельными заданиями.
IQ2 GPU exact cached helpers не добавлялись; их allowlist расширять этим флагом
нельзя. DSpark batch-dense Q5/Q4 — отдельная работа с собственным dispatch proof.

## Независимое ревью соседнего dense proof и ограничения скорости

`/mnt/ai/logs/m31-port/q2-cpu-cross-review.json` пересчитывает **сырые**
`q2-dense-mmv-component-v1.json`, не доверяя rc0 или готовой summary.
Проверены 26signatures, для каждой полный набор marked K1/2/3/4 ×2 семейства
активаций: всего 208marked cases/6 609 920F32, bit differences/max_abs/rms=0.
K8 не помечен; MSA4096 значений после recapture не изменились. Общий probe
может завершиться rc0 с ненулевыми exploratory diff, поэтому критерий приёмки
применяется именно к marked K1..4. Проверка marker API сохраняет прежние
ограничения leaf NONE/immutable WEIGHTS/contiguous2D/hintNONE/fusion fence.

По source pinned stock CPU kernel IQ2_XS использует AVX2-ветку с 32 индексными
8-байтовыми чтениями grid на 256 элементов (LUT4KiB); знаки восстанавливаются
битовыми операциями и shuffle. IQ3_XXS использует 64 индексных 4-байтовых grid
и 32 индексных 8-байтовых sign-чтения на 256 элементов. В обоих случаях итог —
FMA по 8F32-lanes и исходное horizontal reduction. Это счёт операций по source,
**не** измерение тактов или прогноз t/s. AVX512-специализированной ветки для этих
двух функций в текущем x86 vendor нет, хотя общий CPU backend собран с ISA EPYC.
Уменьшение байтов экспертов может частично съедаться разной стоимостью unpack,
синхронизаций и другими весами; итог решает полный model benchmark.

Будущий exact IQ2 pair helper мог бы разделить grid/sign/scale unpack между
двумя RHS, сохранив отдельные I32-lanes/FMA/reduction. В этой волне такой helper
не реализован и ему не приписывается выигрыш. Текущий IQ2 pair fallback точен,
но выполняет два stock dot. Default IQ3 arithmetic headers и native-gpu profile
на момент ревью побайтно совпадают с HEAD; final model regression — отдельный gate.

Повтор на конечных canonical q2-v1 CPU ggml libraries: **PASS** с теми же
1008 сравнениями/53 084 160F32. Receipt
`/mnt/ai/logs/m31-port/q2-cpu-final-tie-v1/receipt.json`,
ELF SHA256 `9b2aefde40ad58c515a127e86ded54c08da24736a3388ed94b16d25aa84a3b5f`,
elapsed2.169 с. Fixture/source snapshot прежний, ссылки ведут на
`build-port7-canonical/q2-v1/build-cpu/ggml/src/libggml{,-cpu,-base}.a`.
Оба stock CPU quant objects (`arch/x86/quants.c.o`, `quants.c.o`) дополнительно
побайтно совпали со старой проверенной сборкой. Финальная model execution/quality
по-прежнему отдельная приёмка. Никакие tools/runtime sources во время canonical
freeze не менялись.

Прямой tie с CUDA model build: `q2-v1/build-cpu` и `build-cuda` содержат
побайтно одинаковые `arch/x86/quants.c.o` (68 512bytes, SHA256 `b77859ca235e92fc731f1b99ec4bd5fc3cb6b2c1456acf0fd32a08404a3f3276`)
и `ggml-cpu/quants.c.o` (71 696bytes, SHA256 `f2681036214b802f5752887e34590822fecdcbc7245d32e5f56ca776ddf72ac2`).
Source и compile flags этих TU одинаковы (`gcc-14 -O3 -DNDEBUG -march=native`).
Тем самым компонент на canonical CPU libraries использует те же dot/activation
quantizer машинные функции, что CUDA model ELF. Это не утверждение равенства
всех статических архивов или model logits.

## Конечная модель: фактический n1 CPU-путь

`/mnt/ai/logs/m31-port/q2-cpu-final-model-execution-v1.json` проверяет конечные
native diagnostic logs и command/result receipts. Во всех 4 запусках реально
напечатано `M31 fused CPU MoE: enabled=1`; env явно задаёт CPU_FUSED1,
CPU_STOCK_EXPERTS1, CPU_DIRECT0, GPU_CACHED_DECODE0, tapsnone и GPU minimum32.
Наблюдаются 96n1forward:32 на code128 (позиции 96–127),32 на 16K (16352–16383),
32 на 32K (32736–32767). Code3000 выполняет один большой forward, поэтому его
CPU-fused init marker сам по себе не считается n1 execution proof.

По детерминированной ветке `routed_cpu_into()` эти 96forward соответствуют
96×57=5472CPU fused layer calls. Это вывод из source + наблюдаемых условий
ветки, **не** отдельно инструментированный kernel counter. Сверка с oracle
и качество — самостоятельные receipts, не выводятся из enabled marker.

Профиль om обрабатывается read-only скриптом
`/mnt/ai/logs/m31-port/q2_profile_counters.py`; он разделяет prompt/decode по
реальному prompt_n, а также singleton prompt tails, first sample и head.
Ordinary TG256samples включает 255n1forward; warmup32 —31. Полные первые
холодные forward и смены bucket сохраняются. Detailed `dense-stages` —
подмножества dense_ms; async compute_ms означает enqueue, read_ms включает
ожидание GPU. По этим host counters нельзя объявлять аппаратную HBM bandwidth.
CPU logical bytes/s также включает quant/OAI/barriers и не измеряет DRAM traffic.

## Фактический профиль генерации Q2 против свежей V8

Источник: `/mnt/ai/logs/m31-port/q2-cpu-final-speed-profile-v1.json`;
read-only parser `/mnt/ai/logs/m31-port/q2_profile_counters.py`.
Все 12request segments прошли проверки: каждый warmup имеет 31decode forward,
каждый 4K/16K/32K —255, позиции непрерывны и согласованы с native TG numerator.
Ни singleton prompt tails, ни первый sample из prefill не включены в decode.
Q2 ниже — pooled510n1forward на глубину (2 прохода), IQ3_S V8 —255 (1 проход).
Полные холодные первые forward, перестроения bucket и выбросы сохранены.

| Контекст | CPU routed V8, мс | CPU routed Q2, мс | dense host V8, мс | dense host Q2, мс | head V8, мс | head Q2, мс |
|---|---:|---:|---:|---:|---:|---:|
|4K|23,024|21,432|12,985|12,502|0,730|0,540|
|16K|24,321|21,977|13,973|13,697|0,733|0,539|
|32K|25,487|19,335|15,225|14,808|0,736|0,541|

Реальное сокращение всего TG interval на n1forward: **2,457/3,134/7,023 мс**,
а не постоянные обещанные−4,7 мс. CPU logical weights/token:
5 117 313 024→4 183 031 808bytes. Деление на полный routed host time даёт
V8 **222,3/210,4/200,8GB/s**, Q2 **195,2/190,3/216,3GB/s**. Это логическая
эффективная полоса, включающая activation quant/OAI/pool barriers; аппаратные
DRAM bytes не измерялись. Q2 заметно варьируется между проходами:
CPU4K22,805→20,059 мс и 16K24,042→19,912 мс,32K19,247→19,423 мс.
Из этих данных нельзя выделить одну причину или обещать константный выигрыш.

Сравнение сделано в одной волне на одной карте и тех же ctx40K/cache7GiB/T80,
но с разными квантами/output routes/UUID, старым frozen V8 adapter и новым
canonical adapter. Это не matched-ID pair и не изолированное сравнение ядер.
Профильное pooled TG — total forwards/total elapsed, поэтому оно отличается
от медианы двух om t/s; официальные скоростные числа брать из bench receipts.
По host dense counters нельзя подтвердить текущую HBM bandwidth: read_ms
содержит ожидание GPU, shared expert работает параллельно CPU. Head cost здесь
включает setup+compute+read, не только время GEMV.

`M31 GPU expert profile` и destructor cache/upload counters — cumulative со
старта GpuExperts: `Engine::reset()` не вызывает `reset_stats()`. Повторяемые
снимки после CPU n1 нельзя суммировать. Счётчик GPU forwards означает FFN слоя,
не весь 60-слойный Engine forward. Sampling_ms включает первый sample вне TG,
interactive_wait_ms — в том числе последний ACK после decode_end; их полные
агрегаты также нельзя механически вычесть как точную TG-only часть.
