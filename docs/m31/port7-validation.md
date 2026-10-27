# Port7: измерения и приёмка

Принятая v8 сохранена: canonical ELF в `build-port6-canonical/v8`, runtime
profile `tools/m31/profiles/native-gpu.json`. Новые кандидаты получают отдельный
ELF, receipt исходников, каталог с замороженными помощниками и runtime profile.
Изменение профиля или исходников после замера требует нового кандидата.

## План

1. Same-day v8, одно exclusive окно: неизменённый `om_bench.py`,
   `warmup,4K,16K,32K`, затем два одинаковых коротких greedy запроса128 токенов.
   Это диагностический круг, не повторная приёмка v8.
2. Для каждого кандидата — короткий численный gate против GPU-offload oracle,
   потом сопоставимый короткий decode. Ошибку качества или результат хуже
   исходного не маскировать повторным выбором удачного прогона.
3. Отдельный CUPTI/nsys decode trace при необходимости: profiler timings не
   попадают в таблицу om. `dense compute_ms` — host enqueue; `read_ms` также
   включает ожидание исполнения GPU и синхронизацию.
4. Финальному кандидату — full gate20, выборочные длинные16K/32K, unchanged
   om два круга4K/16K/32K и исходный GSM8K100+tools23 на том же ELF/profile.
   Для DSpark дополнительно строгое совпадение raw greedy токенов с обычным
   движком, включая отказ/частичную приёмку и откат KV/индексера.

Численный контракт: top1≥99%, meanKL≤1e-3, maxKL≤.05 против GPU-offload
llama.cppPR31192. Итоговое качество GSM≥95 и tools≥19. Цель TG≥35 на каждой
глубине, stretch≥40; префилл сравнивается с принятой v8, а не только с floor1000.
При малом отличии PP нужны сырые данные и оценка разброса, а не округлённый PASS.

## Инструменты

`port7_bench_prepare.py` готовит новый job и ничего не запускает. Он использует
принятый v8 adapter, текущий immutable ELF/receipt, явный native-gpu profile,
сбрасывает унаследованные debug/seed переменные и при необходимости добавляет
замороженный `port7_decode_client.py`. Клиент сохраняет полные HTTP responses и
hash message JSON. Это не raw token IDs и не численная приёмка DSpark.

`port7_bench_report.py` сохраняет все stage intervals, raw speed summaries,
сравнение каждой глубины с v8 и отсутствие общей приёмки при недостающих гейтах.
Один круг всегда остаётся диагностикой. Сложение пересекающихся stage intervals
не объявляется полным разложением wall time.

Карточные задания только через `card_launch.py`/aictl/общий card lock; speed job
держит exclusive `.build.lock`. Root объявляет окно; агенты не запускают тяжёлые
сборки/CPU probes до его окончания. Никаких изменений защищённых пресетов.

## Текущее окно

`codex-m31-port7-v8-baseline-v1`, каталог
`/mnt/ai/logs/m31-port/port7-v8-baseline-v1`.
Проверено совпадение48 ENV, ELF39a86…, adaptere87…, CPU80/context40000/
chunk32768/cache7/seed0 с принятой v8. Дополнительные явные unset взяты из
переносимого профиля и исключают утечку debug/seed controls.

Baseline завершён: session/оба clients/server rc0, unit inactive/dead,
aictl active=null, конфигурация побайтно равна бэкапу до сессии.

| Глубина | PP ток/с | TG ток/с | CPU routed мс/forward | Dense read мс/forward |
|---|---:|---:|---:|---:|
| 4K | 1378,7 | 27,90 | 20,726 | 11,661 |
| 16K | 1528,2 | 27,01 | 21,224 | 12,472 |
| 32K | 1246,1 | 25,98 | 21,078 | 13,875 |

В каждом om запросе255 отдельных decode forward: первый предсказанный токен
получается из prefill logits. Tiny128×2 дал28,69/28,44 ток/с и одинаковый
message SHA `87984ce552fe9b84d58c72552f5abff87a4add7bbcfbade08f64971730b1b983`.
Peak GPU30483MiB. Полные receipts и разложение host intervals:
`/mnt/ai/logs/m31-port/port7-v8-baseline-v1/report.json`.

Цифры воспроизводят исходное узкое место. Отдельный CUPTI trace должен
определить долю реального исполнения kernels внутри `read_ms`;0,7–0,8мс
host `compute_ms` не доказывают столь же короткую арифметику GPU.

## Actual CUDA trace v8

Первый профильный кейс в `port7-dense-trace-v1` завершился успешно. По124
steady токенам после пропуска первых4: kernel union11,037мс/токен,
DMA union0,330мс/токен,3754kernel activities/токен.57hidden D2H занимают
суммарно0,039мс DMA/токен; связанный host `cudaMemcpyAsync` ждёт граф
около9,09мс. Большая часть `read_ms` приходится на реальную GPU работу.

Вклад kernel activities: quantized MMV6,225мс/241запуск, broadcast1,411мс/
952запуска, float MMV0,976мс/117, RMS0,534мс/361, argsort0,499мс/117,
RoPE0,410мс/240. Это profiler diagnostic: его42,7мс wall/токен не является
скоростью om из-за profiler overhead.

Независимая проверка десяти выбранных steady окон через SQL overlap и
endpoint sweep совпала до наносекунды с kernel/DMA/device unions:
`port7-dense-trace-v1/v8-independent-interval-audit.json`.

CandidateOFF в той же сессии отказал до первого forward: planner увидел
недостаток свободных expert slots. Root установил причину: новый GPUStats
header был смешан со старым архивом. Это ошибка сборки эксперимента, не
численное расхождение. PinnedON и последующие сравнения не исполнялись;
нужна новая сборка с согласованными header/archive и новый job.

## Строгое сравнение greedy DSpark

`port7_greedy_compare.py` сравнивает raw int32 ID только после обязательного
preflight. Он проверяет реальные prompt IDs, фактические chunks и padded nk
из `M31 prefill-plan`/`M31 forward`, context/maxchunk/target CLI, explicit ENV
и unset, SHA ELF и build receipt. F16 main/index KV подтверждаются явными
типами в исходнике Engine, SHA которого совпадает с frozen build receipt.
Generated IDs должны лежать именно в output каталоге выбранного command/case.
Для pilot с несколькими командами/повторными prefill есть явные selectors.

Коды:0 = strict greedy equality,2 = валидное численное расхождение,
3 = `invalid_reference`, scoring не выполнялся. Это позволяет отделить ошибку
геометрии эксперимента от ошибки speculative rollback.

Первое сравнение actualspecV2 с serialref было неверным: reference использовал
ctx4096/chunk2048 (2048+1276), spec — ctx40000/chunk32768 (один3324).
117diff не являются доказательством ошибки DSpark. Старые artifacts сохранены.
Generic comparator отверг эту пару до scoring с rc3. Новый samegeometryserial
дал128/128 одинаковых raw IDs и прошёл весь mandatory preflight:
`port7-dspark-matched-serial-numeric-v1/strict-greedy-compare-v1.json`.
Аналогичный audit persistentV3 pilot также прошёл128/128:
`port7-dspark-persistent-v3-pilot-v1/strict-greedy-compare-v1.json`.

Эти проверки доказывают выбранные128 greedy outputs. Полные gate20,
длинные выборки, GSM/tools и om финального кандидата остаются отдельной
приёмкой; их не подменяют component tests или timing численных jobs.

`port7_memory_watch.py` использует NVML physical GPU0, interval100ms и работает
только внутри разрешённого aictl worker. Первый реальный draft run дал882samples,
peak26520,44MiB/minfree6086,56MiB. NVML включает память driver/других процессов;
sampling может пропустить короткий peak. Child exit code сохраняется.

## Пилоты V6/V7 и фактический verify-tail

V7 causal mini использует один frozen ELF и собственный serial reference для
каждого профиля shared overlap. Parity 128 строк у обоих профилей побитная;
все шесть continuations по 128 raw ID совпали с соответствующим serial.
Overlap 0 → 1: depth 1 — 23,317 → 23,739 ток/с; depth 3 — 19,680 → 20,051;
depth 7 — 14,578 → 14,651. Это диагностические timings, улучшение недостаточно
для продвижения относительно обычного decode. Попытка mini-v1 остановилась
до модели из-за отсутствующего parent каталога; numerical score у неё нет.

`GGML_CUDA_GRAPH_CHUNK_SIZE=0` сохранил 128 строк с maxabs=0 и top1 diff=0.
Первоначальный postcheck ошибочно требовал точного нуля у вычисленного KL
(~1e-18 при одинаковых logits); original failure сохранён, исправленный
read-only анализ лежит в `port7-v6-chunk0-parity-v1/corrected-analysis.json`.
После warmup dense 71,37 → 72,09 мс/K8; выигрыша не видно, окна разные.

Первый реальный `port7_verify_dump` для `code_18_3000` завершил native и
GPU-offload oracle с rc0. Проверены 129 строк: anchor P−1 отдельно и tail
P..P+127. Top1=100%, mean/max KL=0, maxabs=0. Все captured features конечны
и побитно совпадают с ordinary serial; реально выполняются verify K1/2/3/4/7/8,
полные/частичные commit и rewind. Receipt:
`/mnt/ai/logs/m31-port/port7-v6-fasttail-code3000-v1/compare.json`.
Это single-case diagnostic (`acceptance_eligible=false`), не полный gate20.
Oracle использует combined IDs из native Dump и подтверждённый runtime PP
план 2048+952; пустой `actual_chunks` в verify Dump не служит PP-рецептом.

Первый PP32 capture A/B дал ~−1,4%, но om UUID создал разные prompts.
Он остаётся diagnostic. Новый PPv3 фиксирует UUID и сначала требует равенства
rendered IDs, фактических chunks/nk, sampler/greedy output, затем считает ratio.

PPv3 с фиксированными UUID завершился rc0. Равны actual rendered IDs, параметры,
PP chunks/nk и по 256 greedy outputs в двух измерениях (warmup 32 также равен).
Weighted PP reuse 1229,09 против ordinary 1246,20 ток/с: −1,373%, оба выше 1000.
`paired-comparison.json` фиксирует correctness отдельно от отношения скоростей;
двух последовательных повторений недостаточно для статистической уверенности.

Строгие source/ENV/atomic-result/actual-geometry/raw-ID audits V5 и V6 sweep
завершились PASS по 20×128 ID каждый. V7 mini также прошёл два отдельных audit
по три arms, с selectors ref0/spec1 и ref2/spec3. Это проверка жадных outputs
выбранного кейса; она не меняет отрицательный результат по скорости.

Для следующего P7-v8 подготовлен отдельный long diagnostic режим:
`--diagnostic-input <frozen16K-or32K.i32> --tail 32 --context 40000 --chunk 32768`.
Oracle получает фактический native PP план через `M31_ORACLE_CHUNK_PLAN`, затем
32 singleton input. Этот режим никогда не помечается final gate20 acceptance.
У verify dump capture window после reset равен 0; max_rows4096 сохраняет anchor
и tail, но server/session оптимизацию late-prefix window он не измеряет.

P7-v8 candidate (отдельный от принятого port6 v8): causal flag matrix
base → VIEW → VIEW+PACKED завершилась rc0. У всех трёх parity 128 строк
maxabs=0; каждый actual depth1 continuation из 128 raw ID прошёл свой строгий
source/ENV/actual-geometry/atomic-command audit. Diagnostic TG:
24,2268 → 23,9257 → 25,2102 ток/с. Компиляция могла выполняться параллельно,
поэтому проценты улучшения не являются clean speed acceptance.

`port7_verify_chain.py` позволяет продолжить подготовленные long diagnostics
при прерывании Codex: ждёт causal atomic rc0, исполняет три строгих raw audits,
затем последовательно native/oracle/compare для каждого prepared long каталога.
Каждый GPU job проходит через card_launch/common lock; CPU compare — S-lock.
Любой отрицательный prerequisite останавливает цепочку; completed jobs повторно
не запускаются. Для P7-v8 frozen controller и source hashes находятся в
`/mnt/ai/logs/m31-port/port7-p7v8-long-chain-v1/`.

P7-v8 selected long verify завершён на 16K и 32K: оба native rc0, все
computed/committed logits и captured features побитно совпали с ordinary n1.
Оба GPU-offload oracle сравнения дали top1=100%, mean/max KL=0, maxabs=0
на anchor+32 tail строках; 14 verify calls и 6 partial commits на кейс.

Первый oracle ctx40000 не смог reserve 17,34 GiB и завершился до inference:
это capacity failure, math score отсутствует. С явного повторного разрешения
координатора применена прежняя port6 selected-long capacity-equivalence policy.
Native requested ctx40000 сохранён; oracle requested16416/32800, actual padded
16640/33024. Отдельные immutable retry jobs/outputs сохраняют исходные headers.
Перед scoring строго проверены runtime contiguous KV min/max, PAD256 active keys,
trained linear RoPE ext0/orig1048576/scale1 и точный PP/singleton replay.
Старый OOM job и остановившийся controller не переписаны.

Результаты: `port7-p7v8-verify-long16384-v1/compare.json` и
`port7-p7v8-verify-long32768-v1/compare.json` в `/mnt/ai/logs/m31-port/`.
Это selected long diagnostics, не full long-context acceptance или GSM/tools.

## Завершение исследования 27 октября: обычный декод

После возобновления в 13:54 UTC (aictl печатает МСК) root ограничил исследование
до 15:50 UTC и поставил практическую цель +10% TG к принятому v8. M3 sweeps
больше не расширяются. Representative HTTP шесть запросов завершились rc0 и
strict PASS по actual IDs/PP: depth1 TG24,54/25,76/23,16; depth3
22,08/21,72/20,97 ток/с на 4K/16K/32K. PP32K1245,9/1241,5. Clean native RPC
same3324/128 также отрицателен: COALESCE0 ordinary25,3477→M3 23,0348;
COALESCE1 ordinary24,6670→M3 22,8065. Все четыре raw128 ID совпали; lazy init
и seed append входят в native predicted_ms. Это не повод продвигать M3.

Оба новых clean компонента обычных кешированных экспертов прошли:

- IQ3 IMMA: 1059 raw и 684 actual случая, 1 079 336 I32 и 414 870 F32 exact.
  Cold quant+dot gate60,864→20,160 мкс, up59,776→20,576; warm43,63→15,16.
- Объединение gate/up: 60 случаев и 1 474 560 F32 exact; contiguous n1
  group0 ускоряет helper в 1,275–1,401 раза; gap fallback около1,00.

Это изолированные kernel/helper результаты, не прирост всей модели. Следующий
интеграционный кандидат должен повторить component proof и ordinary clean mini.

`tools/m31/port7_ordinary_aba.py` готовит отдельный OFF→ON→OFF job, сохраняя
исходный audited two-arm recipe. Пример после готового immutable ELF:

```
python3 tools/m31/port7_ordinary_aba.py freeze \
  --directory /mnt/ai/logs/m31-port/NEW-ordinary-aba \
  --candidate /mnt/ai/Xolotl-m31/build-port7-candidate/NEW \
  --on-env M31_GPU_CACHED_IQ3_IMMA=1 \
  --on-env M31_GPU_CACHED_GATE_UP=0 \
  --on-env M31_GPU_CACHED_RANK_OUTER=1
```

Job measurement=true, 3324 prompt +128 greedy, same ELF/context/chunks, без
sidecar. Primer сохраняется внутри decode_ms. Scorer требует одинаковые raw ID,
PP chunks, zero sidecar bytes, source/profile/job fingerprints и подтверждённый
seed290/actual route count. Скорость оценивается отдельно от math PASS:
conservative ratio = ON/max(OFF1,OFF2). Один такой результат не является
устойчивым приростом или om/quality acceptance. После END отдельно проверяется
атомарный внешний card result. Проверка scorer на сохранённом actual negative
A/B не дала ложного продвижения; подмена job hash отвергнута. AST/py_compile PASS.

Strict-spread80 CPU skew завершился PASS4,250с против исходных4,328с, с 80
physical cores без SMT duplicates; это private component, не model TG.
MWAITX38 arms PASS, но wake latency хуже PAUSE: sameCCD около305 против85нс,
otherCCD341–346 против240нс. Power не измерялся, runtime не меняется.

Каноническая схема после freeze: CPU→CUDA jobs3, конечные CPU fixtures, затем
same-input default flagsOFF против frozen port6 v8. Только положительному
ordinary winner нужны расширенные gates/om/GSM/tools. V9 paired PP и bounded
HTTP lease пока prepared-only; их CPU fixtures не означают actual model proof.

## Каноническая сборка и ограниченный THP trial

Canonical v1 собрал семь CPU ELF, но fixture не импортировал gguf_writer из
снимка: пропущены top-level tools/gguf_writer.py/gguf_reader.py. Runtime math
ещё не выполнялась. Сохранены failure receipt/logs, автоматический GPU chain
остановился до запуска. Root исправил только сбор manifest поддержки.

Новый canonical v2 завершился 14:28 UTC: CPU и CUDA, семь ELF каждого типа,
все конечные fixtures PASS; source/vendor stable, protected v8 unchanged.
Receipt: `/mnt/ai/Xolotl-m31/build-port7-canonical/v2/build-receipt.json`.
Исходники заморожены; последующий git commit меняет metadata, не source hashes.
Default code128 на конечном CUDA ELF уже прошёл GPU-offload oracle:128 rows,
top1=100%, mean/max KL=0, maxabs=0. Один unchanged om круг4K/16K/32K ожидается
в очереди; это диагностический круг, не новая полная приёмка.

Полный clean ordinary cache ABA на P7-v11: OFF1 26,1817, ON(IMMA/rank)26,1542,
OFF2 25,0592 ток/с. Все128 raw ID и actual PP recipe равны; IMMA10020 и
rank_outer2806 фактически выполнялись, primer108,198мс внутри serial timer.
Conservative ON/max(OFF1,OFF2)=0,99895; +10% нет. Drift OFF−4,29% не позволяет
объявлять небольшой положительный эффект относительно одного медленного OFF.
Независимый source/profile/args/raw/counter audit PASS. Дополнительный combined
ON не запускался, поскольку условие первоначального +5–9% не выполнено.

Own anonymous THP16GiB probe:100% AnonHugePages за0,116754с,8threads,
free high-order bytes до trial69,06GB. Никаких глобальных настроек, регистрации
CUDA или model throughput в этом micro измерении не было.

Root разрешил один bounded full-model THP trial на финальном canonical ELF:
OFF→THP1/FORCE1/REGISTER1→OFF, те же3324 IDs/128greedy, остальные новые пути0.
Wrapper ограничивает child180с, внутренний общий budget575с/внешний590с,
штатно останавливает только свой child при stall. OFF1 дал25,1945 ток/с,
после регистрации AnonHugePages=0. Early during-fill ON snapshot PID2513131
показал2,575GB huge из2,698GB anonymous RSS; это concurrent partial phase.
ON затем не завершил startup за180с, остановлен, post-registration marker и
snapshot отсутствуют. OFF2 не запускался. Это startup feasibility timeout,
не math mismatch и не отрицательный результат TG. Нельзя распространять ранние
95,4% или micro100% на всю153GiB CUDA-registered arena. Полный FORCE bypass
guard здесь не принят; глобальную compaction/hugetlb policy не меняли.

Все THP trial artifacts: `/mnt/ai/logs/m31-port/port7-canonical-v2-thp-aba-v1/`.
Default canonical diagnostics продолжаются отдельным chain с baseline flagsOFF.

## Свежий контроль замедления машины

Первый canonical om круг дал PP1382,1/1540,9/1256,2 и TG25,81/25,82/24,69.
TG32K ниже25; сравнение только с прежним same-day baseline не позволяло ни
объявить регрессию source, ни списать её на noise. По указанию root запущена
новая clean fixed-UUID пара: immutable port6 v8 → canonical v2, одинаковые
helpers/profile75/tokenizer/corpus, warmup+4K/16K/32K.

Оба outer/session rc0; strict scorer272PASS: actual prompt IDs, параметры,
полные PP recipes(pos/requested/n/nk), native32+3×256 IDs, ACK/sampling и parsed
choices равны. V8→canonical:

- PP1381,53→1379,30 /1542,21→1532,31 /1255,80→1251,58: −0,16/−0,64/−0,34%.
- TG25,5578→26,1536 /25,2706→24,3964 /23,8875→24,8665: +2,33/−3,46/+4,10%.

Неизменённый v8 также замедлился относительно прежнего замера. Эта одна пара
подтверждает равенство выбранных outputs/geometry и не показывает устойчивой
регрессии canonical относительно текущего control; она не доказывает +10%,
полную приёмку или постоянный нижний порог25. Reference official v8 сохранён.
Receipt: `port7-canonical-v2-fixed-uuid-pair-v1/paired-comparison.json`.

Root разрешил последний bounded placement trial на existing flags: final
canonical serial3324/128 OFF→strict-spread80→OFF, остальные новые режимы0.
Создан новый logs-only harness `port7-canonical-v2-cpu-placement-aba-v1` с теми
же180с/arm и590с total caps, latest-start15:40UTC. M31_CPU_AFFINITY сам включает
strict pool; ранние task masks регистрируются без внешней смены affinity.
Pool создаётся до arena.load: наследование узкой main mask загрузчиками —
реальная часть startup этого профиля, не скрыто исправляется. Пока PENDING.

## Закрытие волны

Model strict placement ABA END rc0: OFF1 26,0889 → ON19,8664 → OFF2 23,5870
ток/с, conservative0,76149. Все128 raw ID и PP recipe одинаковы. ON реально
включил strict80; startup112,5с против OFF83,4/81,6с. Early snapshot показал
mainCPU93 и ещё шесть потоков с той же узкой маской. Это наблюдаемая оговорка
инициализации; affinity извне не исправляли. Прирост private CPU component
не перенёсся на полный движок. По указанию root новые GPU trials прекращены.

Host stage-summary matched пары позволяет отделить общий drift: dense read
на32K V8/canonical13,873/13,871мс, routed24,289/23,150мс; на16K routed менялся
в обратную сторону23,680→25,207мс. CPU compile audit нашёл одинаковые flags
и побитно одинаковые hot quants objects/pool functions. Здесь нет доказательства
постоянной source-регрессии; нет и устойчивого +10%.

Read-only resource/config audit15:10UTC PASS: своих model PID/experiment units
и bench listeners8180–8182 нет; временный preset m31-xolotl-check отсутствует.
m31/hxq3k/hxq3k-exact/hxq3k-lite и остальные top-level aictl config keys равны
backup начала port7. Minerhold сохранён, GPU1 miner и CPU duty active. Текущая
модель m31-vision-test принадлежит другой задаче, её не останавливали.
`port7-final-resource-config-audit-v1/audit.json` в `/mnt/ai/logs/m31-port/`.

Итог: accepted port6 v8 сохранён, canonical defaults и выбранные opt-in пути
имеют точные проверки, но цель35–40+ и более поздняя практическая +10% TG
не достигнуты. Полные новые gate20/GSM8K100/tools23 не запускались при отсутствии
winner. Одиночный default om, fresh matched pair и selected long diagnostics
не переименовываются в полную приёмку новой сборки.
