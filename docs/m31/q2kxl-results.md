# MiniMax-M3.1-Flash UD-Q2_K_XL — итог испытания

**Q2_K_XL в Xolotl не принят.** Базовый режим достиг скоростного минимума
на 32K и получил GSM8K97/100, но tools17/23 ниже порога19. Дополнительный
кешированный режим совпал с GPU-размещением e3 на code128, однако не прошёл
длинные численные гейты против обычного e3 oracle с батчем2048. Его новые
om_bench и GSM/tools по объявленным условиям не запускались.

Официальная IQ3_S v8 сохранена. Новая поддержка типов и исследовательские
режимы остались opt-in; постоянных пресетов, deployment или push нет.

## Скорость против качества

Одна RTX5090, карта0, EPYC9654. PP/TG — токены/с на4K/16K/32K.

| Конфигурация | PP4K /16K /32K | TG4K /16K /32K | GSM8K100 | tools23 |
|---|---:|---:|---:|---:|
| llama.cpp IQ3_S, прежний референс |425,35 /474,4 /433,0|23,33 /22,63 /21,92|96|19|
| Xolotl IQ3_S v8, прежняя приёмка |1390,85 /1535,00 /1246,65|27,445 /27,015 /25,305|97|19|
| Xolotl IQ3_S v8, свежий контроль |1382,6 /1539,0 /1254,9|25,97 /24,46 /23,12|не повторяли|не повторяли|
| llama.cpp Q2_K_XL e3 — выбран |484,0 /532,8 /479,1|26,11 /25,21 /24,37|**97**|**19**|
| llama.cpp Q2_K_XL e4 |489,4 /539,0 /483,8|25,22 /25,02 /23,46|не запускали|не запускали|
| Xolotl Q2_K_XL baseline, медианы2 |1678,8 /1550,25 /1302,1|27,785 /26,635 /27,605|**97**|**17 — FAIL**|
| Xolotl Q2 cached-e3, cap6656 |не запускали|не запускали|не запускали|не запускали|

e3 хранит экспертов слоёв10/11/12 на GPU; e4 добавляет13. Выбран e3:
генерация выше на всех трёх глубинах. e4 помещается, но после compute остаётся
около1652MiB, меньше ещё одного слоя размером2214MiB. Это выбор среди
проверенных вариантов, не доказательство глобального оптимума всех настроек.

Исходные серии baseline Xolotl Q2:

| Серия | PP4K /16K /32K | TG4K /16K /32K |
|---|---:|---:|
|v1|1670,0 /1545,2 /1299,0|26,67 /24,71 /27,85|
|v2|1687,6 /1555,3 /1305,2|28,90 /28,56 /27,36|

На32K оба прогона превышают PP1000 и TG25. На16K v1 ниже25; медиана двух
не доказывает стабильность. Свежая v8 тоже ниже исторических результатов,
хотя её ELF прежний. Причина текущей вариативности не установлена.
Два Q2 прогона и один контроль v8 с разными UUID/входами не являются строгим
экспериментом с одинаковыми prompt IDs или устойчивым приростом≥10% везде.

Использован исходный `~/om_bench.py`: warmup,4K/16K/32K, greedy,
256 выходных токенов, ignore_eos. Скоростные окна — exclusive build-lock,
без наших сборок и тяжёлых CPU-проверок. Корпус один; native tokenizer
насчитал259481 токен, llama.cpp259480. Исторические quality97/19 v8
не переносились ни на свежий контроль, ни на Q2.

## Полное качество двух базовых конфигураций

Исходный `~/eval_duty_1023.py`, первые100 GSM8K,20 инструментальных запросов
и3 продолжения. Temperature1,top_p0,95,max_tokens6000; фактические defaults
обоих серверов top_k40/min_p0,05, размышления по умолчанию. Выполнен один
стохастический прогон каждого режима, без перезапусков для подбора оценки.

| Режим | GSM ошибки, индексы с0 | tools ошибки | Итог |
|---|---|---|---|
|llama.cpp Q2 e3|12,73,85|tool5,9,11,18; follow3/3 верны|97/100,19/23 PASS|
|Xolotl Q2 baseline|12,73,89|tool9,10,11,13,18; follow1|97/100,17/23 FAIL|

У обоих ровно123 уникальные упорядоченные строки, HTTP/card/client/server rc0,
файлы ELF/client/corpus/profile прошли post-SHA аудит. Length/leak/err по
записанным полям —0. Независимо перепроверены gold/pred и сохранённые tool
аргументы. Harness сохраняет не все полные ответы, поэтому независимое
повторное извлечение pred/leak из полного HTTP-потока не заявляется.

В native9/10 выбран shell/find вместо search_files,11 — поиск контактов через
shell вместо send_message,13 — умножение через shell,18 — status вместо restart.
Follow1 дал верные `103,38 EUR`, но исходный тест требует `103.38`. Скорер
оставлен без изменений; даже возврат этого одного пункта не достиг бы19.
Зарегистрированных JSON decode/check исключений нет. Это не доказательство
одной ошибки parser, quant kernel или KV-пути. RNG/frontend и часть численных
настроек серверов различаются; один прогон не оценивает статистическую
значимость разницы. По заданному строгому порогу baseline всё равно не принят.

## Численные контракты и результат кеша e3

Первый oracle повторяет прежнюю методику порта: PR31192, все экспертные
веса хранятся на CPU, GPU op-offload включён, F16 KV, fusion выключен,
разбиение воспроизводит actual native. Это GPU-префилл, не strict CPU oracle.
Stock Q2 и конечный baseline fused-путь прошли выбранные4 случая; финальные
224 строки побитно равны, mean/maxKL0. Независимые raw F32 SHA это подтвердили.

| Случай baseline | Фактическая геометрия | Строки |
|---|---|---:|
|code128|PP96 +32 teacher-forced n1|128|
|около3000|PP3000, последние32 строки головы|32|
|16K|PP11904+4448, затем32 teacher-forced n1|32|
|32K|PP11904+9088+7424+4320, затем32 teacher-forced n1|32|

Итого96 реальных n1 forward. Это выбранные позиции, не full gate20 и не
сравнение свободных256-token om-ответов с oracle. Для больших baseline
чанков oracle40K получил OOM; physical16/32K повторы прошли проверки
ID/позиции, active n/nk, KV-типы и train-RoPE. Native остался ctx40000,
physical capacity40192. Native PP NK наблюдается в runtime; singleton NK
проверяется по последовательным позициям и привязанному к исходнику
kv_length/PAD256 invariant, не по отдельному runtime n1 NK счётчику.

Второй oracle соответствует GPU-размещению e3: три полных экспертных слоя
10/11/12 на CUDA0, остальные CPU. Короткий F16/unfused/ctx40000 тест baseline:
PP96 побитно равен, n1 tail31/32 top1; общий top1=127/128=99,21875%,
meanKL0,0105963 и maxKL0,360217 — **KL FAIL**. Точность первого контракта
не выдаём за совпадение с e3. HTTP e3 дополнительно использует Q8 KV и
штатную fusion, поэтому его качество измерено самостоятельно.

Для проверки конкретного различия использован существующий native cached
путь без новых C++-ядер. Cache9,88GiB даёт ровно384slots; decode-only seed
покрывает все128 экспертов каждого из трёх слоёв e3. Rank — синтетическая
placement-перестановка7296, не частотная калибровка: count/training0,
calibrationnone, legacy weight_sum явно обозначен как priority1/0.
Новая Q2 identity связана с четырьмя user-verified SHA и geometry seal.

Одна bounded short попытка прошла: aggregate128 и tail32 отдельно top1=100%,
KL0/maxabs0, весь raw F32 побитно равен e3 gold. Pool384, один seed384,
7296 lookups/384hits/96hybrid forwards, последующих загрузок весов нет.
Полный pool плюс источник no-admission доказывают покрытие только слоёв e3;
отдельный наблюдаемый per-layer счётчик не выдумывается.

Продолжение зафиксировано как отдельный профиль: cache9,88/384slots,
ctx40000/F16/arena140/T80, maxchunk6656 с adaptive planner. Native выполнил4
входа без ошибки, для16K PP6656/6656/3040, для32K
6656/6656/6144/5376/4736/3168, затем32n1. Это исполнение и counters,
не численная или скоростная приёмка. Попытки большого resident-e3 oracle
получили resource failures; один runtime OOM загрязнён майнером (ниже).

Для конечной независимой сверки взят обычный PR oracle: e3/F16/ctx40000,
PP≤2048 как у выбранного референса, штатное отсечение невыбранных выходов
FFN59, fusionOFF. Принудительное сохранение всех FFN59 rows отключено;
графы, precision и FORCE_MMQ не менялись. Те же полные input IDs, позиции,
выходные rows и32 teacher-forced n1; causal/KV/RoPE/storage guards прошли.
**Это явно cross-batch сравнение с разным PP-разбиением и head-crop policy.**
Контрольcode128 перед ним побитно равен прежнему e3 gold.

| Cached-e3 против ordinary2048 | Top1 | MeanKL | MaxKL | Итог |
|---|---:|---:|---:|---|
|code128,128rows|100%|0|0|PASS|
|около3000,last32|87,5%|0,226658|1,152634|FAIL|
|16K,tail32|84,375%|0,213821|2,298108|FAIL|
|32K,tail32|68,75%|1,980432|13,538279|FAIL|

Допуски99%/1e-3/0,05 не пройдены, включая отдельные decode-фазы. Все значения
конечны, чистый повтор исполнился rc0; ошибка не списана на OOM. Причина
не локализована до одного оператора: меняются пакетная арифметика и policy
последнего FFN. Новые cached om/GSM/tools по заранее объявленным условиям
не запускались; baseline97/17 этому профилю не приписываются.

## Реализация, память и проверенные компоненты

Модель: четыре пользовательски проверенных шарда, ревизияb17b7f54.
Повторный полный SHA payload не выполняли. Все1188 заголовков проверены:
payload134,019GiB, эксперты124,664GiB. Gate/up преимущественно IQ2_XS,
down IQ3_XXS и три IQ4_XS; NVFP4 нет. Dense преимущественно Q5_K,
частьQ6_K/Q8_0; голова Q4_K,691421184B.

Opt-in `M31_CPU_STOCK_EXPERTS=1` разрешает IQ2_XS в существующих direct/fused/
batched CPU-обёртках через штатные nrc=1 traits, с проверкой geometry/stride.
Прежние операции и порядок редукции сохранены. Старые IQ3/IQ4/NVFP4 форматы
принимаются независимо от нового флага. Нативный GPU FFN уже поддерживал
IQ2 MMQ/MMVQ; добавления нового ядра не потребовалось.

Exact-MMV marker расширен на Q4_K/Q5_K для opt-in batch-dense. Обычный n1
использует прежние MMVQ. DSpark/batch-dense в измеренных режимах выключены.

- CPU:1008 сравнений/53084160 F32 побитно со stock ggml graph, реальные слои
  3/52/59, K1/2/4/8, direct/fused/batched, masks/recovery. Повторено с конечными
  canonical CPU libraries; CPU/CUDA dot/quantizer objects побайтно одинаковы.
- GPU FFN:12 случаев/48stage checks/217810944 F32 побитно compact vs remapped
  padded slots, реальные IQ2/IQ3/IQ4, N1/32/256/1024.
- Dense:208 marked K1–4 случаев/6609920 F32 побитно с отдельными stock n1,
  26role/type/shape signatures; API189 checks с конечным CUDA archive,
  MSA isolation4096 значений без расхождений. Independent scorer проверяет
  нулевые различия, одного rc0 недостаточно.

Canonical `build-port7-canonical/q2-v1`: CPU/CUDA по7ELF,49 этапов PASS,
source/vendor/protectedV8 stable, GCC14/CUDA13.1/sm120a. HIP/SYCL и другие
CUDAarch не проверялись. После build989 существовавших source-файлов
не менялись, добавлены standalone reproducibility helpers/docs.

Baseline runtime: ctx40000, F16 main/index KV, maxchunk32768/adaptive,
CPU80/persistent pool, arena140GiB, cache7GiB/272slots, seed0, native MMQ,
fused CPU IQ2; cached decode/DSpark/batch-dense/fusionOFF. Общий q2-native
JSON наследует CLI arena170; в проверенном immutable resolved profile выбран140.
Reference e3: ctx65536,Q8_0 KV,ubatch/batch2048,threads64/96,CUDA experts10–12,
обычная fusion, без DSpark. Это сравнение настроенных конфигураций целиком.

Логические CPU bytes/token5,117→4,183GB (−18,26%). Host counters свежего
v8→Q2 pooled routed time23,024→21,432 /24,321→21,977 /25,487→19,335мс;
экономия полного интервала2,457/3,134/7,023мс непостоянна. Это не аппаратная
полоса DRAM и не matched-route causal эксперимент. Оценка−4,7мс не обещает
такого же результата на каждом запросе.

В baseline om32K H2D экспертов≈535,4GB против≈818,7GB свежего v8 (−34,6%),
но PP32 вырос лишь≈3,8%. Peak VRAM30167MiB. Вся использованная arena
133857017856B зарегистрирована, slot stride27615616B. Tracked PP host reads
соответствуют router IDs/weights:114reads/1824*n bytes. Это не hardware D2H
профиль. Cached decode дополнительно читает weighted terms на CPU:
294912B/n1 для e3, вне Engine host-read counter; primer≈121,9мс учитывается
в API-фазе, где реально сработал. Prefill и decode счётчики не смешивались.

## Внешний watchdog и восстановление ресурсов

После baseline измерений появился внешний aictl-hold-watch. Он проверял
состояние без lock, затем `aictl miner start` (внутри start_miner с force=True) мог дождаться lock
после нашего up и запустить Keryx рядом с уже активной моделью. Journal/stdout
и source подтвердили гонку. Самый ранний miner start16:58:33 произошёл до
первой записи watch; его инициатор отдельно не приписывается watchdog.

Независимый timeline: baseline speed2/freshv8/e4, обе baseline quality и short
cached закончились до инцидента. Native6656 и первый40K reserve OOM без
пересечения Keryx. Cap16 runtime OOM17:17 и первая ordinary-rest3 попытка
17:40 загрязнены конкурентным майнером; они не доказывают intrinsic memory
limit или numeric FAIL. Замер≈7,4GiB сделан позже; точный VRAM во время раннего
OOM неизвестен. Чистый повтор изолирован:214 NVML samples/209 active,0 посторонних.

В17:41:50 временно приостановлен только главный shell watchdog (SIGSTOP),
child был sleep, aictl lock не заморожен. Следующий собственный aictl up
штатно убрал майнер0. GPU1 не останавливалась. Конфиг watchdog не менялся;
в18:03:28 восстановлен SIGCONT тот жеPID, unit bytes равны исходным.
Readonly NVML monitor остановлен:1066 samples,0 contaminated active.

Финальный post-restore audit PASS15: своих model/worker/experiment PIDs,
listeners и временных preset keys нет, card lock свободен. Все13 исходных
пресетов/top-level keys и protected v8 ELF unchanged, GPU1 miner и CPU-duty
работают. Файл `/run/aictl.miner-hold` сохранён. В момент финального аудита
miner0 был остановлен; восстановленная внешняя policy может вернуть его
на свободную карту. Зрение и его worktree не трогались. Ничего не удалялось.

## Сохранённая работа и следующий технический шаг

Ветка `aiserver/m31-port`, локальные коммиты без push/PR:
a9e482aa(runtime opt-in),adcf5daa(22reprotools),68c8a7e3(bounded cached probe),
3268df5a(5qualification/resource helpers). Docs commit/finalHEAD записан
в `q2-final-summary.json`; он не требует повторной сборки неизменённого runtime.

Следующий вариант — сначала локализовать первое расхождение пакетов, затем
проверить арифметические подбатчи2048 внутри большого слоевого батча:
attention/indexer/dense/FFN выполняются по этим подгруппам, а веса экспертов
слоя остаются на GPU между ними. Это могло бы сохранить амортизацию PCIe
и воспроизводить обычную2048-геометрию. Здесь такой режим не реализован,
его скорость и качество не заявляются. Уменьшение внешнего chunk само по
себе снова увеличивает число проходов по весам.

Основные evidence относительно `/mnt/ai/logs/m31-port/`:

- `q2-wave/results.json`, `q2-reference-e3-e4-selection-v1.json`;
- `q2-native-speed-v1/summary.json`, `q2-native-speed-v2/summary.json`,
  `q2-wave-iq3-v8-control-v1/`;
- `q2-quality-independent-final-summary-v1.json`,
  `q2-reference-quality-e3-v1/quality-post-audit-v1.json`;
- `q2-numeric-contracts-summary-v4.json`, `q2-final-selected-byte-seal-v3.json`;
- `q2-cached-e3-code128-v1/strict-compare.json`,
  `q2-cached-e3-crossbatch-final-summary-v1.json`;
- `q2-cached-e3-crossbatch2048-rest-retry-v2/gpu0-isolation-audit-v2.json`;
- `q2-watchdog-interference-v1/independent-timeline-audit-v1.json`,
  `suspend.json`, `restore.json`, `monitor-summary.json` в том же каталоге;
- `q2-final-resource-audit-post-restore-v1/audit.json`.
