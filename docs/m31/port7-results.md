# MiniMax-M3.1-Flash → Xolotl: итог port7

**Новый быстрый режим не принят: устойчивых +10% к v8 и цели35–40ток/с
не получили.** Точные полезные реализации сохранены opt-in; профиль
`native-gpu.json` и защищённые ELF принятой port6 v8 не изменены.
Каноническая port7 сборка CPU/CUDA прошла, выбранные численные проверки
точны. Финальные GSM8K/tools/full gate20 нового режима не запускались:
по уточнённому решению координатора они требовались при наличии winner≥10%.
Результаты97/100 и19/23 ниже относятся **к старой принятой v8**, не к port7.

## Скорость и сохранность исходного пути

Одна карта RTX5090, CPU80, arena170GiB/cache7GiB/seed0, context40000,
F16 main/index KV, adaptive maxchunk32768. Новые режимы port7 выключены.
Исходный ~/om_bench.py не изменён: warmup,4/16/32K,256samples,greedy/ignore_eos.
Первый круг canonical использует исходный клиент; парный контроль — frozen-копию
того же алгоритма с fixedUUID/responseobserver, явно отдельную от полной приёмки.
Замеры выполнялись под exclusive build-lock без наших сборок/CPU probes.
Соседняя сессия зрения сериализовалась общим card/build-lock; её не останавливали.

| Контекст | Принятая v8 PP | Принятая v8 TG | Свежая v8 PP/TG | Port7 canonical PP/TG |
|---|---:|---:|---:|---:|
|4K|1390,85|27,445|1381,53 /25,56|1379,30 /26,15|
|16K|1535,00|27,015|1542,21 /25,27|1532,31 /24,40|
|32K|1246,65|25,305|1255,80 /23,89|1251,58 /24,87|

Первые два столбца — прежние медианы двух прогонов port6. Последние два —
**одна свежая парная диагностика** V8→canonical с фиксированными UUID,
одинаковыми rendered/native prompt IDs, sampling,48старыми+27нулевыми ENV,
одинаковым tokenizer и одинаковой копией старого Python adapter
с одинаковыми observation-only wrappers.272проверки strictscorer прошли: actual PP pos/requested/n/nk, native outputs,
ACK и parsedchoices совпали. Равны все800nativeoutputs:32warmup+3×256.

PP canonical отличается на−0,16/−0,64/−0,34%; TG — на+2,33/−3,46/+4,10%.
Это не устойчивый выигрыш и не доказательство буквального отсутствия любой
регрессии по каждой глубине. Суммарная TG по времени трёх256-token запросов:
24,884→25,117ток/с (+0,94%). На32K **обе свежие сборки ниже25**, хотя PP≥1000.
Новую полную приёмку по минимумам не объявляем. Официальной принятой
конфигурацией остаётся port6 v8 с её прежними quality/speed receipts.

Первый отдельный круг canonical дал PP1382,1/1540,9/1256,2 и
TG25,81/25,82/24,69. Поэтому мы не сослались на ранний baseline как на
сопоставимый контроль, а повторили старый V8 на тех же входах и в свежем окне.
Ещё ранний same-day V8 давал27,90/27,01/25,98. Причина изменения текущего
CPU state не установлена; температуры/фон не являются доказанной причиной.

В свежей паре dense read на32K13,873→13,871мс практически равен, routed
24,289→23,150мс. На16K routed23,680→25,207мс меняется в обратную сторону.
CPU quants.c.o побайтно одинаков, hot pool functions и compiler/ISA flags
одинаковы; fused callback одинаково выделяет72B стека, нового1KiB spill нет.
Новые CUDA compilation flags добавляют−O3/−DNDEBUG; по interval audit это
не выглядит основным источником снижения скорости, causalабляции не было.

Сопоставление с прежним llama.cpp23,33/22,63/21,92TG и425–433PP остаётся
в отчёте port6: его принятая v8 была быстрее на15–19%TG и≈2,9×PP32K.
Текущую одноразовую пару не выдаём за новый согласованный llama.cpp benchmark.

## Что реально ограничивает декод

Actual CUPTI trace V8:124steady token windows,3754kernels/token,
GPU kernel union11,037мс, DMA union0,330мс. Host compute_ms — enqueue;
read_ms включает исполнение GPU.57normD2H по24KB занимают≈0,039мс DMA,
а связанные host memcpy API ждут предыдущие GPU графы≈9,09мс.
Поэтому прежние12мс read нельзя целиком считать потерями синхронизации.

| Работа GPU | Логические bytes/token | Время ядер | Эффективная полоса |
|---|---:|---:|---:|
|Q6_K+Q8_0 fixed weights|9,5536GB|6,225мс|1,535TB/s|
|F32 indexer projections/router|1,1230GB|0,976мс|1,151TB/s|
|Все unique dense weights включая norms/head|10,6798GB|не отдельная activity|не hardware counter|

Голова ужеQ6_K:200064×6144,1,0083GB. ПредложениеF16→Q8 к ней неприменимо.
Точный greedy argmax требует оценок всего словаря; переквантование головы
потребует отдельной проверки качества и даст лишь малую долю выигрыша.

Внутренние паузы60densegraphs≈0,339мс/token, shared≈0,0085мс.
CUDA graphs уже используются; stock PDL/graph допускают перекрытие ядер.
Суммы отдельных kernel/role интервалов нельзя складывать с CPU wall:
например dense kernelunion≈8,525мс против≈10,138мс суммы activities.
PDL-only причинную абляцию не делали.

CPU routed читает логически5,117GB/token; ранние≈21,5мс дают≈238GB/s,
близко к измеренным240–290GB/s референсным диапазонам. Идеальное доведение
только указанных GPU weight categories до условных1,792TB/s сэкономило бы
≈1,243мс — лишь+3,62% от35,6мс/token. Для+10%TG нужно−3,236мс latency.
Это условная оценка по bytes, не гарантированный предел по всем операциям.
Реалистичные30–35ток/с требуют экономии также CPU/индексера/координации;
ещё один CUDA graph или ускорение уже насыщенного GEMV не обещает35–40.

## Точный кеш экспертов и новые ядра

Native cached hot162 на выбранном code128 отвергнут: top1≈88,28%,
meanKL≈0,382/max≈15,426. Это не общий запрет native GPU; редкие seed0 hits
на том же тесте укладывались в допуски, но не давали нужного выигрыша.
Точный cached helper использует прежнюю activation quantization/dot,
CPU libm OAI и исходный порядок weighted rank sum.

Новый IQ3 integer IMMA сохраняет I32 partials восьми CPU lanes и исходный
FMA/reduction epilogue. Raw1059+real684cases дали I32/F32diff0;
cold gate/up≈60→20µs, около3× на отдельной матрице.
Интеграционный FFN proof:120cases/2,949,120weightedF32 побитно CPU,
N1..8, реальные weights/scales, GROUP/gap fallback, workspace accounted.
N1helper median speedup1,538×GU0/1,607×GU1, ranges1,489–1,662/1,306–1,725.
Rank-outer merge bitwise1281cases/4rounding modes:11,8–18,4→1,8–2,0µs;
это10–16µs на hit-layer, не весь GPU join.

Clean same-ELF3324/128 ABA cacheOFF→exact290+IMMA+rank→OFF:
**26,1817→26,1542→25,0592ток/с**, все128 IDs равны.
Conservative ON/max(OFF)=0,99895; OFF drift−4,29%, +10% не подтверждён.
Primer6,159GB/108,198мс включён TG. Реальны10020IMMAcalls/2806rankcalls,
hits3567/29184=12,22%. GU был0; GU1 имеет componentproof, не этот modelgate.
Сравнение с прежним exact-cache run: worker6,378→3,787мс/token,
join1,286→0,431мс. Это cross-run, не текущие OFF→ON counters:
у OFF worker отсутствует. Общего выигрыша нет.
Steady slices нельзя «улучшать» вычитанием primer из принятого таймера.
CPUmiss counters покрывают только hit-layers и пересекаются с GPU helper.
Гипотеза gross starvation CPU expf worker не подтверждена измеренными временами.

RMS-only fusion — отдельный mode2, только stock RMSNorm→MUL, с сохранением
fences; broad unfenced fusion ранее отвергнута по качеству.116componentcases/
408064F32 точны. Реальные отдельные OFF/RMS/LOOKUP модельные teacher-forced128tailrows
побайтно logits/tokens равны, top1=100%, KL/maxabs0. Это short gate,
не long/full20/performance acceptance. Ожидаемая экономия RMS≈0,30мс.
N1policy включает также singleton prompt/tail; n>1 unfused. Defaults0.

## DSpark, CPU и prefix

DSpark target layers[4,18,32,46,60] — residual outputs HF[3,17,31,45,59].
Наивные joinedK2/K8 и batched FA на длинных Nk отвергнуты. Точный layer-major
верификатор сохраняет n1 attention/reduction, rollback/partialcommit;
selected16K/32K anchor+tail32 против GPU-offload oracle побитно точны.
FusionOFF; capacity-equivalence guard сопоставил native context40000 с oracle
actual16640/33024: activePP/nk/RoPE совпали. Это выбранные long cases P7-v8, не full20 финального canonical ELF.

Actual HTTP depth1 TG24,54/25,76/23,16; depth3 22,08/21,72/20,97 для4/16/32K
(verifier2/4rows соответственно).
NativeIDs/PP recipes равны между глубинами, ordinary-arm в этом job нет.
Clean RPC ordinary25,348→DSpark23,035; coalesceON24,667→22,807.
Lazy prepare151–166мс входит TG, verify93–95%TG; отдельного seed_ms нет.
У MoE bytes экспертов растут почти с числом проверяемых токенов, а точная
геометрия внимания сохраняет значительную часть dense работы. DSpark defaultOFF.

CPU weightedbatch K2/4/8≈1,07/1,13/1,18× component; T80 лучше64/96 именно
в этом CPUbatch sweep. Paired quant обычного n1 не дал устойчивого выигрыша.
Private spread80 component≈1,8%, MWAITX latency хуже PAUSE, power не измеряли.
Последний полный model placement ABA:26,0889→19,8664→23,5870ток/с,
все128IDs/PP равны, conservative0,76149. ONstrict80 реально закрепил main
и вспомогательные потоки на CPU93; startup≈112,5с. Это проверка пакета
placement+inheritance, не изоляция одной причины замедления. Default не меняли.

THP16GiB own mapping получил100% huge pages за0,117с. Полная153GiB arena
с существующими THP1/FORCE1/register1 не завершила startup за180с; наш PID
штатно остановлен. Рано затронутые2,575GB арены были100%AnonHugePages,
но post-registration marker/smaps и ONTG не получены. Это timeout feasibility,
не mathFAIL и не доказательство register-split. После освобождения freehuge
84,95GiB<guard190,63GiB; force не повторяли, global settings не менялись.

Exact prefix cache — opt-in только полного одинакового single-chunk prompt;
GPU cold/reuse/cold byteproof PASS. Partial/multi-chunk reuse выключены.
Bounded capture VIEW имеет split3840+256/lifetime/epoch/raw53checks;
CPU controller9720cases/55728rows иhost958checks PASS.
Actual HTTP VIEW cancel/next и pairedVIEWPP отложены; не заявляются проверенными.
Server greedyDSpark/grammar/temp1 fallback/cancel recovery имеют smoke,
но не заменяют GSM/tools. Vision этой волной не затронут.

## UD-Q2_K_XL: что нужно дальше

Проверены заголовки всех1188tensors: declaredpayload134,019GiB,
experts124,664GiB против152,508GiB. На15:10 `.aria2` sidecars отсутствуют;
checksum payload и runtime этой модели в port7 не проверялись.

- Gate/up слоёв3–58: IQ2_XS,112tensors/72,84375GiB.
- Gate/up59 и54down: IQ3_XXS,56tensors/48,234375GiB.
- Down52/58/59: IQ4_XS,3tensors/3,585938GiB. NVFP4 отсутствует.
- Основные dense веса Q5_K, частьQ6_K/Q8_0, headQ4_K; это другая математика.

IQ2_XS уже есть в reader, stock CPU vecdot и CUDA MMQ/MMVQ/getrows.
Plain ordinary CPU RoutedGraph fallback и native GPU prefill по source готовы.
Custom direct/fused/batched/cached allowlists IQ2 пока не разрешают; это может
свести обычный decode к более дорогому stock graph, скорость неизвестна.
Минимальный следующий шаг — bounded opt-in stocktraits IQ2 в CPU wrappers
с exact component/model proof; CPU pair helper может fallback на stock dots.
Exact GPU cached нужен отдельный IQ2 kernel, новый IQ3 его не заменяет.
DSpark batch-dense marker допускает F32/Q6/Q8, поэтому Q5projections/Q4head
требуют нового dispatchproof либо BATCH_DENSE0. Старые ranks/quality не переносить.

По полным headers routed bytes−18,2573%: условно−934,3MB/token,≈3,926мс
при238GB/s; dense scan−1,480GB/token,≈0,826мс при1,792TB/s. Это byte-potential,
не прогноз скорости: стоимость IQ2 decode/dispatch и качество ещё неизвестны.
Следующая волна должна начать с compatibility/short-long GPU-oracle,
затем реальных om/GSM100/tools23 на новом кванте.

## Работа шести субагентов

| Агент | Результат |
|---|---|
|dense_latency|CUPTI/bandwidth/gaps, exact MMV API, RMS/lookup и численные proofs|
|cached_decode|Калибровка кеша, точный IQ3/GU helper, rank merge, model ABA|
|routed_cpu|CPU batch/pair/perf/skew/placement, THP и полный UD-Q2 header audit|
|dspark|Draft/точный verifier/rollback/controller, независимые math/source audits|
|prefix|Exact prompt cache, capture/view lifetime, matched-ID наблюдение и index|
|validation|Единая очередь GPU/build, immutable receipts, clean benches/gates/resource audit|

## Сборка, коммиты и границы приёмки

Canonical `/mnt/ai/Xolotl-m31/build-port7-canonical/v2`: CPU/CUDA по7ELF,
49stages PASS, source/vendor stable, protectedV8 before/after unchanged.
Первый canonicalv1 CPUbuild прошёл, но fixture не импортировала gguf_writer:
исправлен support snapshot, повтор выполнен в новомv2, failure не удалён.
GCC14/CUDA13.1/sm120a; HIP/SYCL иполныйMSVC/ARM model build не выполнялись.
Canonical CPU fixtures запускаются CPUbackend даже для CUDAELF; это не GPUquality.

GenerateSHA `b31953cd7fcc656dc5955abe4f3d2f4f9eae982836908ae1ba2b301943f27c24`.
ServeSHA `230e88fefeb40b9be83edf41636ea8f9b65010f381bc942d5b28ada6af8f6e49`.
Canonical defaultcode128 отдельно PASS128rows против GPU-offload oracle,
top1=100%, KL/maxabs0. Full20/GSM100/tools23 этого ELF **не проводились**.

Локальная ветка `aiserver/m31-port`, базаfaeb1332:
`57229a20` —60runtime/fixturefiles; `dd4bc35b` —99selectedreprotools.
Итоговый docs commit указан в `port7-final-summary.json`. Пуша/PR нет.
Historical/stale/unapplied probes сохранены локально untracked, не удалены
и не подключены автоматически к production. Модели не удалялись/не качались нами.
После canonical убран только trailing space standalone GU proof cpp,
tokens идентичны; runtime/CMake/конечные target sources не менялись.

## Ресурсы и главные receipts

Эксперименты завершены, новых GPU/quality jobs не назначено. Итоговый audit
15:10/15:13 PASS: наших model processes/listeners8180–8182/активных experimentunits нет,
свои temp preset keys восстановлены, защищённые m31/hxq3k presets совпали сbackup.
/run/aictl.miner-hold присутствует; майнер карты0 остановлен, pearl-miner
карты1 и CPU llm-chat работают. В момент аудита
GPU0 принадлежит legitimate чужой m31-vision-test; её не трогали.
Прод/up* и ветка/worktree зрения не изменялись, системных THP/clock/NUMA правок нет.

Основные пути относительно `/mnt/ai/logs/m31-port/`:

- `port7-evidence-index-v4.final.json` и `.final.audit.json` —37variants/26components,
  metadata index с явным отсутствием новой quality/release acceptance.
- `port7-canonical-v2-fixed-uuid-pair-v1/paired-comparison.json` — свежий контроль.
- `port7-canonical-v2-default-code128-v1/` — выбранный GPU-oracle gate.
- `port7-p7v11-ordinary-imma-rank-aba-v1/` — точныйcache ABA и независимые audits.
- `port7-p7v11-rms-lookup-gate-v1/`, `port7-cached-imma-gu-component-v1/`.
- `port7-dense-v1/{v8-bandwidth-breakdown-v1,v8-graph-gap-audit-v1}.json`.
- `port7-canonical-v2-thp-aba-v1/`, `port7-canonical-v2-cpu-placement-aba-v1/`.
- `port7-ud-q2-header-audit-v2.json`, `port7-ud-q2-vs-iq3-header-full-v2.json`.
- `port7-final-resource-config-audit-v1/audit.json`.

Полные тематические объяснения и списки файлов — `docs/m31/port7-*.md`.
Волну закрываем по согласованному условию остановки: устойчивого winner≥10%
нет; для текущего кванта оставшиеся оценки не дают обоснованного обещания35–40.
Главный следующий рычаг — меньше bytes/token при подтверждённом качестве,
а не обещание ускорить уже насыщенные GEMV вдвое.
