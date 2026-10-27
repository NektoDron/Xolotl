# UD-Q2_K_XL: очередь, сборка и проверяемые результаты

Координатор очереди: `q2_validation`. Текущее состояние и ссылки на результаты:
`/mnt/ai/logs/m31-port/q2-wave/{queue,results}.json`.
Это новая проверка другого GGUF. Прежние оценки IQ3_S не считаются качеством Q2.

## Сборка и неизменяемые входы

`build-port7-canonical/q2-v1` — новая каноническая CPU/CUDA сборка,
49 стадий завершены 2026-10-27 15:38:36 UTC. Снимки исходников и вендора
стабильны, защищённая V8 не изменена. По семь ELF каждого backend;
fixture-проверки CUDA ELF здесь используют CPU backend и не заменяют GPU гейты.
Историческое имя каталога сборщика не означает повторное использование ELF port7.

CUDA generate SHA256:
`2b51de98bbae67d6f9d14a83da1d530d6cf04fa47cdbe7ab917bd99c466e9a8c`.
CUDA serve SHA256:
`2e2a4de354ad3b4fb1e12a343f42425a08401b8b20d3a2edfc423a5bd58a7a90`.

Финальные HTTP задания готовит `tools/m31/q2_validation_prepare.py`.
Они копируют канонические ELF, tokenizer, Python adapter и session runner;
в манифестах сохранены SHA256 клиентов, корпуса, GSM и resolved профиля.
Профиль: CUDA, F16 KV, ctx40000, maxchunk32768/adaptive, CPU80,
arena140GiB/cache7GiB, CPU fused и opt-in stock IQ2_XS,
без cached GPU decode и DSpark. Все 76 ENV и профиль равны между
двумя скоростными кругами и качеством; подготовка проверена в
`q2-wave/final-http-provenance-v1.json`.

`q2_validation_summarize.py` проверяет завершённый session/card result,
финальный serve rc0, SHA входов и полные исходные наборы. Качество —
100 уникальных GSM +20 tool +3 follow-up, temperature1/top_p0.95/max_tokens6000;
`OM_OFF`, `EVAL_TOPK`, `EVAL_MINP`, `EVAL_MAX` удаляются из среды quality клиента.
Приёмка качества и корректность выполнения учитываются отдельно.

## Контракты сравнения

- Начальные stock гейты: старый canonical port7 ELF, CPU RoutedGraph,
  IQ2 opt-in/fused выключены; code128, code3000,16K,32K; maxchunk8192.
- Финальные гейты: новый канонический ELF, тот же resolved профиль,
  что у финального HTTP; в том числе production maxchunk32768/adaptive
  на16K/32K. Oracle повторяет реальные границы native chunk.
- Основной oracle: PR31192, GPU op-offload, CPU storage экспертов,
  F16 KV и выключенная fusion. Это прежний принятый контракт порта.
- Reference HTTP отдельно: штатные CUDA graphs/fusion, q8 KV, ctx65536,
  ubatch2048, CPU64/96,3 либо4 целых экспертных слоя на GPU.
  Его GSM/tools — проверка выбранной рабочей reference конфигурации.
- Дополнительная выбранная code128 проверка GPU resident expert placement
  должна быть обозначена отдельно; она не подменяет основной oracle.

Выбранные четыре случая не называются full gate20. Если oracle требует
меньший физический KV context, исходный OOM сохраняется; разрешённое
capacity equivalence должно иметь проверки contiguous KV, padded nk,
RoPE и фактического chunk plan.

Командные jobs иногда объединены в одну карточную сессию. Производные
per-case receipts содержат исходный combined job/result SHA и точный
реальный argv/log/returncode; отдельные per-case systemd units при этом
не объявляются запущенными.

## Чистота окон и контроль IQ3

Карта0 только через собственный временный aictl preset под общим card-lock;
named down `--if-owned` выполняется после возврата up, затем откат ключа.
Сборки/корректность используют shared build-lock, speed — exclusive.
Quality может идти при shared build-lock; новые speed одновременно запрещены.
Все долгие задачи systemd User=nekto. Карта1, vision worktree и её units
не затрагиваются; miner-hold0 сохраняется.

Дополнительный свежий IQ3 контроль подготовлен в
`q2-wave-iq3-v8-control-v1`: точный защищённый V8 serve,
исходный принятый adapter SHA `e87e70...`, прежние48 ENV,
arena170GiB/cache7GiB/ctx40000/maxchunk32768 и исходный om client.
Это один свежий скоростной круг другого кванта, без нового IQ3 quality прогона
и без утверждения побитно одинаковых random UUID между benchmark запросами.
Исторические принятые медианы и свежий контроль должны оставаться отдельными
строками итоговой таблицы.

Финальные числа, quality и состояние ресурсов берутся из `q2-wave/results.json`
и общего `REPORT-q2kxl.final.md`; до завершения соответствующего job они pending.

## Завершённая скорость и выбранная численная проверка

Два canonical Q2 круга прошли session/server/client rc0 и SHA аудиты.

| Конфигурация | PP4K/16K/32K | TG4K/16K/32K |
|---|---|---|
|Q2 canonical, круг1|1670/1545,2/1299|26,67/24,71/27,85|
|Q2 canonical, круг2|1687,6/1555,3/1305,2|28,90/28,56/27,36|
|Q2 canonical, медианы2|1678,8/1550,25/1302,1|27,785/26,635/27,605|
|IQ3_S protected V8, свежий1круг|1382,6/1539/1254,9|25,97/24,46/23,12|
|llama Q2 e3,1круг|484/532,8/479,1|26,11/25,21/24,37|
|llama Q2 e4,1круг|489,4/539/483,8|25,22/25,02/23,46|

Медианы Q2 выполняют PP32K≥1000/TG≥25 на всех глубинах;
единственная исходная точка TG16K24,71 сохранена, а не отброшена.
Один свежий V8 круг не является устойчивой оценкой нескольких повторов;
UUID и результаты генерации между квантами не фиксировались одинаковыми.
E3 выбран для reference quality: TG выше E4 на всех трёх глубинах,
pooled25,210 против24,541. У E4 немного выше PP, но больше VRAM.

Stock и финальные четыре выбранных гейта с CPU storage экспертов и GPU
op-offload побитно точны: по224 строки, финальные96 n1 forwards.
Production actual PP:16K11904+4448,32K11904+9088+7424+4320,
плюс32 n1 в каждом длинном случае. Original oracle ctx40000 получил
OOM compute reserve21,138GB; обе ошибки сохранены. Авторизованные oracle
physical16384/32768 прошли explicit capacity guard и независимый byte seal.
Native runtime logs показывают PP n/nk; padding32 n1 доказан привязанным
к финальному исходнику `kv_length`, без утверждения о несуществующем
native runtime planner trace для singleton fastpath.

Дополнительный E3 resident expert oracle **не прошёл KL допуски**:
top1=127/128 (≥99%), meanKL0,0105963 и maxKL0,360217
при порогах0,001/0,05. В нём подтверждены слои10/11/12 на GPU,
F16 KV и выключенная fusion, PP96+32n1. Это отдельный контракт от
основного oracle с CPU storage и от HTTP reference q8/64K/default fusion.
Побитность основного oracle не выдаётся за совпадение E3 resident варианта.

Baseline reference E3 quality завершена: **GSM97/100,tools19/23**,
123 уникальные строки, все три follow-up верны, length/leak0.
Ошибки GSM:12,73,85; tools:5,9,11,18. Проверены неизменность исходных
клиентов/данных/frozen файлов и card/session/server/client rc0.
Это один стохастический прогон, а не утверждение о статистическом улучшении.
Baseline native quality завершена полностью независимо от дополнительного
resident oracle результата: **GSM97/100,tools17/23**, length/leak0,
123 строки и все12 integrity/protocol/SHA проверок прошли.
Порог tools≥19 **не выполнен**, поэтому baseline не объявляется принятой.
Ошибки GSM:12,73,89; tools:9,10,11,13,18; follow-up:1.
Дополнительный независимый аудит подтвердил все11 SHA и исходные sampling
параметры; отдельной скрытой смены sampler defaults не обнаружено.
В частности, ответ103,38 вместо103.38 в follow-up1 остаётся официальной
ошибкой набора; score не пересчитывался. Для продолжения
после сбоя контроллер `codex-m31-q2-quality-chain-v1` ждёт завершения
reference execution и cleanup, затем ровно один раз отправляет immutable
native quality через guarded launcher. Состояние:
`/mnt/ai/logs/m31-port/q2-quality-chain-v1/status.json`.
Численная оценка reference не используется для выборочного запуска native.

После **обоих** полных baseline quality и cleanup root разрешил ровно один
дополнительный code128 cached native E3 trial существующего ELF:
cache9,88GiB/384slots, decode-only seed384 для слоёв10/11/12,
ordinary hot0. Эта единственная попытка завершилась rc0 и прошла отдельные
aggregate128 и decode32 проверки побитно против resident E3 F16 oracle,
включая полное покрытие E3 и все33 snapshots counters. Это не результат
baseline и не новая quality/benchmark приёмка.
Guarded deferred launcher и отдельный comparator находятся в
`/mnt/ai/logs/m31-port/q2-cached-e3-code128-v1/`; обычный `card_launch.py`
к уже замороженным helper-файлам этого каталога повторно не вызывается.
`launch_once.py --directory <каталог> --execute` проверяет обе полные
123-row baseline execution, завершение units/worker и SHA, затем создаёт
неповторяемый `ONE_ATTEMPT_STARTED`. Ни пороги quality, ни её отдельные
ответы не выбирают, какой baseline запускать. Последующие long/performance/
новая quality требуют отдельного решения root после результата этого trial.

Root затем разрешил конечный cached E3 профиль maxchunk6656/adaptive,
cache9,88GiB/384slots/seed384, остальные параметры прежние. Новый frozen
профиль SHA `fd78776df17eae2ee1c53ca545e722b7d48b4b0aa4b238e2a81f66d7f99b8acb`;
используются отдельные `q2_cached_validation_prepare.py` и
`q2_cached_validation_summarize.py`. Прежние baseline helper-файлы сохранены
побайтно. Rank и receipt SHA явно привязаны к новым HTTP манифестам.

Все четыре native случая этого профиля завершились; свежие resident E3
oracle shorts прошли. Oracle long16K при requested40000 не выделил
12,988GB compute buffer. Разрешённый physical16384 прошёл reserve
(model15415MiB,KV1920+480MiB,compute6108MiB), но затем получил CUDA OOM
в `ggml_cuda_pool_vmm::alloc` внутри `ggml_cuda_mul_mat_q` на первом actual PP.
Этот abort−6 сохранён в `q2-cached-e3-final-cap16-ctx32-v1`; следующий
long32 oracle не запускался из-за failfast. Позже root обнаружил внешнюю
коллизию `aictl-hold-watch`: он force-started miner0 после нашего up несмотря
на miner-hold. Поэтому этот dynamic-pool OOM **нельзя считать доказательством
внутренней нехватки workspace**; это сохранённый отказ при возможной ресурсной
контаминации, а не измеренное расхождение логитов.
Новые speed/quality не запускались до решения root; их подготовленные
манифесты не считаются результатами.

Новый явно разрешённый контракт ordinary PR oracle использует stock head
pruning без full-FFN59 instrumentation, prefix batch≤2048, F16/residentE3,
fusionOFF/requestedctx40000; native профиль6656 неизменен. Code128 control
побайтно равен старому E3 gold. Для остальных случаев это **cross-batch**
сравнение одинаковых IDs/positions/selectedrows, а не совпадающих PP shapes.
Первая сессия остальных3 была graceful-aborted по STOP root после обнаружения
внешнего miner; прежние данные сохранены, timings не используются.
Root временно SIGSTOP только mainPID внешнего watcher, без конфигурационных
правок; CONT выполняет root после всех наших GPU работ. Новые попытки требуют
runtime NVML проверки GPU0 без чужого miner, отдельно от GPU1.

После ROOTREADY выполнена свежая сессия остальных3 в новых output каталогах
`q2-cached-e3-crossbatch2048-retry-v2`. Её отдельный
`gpu0-isolation-audit-v2.json` прошёл:214 NVML samples,209 при active own
preset, только три собственных oracle PID, без miner/vllm/неизвестных GPU0
процессов. Интервал взят из journal конкретного invocation; первый
неуспешный audit-v1 с нулевыми timestamps после transient-unit GC сохранён
и не используется как доказательство.

Чистое **cross-batch ordinary2048 сравнение не прошло**:

| Случай | Top1 | Mean KL | Max KL |
|---|---:|---:|---:|
|code3000 last32|0,875|0,226658|1,152634|
|16K|0,84375|0,213821|2,298108|
|32K|0,6875|1,980432|13,538279|

Отдельные long decode32 также не прошли. Семантические/provenance guards
прошли, ordinary128 control остался побитно точным. Этот отрицательный
результат относится к явно указанному другому PP разбиению и stock head
pruning; он не подменяется прежним matched-shape short PASS. По условию
конечного плана новые cached speed/quality после этого не запускались;
дальнейшие варианты требуют нового решения root.

## Закрытие волны

Root закрыл testing wave без новых GPU вариантов. Q2 baseline не принят
из-за tools17<19; cached E3/6656 не принят по ordinary cross-batch numeric.
Подготовленные cached2om и новая GSM/tools не запускались по явному условию
плана. Историческая принятая IQ3_S V8 остаётся прежней; новую Q2 quality
приёмку из других конфигураций не переносили.

Финальные read-only resource audits:
`q2-final-resource-audit-pre-restore-v1/audit.json` и
`q2-final-resource-audit-post-restore-v1/audit.json`, оба PASS15checks.
Своих моделей/PID/worker/experiment units/listeners8179–8183 нет, card-lock
свободен, временные preset keys сняты, все исходные модели в aictl и top-level
config keys равны backup. Protected V8 ELF/receipt SHA сохранены, GPU1 Pearl
и CPU duty HTTP живы. Vision worktree/units не затрагивались.

Root восстановил исходный foreign watcher CONT18:03:28UTC, его unit bytes
не менялись; read-only NVML monitor остановлен. В postrestore snapshot
watcher stateS, miner0inactive и GPU0apps отсутствуют; miner-hold сохранён.
В дальнейшем внешний watcher может снова запустить miner0 согласно своей
исходной политике; эта возможная активность не является нашей моделью.
`q2-wave/{queue,results}.json` помечены CLOSED_NOT_ACCEPTED, пустой очередью,
без автоматического продолжения старых prepared jobs. Source/tooling коммиты
делает root локально; push не выполнялся.
