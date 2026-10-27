# UD-Q2_K_XL: нативный GPU FFN экспертов

Для IQ2_XS в штатном префилле не требуется новое CUDA-ядро. Аудит исходников
27.10.2026 выполнен на базе `d6e96c4b`; он сам по себе не является приёмкой
модели или скорости. В этой части волны runtime-код не менялся.

Итог волны — **NOT_ACCEPTED**. Базовый Q2 Xolotl дал GSM8K97/100,
tools17/23 при требовании19. Отдельный cached E3 candidate прошёл только
короткий побитный контроль. Его native6656 execution/counters прошли, но
чистые ordinary2048 cross-batch проверки на3000/16K/32K не прошли допуски.
Новые speed/quality этого candidate не запускались; defaults принятой IQ3v8
сохранены.


`m31_gpu_experts.cpp` и `gpu_device_experts_impl.hpp` создают gate/up/down через
`ggml_mul_mat_id` с фактическим типом каждой матрицы. Нативный путь оставляет
gate, up, swiglu_oai, down, weighted sum и residual на карте. IDs и веса роутера
загружаются отдельно; чтение промежуточного FFN в RAM этому пути не требуется.
IQ2_XS не попадает в ограниченный список CPU-exact CUDA и не должен включаться
в нём простым расширением allowlist.

В замороженном vendor `build-port7-canonical/v2/vendor-managed` IQ2_XS есть в
MMQ, MMVQ и GET_ROWS. В `ggml_cuda_mul_mat_id` малый batch выбирает штатный
MMVQ, а большие batch на sm_120 используют MMQ. Решение о MMQ на NVIDIA с
tensor cores возвращает true для IQ2_XS при достаточном shared memory.
Q4_K/Q5_K/Q6_K также штатно поддерживаются; DSpark exact-MMV dispatch проверяет
другой агент отдельно.

Размещение GPU-кеша считается по реальным byte sizes. Для смеси IQ2_XS,
IQ3_XXS, IQ4_XS кратность stride — LCM(128,74,98,136)=3 945 088 байт.
Максимальный raw span трёх ролей у защищённого слоя 59 — 24 477 696 байт.
Component с реальным CUDA backend подтвердил slot stride 27 615 616 байт;
бюджет 7 GiB даёт 272 slots (6,99558 GiB), два банка по128 — 6,58408 GiB.
Хотя Q2 содержит меньше экспертных байтов, LCM и защищённый слой увеличивают
slot относительно старого IQ3. Нельзя переносить старое число slots/rank seed.
Реальные allocation/slots подтверждены init receipts ниже.

Отдельный component `tools/m31/q2_gpu_slot_check.cpp` сравнивает компактные
штатные GGML CUDA tensors с padded cache views и перестановкой slots. Он
читает исходные полные IQ2 gate/up и смешанные IQ3/IQ4 down экспертов
0/17/67/127 в слоях3/52/59, N=1/32/256/1024. Gate/up/down/final сравниваются
побитно, включая исходный порядок weighted rank sum и shared/residual add.
Входы синтетические, ограниченные, все матрицы полных размеров6144/3072.
Оба плеча используют GPU-математику, совпадение с CPU не требуется.

Это проверка slot geometry/remap и stock GPU dispatch, не сравнение с полной
llama.cpp моделью и не тест пропускной способности. Кандидат component имеет
четыре slots; поведение полного272-slot pool проверяется модельными gates.
Для проверки выходов component читает gate/up/down на хост после завершения
графа; это диагностическое чтение не добавляется в рабочий FFN.

Сборка подготовлена `tools/m31/q2_gpu_slot_build.py` против канонических
port7 CUDA libraries, одним CXX compilation. Она сохраняет source snapshot,
точные команды, хеш ELF и receipt исходной канонической сборки. CPU/HIP/SYCL
runtime не менялись; другие backends этим CUDA component не проверяются.

Результат component **PASS**: 12 cases, 48 проверок стадий,
**217 810 944 F32 значения** побитно равны, все finite, max_abs=0.
Фактические backend allocation подтвердили max span/LCM/stride из расчёта.
ELF `b8feb9ca95591dde6686e00dc98d371beb6374e90ed3eac9057e4b433192297d`.

Результат: `/mnt/ai/logs/m31-port/q2-gpu-slot-component-v1.json`, SHA256
`95af8712d6b9759cde764691c8ba28fb43262bcfe061e2ecd333128a52e97a8d`.
Build receipt: `build-q2-gpu-slot/v1/build-receipt.json`.
Запуск: unit `codex-m31-q2-components-stock-native-v1`, команда0;
лог `/mnt/ai/logs/m31-port/q2-components-stock-native-v1/q2-components-stock-native-v1-card-job-00.log`.
Временный aictl worker/общий card lock предоставил validation; GGML fusion
выключен. Других GPU модельных процессов этот компонент не запускал.
Модельные короткие/длинные gates и speed/quality имеют отдельные receipts.


## Production geometry: загрузка и counters

`/mnt/ai/logs/m31-port/q2-gpu-final-production-native-audit-v1.json` проверяет
готовые native correctness jobs канонической Q2 сборки: ctx40000,
max_chunk32768, fusedCPU1/stockIQ2wrapper1, native GPU FFN. Все четыре случая
загрузились и завершились. Реальные272slots занимают7 511 447 552байт;
вся арена133 857 017 856байт зарегистрирована для прямой H2D.

16K проходит PP чанками11904/4448 и32 singleton строками. 32K —
11904/9088/7424/4320 и32 singleton строками; padded Nk соответственно
12032/20992/28416/32768. GPU FFN workspace сохраняет high water
1 755 697 152байт. Это проверка реальной формы больше8192; component сN1024
и4slots не заменяет её численную сверку с oracle. Используется одна
MMQ family. НаNVIDIA tile J зависит отn, а усреднениеn*used/slots применяется
только наRDNA3/4. Разныеn всё равно требуют собственных model gates.

32K counters:224prefetch batches,28 672experts,526 141 882 368байт — точно
четыре полных прохода по слоям4–59 согласно GGUF headers. Прямые загрузки
первого routed слоя —9 268 051 968байт/511experts. Всего535 409 934 336байт.
В stock diagnostic с максимальным chunk8192 было пять проходов и
669 230 678 016байт: payload сократился19,9962%. Это сравнение H2D payload,
а не утверждение о приросте скорости.

На всех PP tracked Engine host reads равны114reads и1824*nбайт:
57слоёв × (4I32IDs + 4F32weights). Вместе с source audit это подтверждает,
что рабочий expert FFN не возвращает gate/up/down в RAM. Счётчик не включает
head/final диагностические чтения и не является аппаратным профилем D2H.

Guard/rank/source audit: `q2-gpu-native-independent-audit-v2.json` в том же
каталоге logs. Свой aictl preset, card lock, build lock, active-model guard,
`--no-switch`, cleanup named `--if-owned`, ожидание up и rollback своего ключа
проверены. Старые rank/seed поля unset; PREFETCH_RANK=1 без загруженного rank
оставляет обычный порядок0..127. CPU-exact CUDA/GpuFastFfn/cached decode/DSpark
в этих профилях выключены. Все correctness jobs используют shared lock и
не служат чистыми скоростными замерами. Full-model oracle gates и HTTP
om_bench имеют собственные receipts и проверяются отдельно.


## Чистые HTTP om_bench: две серии

Проверены завершившиеся `q2-native-speed-v1` и `q2-native-speed-v2`:
measurement=true с exclusive build lock, исходный `~/om_bench.py`, все
HTTP/client/server returncode0. Независимые лёгкие log audits не запускали
других probes/сборок и не перечитывали большие ELF/libraries. Двадцать шесть
config/live checks совпали: CUDA0,ctx40000,chunk32768,t80,fusedCPU1,
stockIQ2wrapper1,nativeGPU,272slots,прямая зарегистрированная арена.

| Серия | PP 4K /16K /32K, t/s | TG 4K /16K /32K, t/s |
|---|---:|---:|
|v1|1670,0 /1545,2 /1299,0|26,67 /24,71 /27,85|
|v2|1687,6 /1555,3 /1305,2|28,90 /28,56 /27,36|
|Среднее двух|1678,8 /1550,25 /1302,1|27,785 /26,635 /27,605|

Это две штатные om серии со случайными UUID и немного различными prompt IDs/
числом токенов. Они не составляют парный эксперимент с совпадающими входами.
На16K v1 TG24,71 ниже25; вариативность не скрывается средним. На32K обе
серии дают PP>1000/TG>25. Полную приёмку качества эта таблица не объявляет.

На32K обе серии исполняют chunks11904/9088/7424/4604, padded Nk
12032/20992/28416/33024. Expert H2D535 409 934 336 и535 391 797 248байт;
разница одного эксперта первого routed слоя согласуется с разными routes.
Prefetch в обеих сериях точно равен четырём проходам по слоям4–59.
Tracked PP host reads остаются router-only в указанном выше ограниченном
смысле. Peak VRAM обеих серий30 167MiB; PCIe width16 во всех telemetry samples.
Gen1/2 встречается при startup/idle, активные samples показываютGen5.

Сводный machine audit:
`/mnt/ai/logs/m31-port/q2-native-speed-pair-gpu-log-audit-v1.json`.
Покадровые request/chunk/counter/config проверки:
`q2-native-speed-v1-gpu-log-audit-v2.json` и
`q2-native-speed-v2-gpu-log-audit-v2.json` в том же каталоге.
Репродуктор только чтения логов — `q2_gpu_speed_log_audit.py` там же.


## Свежий IQ3 v8 контроль: скорость и байты считаются отдельно

Свежий контроль защищённой V8 — `q2-wave-iq3-v8-control-v1`, одна штатная
om серия: PP1382,6/1539,0/1254,9 иTG25,97/24,46/23,12. Среднее двухQ2
относительно этого контроля даёт PP+21,42/+0,73/+3,76%, TG+6,99/+8,89/+19,40%.
Контроль один, Q2 серий две, UUID/входы различаются. Эти данные не доказывают
устойчивых+10% на всех глубинах и не заменяют историческую IQ3 quality приёмку.

Raw expert H2D на4K/16K меньше18,257%. На32K у свежей V8 пять PP проходов
10240/7936/6528/5632/2682 и818 696 650 752байт. УQ2 четыре прохода и
в среднем535 400 865 792байт: меньше34,603%, тогда как API PP вырос3,76%.
Это разные измерения, их нельзя подменять друг другом или объяснять весь
результат только меньшим размером весов.

Вместе с байтами поменялись операции: gate/up112tensors IQ3→IQ2,
down12tensors IQ4→IQ3, триNVFP4down→IQ4; основные dense используютQ5_K/Q6_K,
headQ6_K→Q4_K. GPU resident dense weights10 679 794 176→9 199 661 568байт;
F32 индексер/роутер/norm не переквантовывались. Изменены стоимость декодирования
блоков кванта, модельная математика/routes, формы чанков и tile dispatch.
CPU routed counters и API timings анализируются отдельно, причинная
атрибуция без отдельной абляции не заявляется.

Machine review сравнений/типов и границ утверждений:
`/mnt/ai/logs/m31-port/q2-fresh-v8-gpu-comparison-review-v1.json`.


## Один ограниченный E3 cached-native diagnostic

После расхождения CPU routed с E3 oracle, у которой три полных expert layers
10/11/12 резидентны наGPU, подготовлен **один** дополнительный code128 trial
существующего канонического ELF. Единственный разрешённый запуск завершился
после обеих baseline qualities и cleanup: все128строк, включая отдельно32n1,
побитно равны E3 gold, max_abs/KL=0. Это не меняет baseline quality профиль.
Подробности и состояние —
`/mnt/ai/logs/m31-port/q2-cached-e3-code128-v1/manifest.json`.

Native cached helper уже использует штатные IQ2/IQ3 MMVQ иGPU swiglu_oai;
нового ядра или runtime изменения не требуется. Важное ограничение: helper
проверяет все cache entries, отдельного whitelist слоёв нет. Pool11GiB имеет
427slots: seed384 оставил бы43 случайных prefill entries и GPU hits вне E3.
Поэтому candidate задаёт9,88GiB — ровно384slots/10 604 396 544байт — и
полный decode-only seed384 экспертов слоёв10/11/12. После priming весь pool
состоит из этого набора, остальные54routed layers возвращают all-missCPU.

Это synthetic placement permutation, **не calibration или hot-frequency
ranking**. У всех7296records count=0, training_route_count=0, calibration=none,
source_trace=null. Поле legacy weight_sum явно хранит placement priority1/0
для совместимости с сортировкой прежнего формата; каждый record содержит
отдельное placement_priority. Наблюдённые usage counts/route weights не
выдумывались. Обычный preflight проверил integrity/geometry/model identity;
он не доказывает калибровку или качество. SHA четырёх GGUF унаследованы из
пользовательской проверки и `m31-q2kxl-download.tsv`, свежего134GiB scan нет.

Frozen профиль/rank/receipt/helpers и отдельный metadata audit подготовлены.
ELF — hardlink уже read-only frozen canonical inode; большой SHA был
повторно проверен после завершения обеих baseline qualities перед единственным
trial.
Guard требует123уникальных kind/id каждой baseline, card/http RC0, завершение
контроллеров и cleanup worker; score thresholds baseline не ограничивают
запуск диагностики. После этого разрешён один attempt marker/один job,
никаких автоматических long/perf/quality каскадов.

Наблюдённые counters на32n1 совпали сsource ожиданием:7296route lookups,384resident routes,
96hybrid forwards, один seed6 964 641 792байт; вместе сPP57 получается153GPU
FFN forwards. All4hits строго в трёх E3 layers следуют из полного seed/pool
identity и подтверждены вместе с реальными aggregate counters; отдельной
per-layer hit telemetry нет.
Primer входит в фактическую API фазу; при обычном PP, заканчивающемсяn>1,
этоTG. При singleton PP tail он может попасть вPP; вычитать его нельзя.

Source-only VRAM оценка: после увеличения pool остаётся примерно7,498GB
вместо10,591GB при7GiB cache. Начальная adaptive proposal может стать8448
вместо11904. Это не allocation/performance proof; prefill≥1000 и выигрышTG
для этого candidate не обещаются. Bounded code128 against E3 gold с
отдельными aggregate128 иtail32 гейтами прошёл; этого оказалось недостаточно
для приёмки длинной геометрии.


Реальный short-trial GPU audit:
`/mnt/ai/logs/m31-port/q2-cached-e3-short-gpu-independent-audit-v1.json`.
Все33снимка совпали с57+3iGPUforwards,228i lookups/12i hits/3i hybrid;
primer ровноодин6 964 641 792байт/121,929ms, после него weight admission нет.
Cached graph scratch664 320байт. Эти timings относятся к correctness job,
они не являются om performance acceptance.

Для decode нужно отдельное уточнение D2H: native cached helper возвращает
наCPU четыре уже взвешенных rank terms —98 304байт наhit-layer,294 912байт
наn1длятрёхE3layers. CPU складывает их в исходном порядке. Это чтение не
попадает вEngine::host_reads; router-only относится только к device prefill.
HTTP reset/rearm/cumulative counters разобраны в
`q2-cached-e3-http-seed-lifecycle-source-audit-v1.json` вlogs.

Следующим отдельным решением root разрешён candidate с maxchunk6656/adaptive,
ctx40000,cache9,88GiB итемже384seed. Замороженный resolved profile:
`/mnt/ai/logs/m31-port/q2-cached-e3-final-profile-v1/runtime-profile.json`, SHA256
`fd78776df17eae2ee1c53ca545e722b7d48b4b0aa4b238e2a81f66d7f99b8acb`.
ENV/unset/rank/seedreceipt полностью совпадают с shortPASS; единственное
runtime CLI изменение32768→6656. Это консервативный VRAM cap, не измеренный
оптимум. Четыре model gates повторно проверили новую PP геометрию; длинные
численные проверки не прошли. Скорость/качество нового candidate не измеряли.
Все GPU jobs запускал validation, не этот аудит.


6656 native четыре correctness cases завершились с правильными counters:
16K PP6656/6656/3040,32K6656/6656/6144/5376/4736/3168. Raw PP H2D
401 534 779 392 и803 105 832 960байт соответственно, плюс по одному primer
6 964 641 792байт при32n1tail. После primer веса не загружались. На32K
минимальная залогированная свободная GPU память2 244 542 464байт. Это не
clean om и не численная приёмка; подробности —
`q2-cached-e3-final6656-native-gpu-counter-audit-v1.json` вlogs.

Свежий E3 oracle с физическимctx40000 наlong16 не создал контекст:
compute запрос12 988 499 968байт/12 386,80MiB. Его resident model15 415,48MiB
и main/indexerKV5 887,50MiB вместе с compute требуют33 689,78MiB при видимых
32 110MiB. Это ресурсная ошибка доforward, численного mismatch нет; long32
в failfast job не запускался. Native успешен. Ошибка сохранена отдельно,
профиль не менялся. Root уже явно разрешил16/32capacity-equivalence fallback после сохранённого
40KOOM каждого случая. Validation использует отдельные outputs и trace
активныхKV/trainRoPE/source guards; дополнительное подтверждение не требуется.
E3storage и native профиль не меняются.
Состояние — `q2-cached-e3-final6656-gpu-preflight-v2.json` вlogs.


Source erratum: ранние GPU audit JSON неверно распространили формулу выбора
MMQ J поn*used/slots наNVIDIA. Висходнике это условие толькоRDNA3/RDNA4;
наsm120 используетсяn. ne02 продолжает влиять на grid/expert_bounds/streamK,
но не на этотJ heuristic. Измеренные proofs/counters/байты от исправления не
меняются. Исторические receipts сохранены; исправление перечисляет
`q2-gpu-mmq-ncols-erratum-v1.json` вlogs.


Physicalctx16384/cap6656 E3 oracle прошёл init, затем получил динамический
OOM в ID MMQ activation Q8 allocation. Стек не доказывает размер свободной
памяти: граф до этого вырос с7007до7949nodes, новый compute reserve не
залогирован. Это не tmp_fixup allocation; MMQ residentne2=128 не умножает
Q8activation на128 — используетсяused=4. Наsm120 Jheuristic используетn.

Root разрешил один ordinary E3 oracle с PPcap2048 и штатным pruning перед
finalFFN59; native6656 остаётся прежним. Отдельный oracle-only overlay
`build-q2-oracle-stock-head/v2` собран против неизменённых PR archives,
ELF SHA256 `8bc774f8d826aa14030ccc73939ba8f65db63b31d976cc193614fef948896b0a`.
Env `M31_ORACLE_STOCK_HEAD_PRUNING=1` пропускает layer-input tap; при
head_tail>0 manifest честно задаёт full_last_layer_before_crop=false и
force_full_ffn59=false. У allrows code128 control full_last_layer_before_crop
остаётсяtrue, force_full_ffn59=false. На каждом decode ordinary trace
проверяет causal attention и KVseq0..end−1, выводит paddedkeys/actualctx и
параметры trainRoPE. Source receipt —
`q2-oracle-stock-head-source-audit-v1.json`. Native/runtime/kernel не менялись.
Контроль128 побитно совпал с прежним E3 gold; остальные cases сравнили
между6656native и2048ordinary по пользовательским допускам. Эта разница
геометрии/отсечения FFN явно сохранена; длинные численные проверки провалены,
source/build audit не является скоростной или модельной приёмкой. Неуспешный overlayv1 сохранён: строгий patch-anchor
остановил preparer до компиляции; v2compile/link завершилсяRC0.


Чистый retry ordinary2048 завершил все три cases сRC0, но численно **FAIL**:

| Selected positions | Top1 | mean KL | max KL |
|---|---:|---:|---:|
| code3000, последние32PP | 87,50% | 0,226658 | 1,152634 |
| 16K,32teacher-forced n1 | 84,375% | 0,213821 | 2,298108 |
| 32K,32teacher-forced n1 | 68,75% | 1,980432 | 13,538279 |

Results: `q2-cached-e3-crossbatch2048-retry-v2/{code3000,long16384,long32768}/compare.json`
вlogs. Порог99%/meanKL1e−3/maxKL0,05 не менялся; shortcode128all128 иtail32
остаются побитным PASS. Native maxchunk6656/fullFFN59 и ordinary2048/pruned
FFN59 имеют разные формы, что явно записано вcontract; тест не доказывает
ошибку отдельного MMQ ядра или конкретную причину расхождения. Его результат
не позволяет принять общий cached E3 режим. Короткий PASS не переносится
на длинные позиции или качество.

Инцидент с посторонним Keryx разобран отдельно в
`q2-watchdog-interference-v1/independent-timeline-audit-v1.md` иJSON.
Baseline Q2speed1/2, свежий IQ3v8, llamaE3/E4speed, обе baseline qualities
и первый cachedcode128 завершены до restart16:58:33UTC и этим инцидентом
не затронуты. Native6656fourcases также не пересекли activeKeryx послеownup.
Первый ctx40K/cap6656 compute-reserve OOM был без overlap; это сохранённый
ресурсный отказ конкретной геометрии. Но cap16 runtimeOOM17:17:42–17:17:54
пересёк restart17:17:47, а первый ordinaryrest3 пересёк restart17:40:43
и отменён. Эти два окна **контаминированы**: нельзя объявлять cap16OOM
доказанной изолированной нехваткой VRAM, а первыйrest3 — чистым modelgate.
Стек локализует failingcall, но не доказывает intrinsicworkspaceblocker.
Непрерывного VRAM trace раннего OOM нет; точные7,4GiB втотмомент не
приписываются.

Cleanordinaryretry подтверждён отдельным
`q2-cached-e3-crossbatch2048-rest-retry-v2/gpu0-isolation-audit-v2.json`:214NVML
samples,209ownactive,0foreignGPU0apps; все три modelкоманды завершилисьRC0.
Этот результат отличен от отменённого контаминированногоrest3. Ни одно из
этих correctness окон не объявляется чистым замером производительности.

Возможное будущее направление — микробатчи2048 внутри слоя при сохранении
residentweights, чтобы ограничить activation/graph память, не перечитывая
слой на каждый подчанк. Это новый дизайн с проверкой causalMSA/KV, crop59,
математической геометрии и lifecycle; он не реализован и не имеет доказанного
выигрыша скорости или качества. В этой волне новых flags/jobs после FAIL нет.
