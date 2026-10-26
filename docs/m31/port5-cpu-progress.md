# Port5: CPU decode executor

## Проверено

`cpu_direct_decode.hpp` сохраняет оригинальные CPU `vec_dot` и `from_float`
из ggml traits, scalar OAI и последовательную F32 сумму исходных top-k ranks.
Поддерживаются IQ3_XXS/IQ4_XS/NVFP4 и исходные strides арены/модели.
Один bounded persistent executor обслуживает GU, OAI и down; default выключен.
Engine-интеграцией владеет агент dense_decode, env `M31_CPU_DIRECT=1`.

Независимый CPU-only probe сопоставил четыре эксперта с переставленными IDs
и weights .1/.2/.3/.4 на настоящих слоях3/10/52/58/59: все выходы побитно
равны исходному ggml CPU graph, включая NVFP4 scale tensors. Повторено
при16/32/48/64 потоках —20 случаев, ноль отличающихся F32.

`cpu_threadpool_config.hpp` даёт явные настройки `M31_CPU_POLL=0..100`
и `M31_CPU_AFFINITY=0-63` либо список/ranges. Без переменных параметры ggml
не меняются. GpuHostExecutor получил optional explicit CPU affinity; worker
placement устанавливается явно, caller mask восстанавливается после submission,
включая исключения. Tiny test проверил наследование single-CPU mask, реальные
worker CPUs, restore и malformed env. Это важно: ggml strict affinity сначала
закрепляет вызывающий поток; созданные после него std::threads иначе наследуют
его единственное ядро.

Canonical `m31.cmake` теперь включает executor.cpp и Threads независимо от
CPU_QUANTS_CUDA. CPU-only `m31_generate` и `m31_serve` собраны; NumPy tiny
chunks1/7/29 сохранили maxabs0.00011285, arena/workspace/pool/residual побитны.
Unit `codex-m31-port5-cpu-canonical-build-v1` завершён rc0.

## Скорость: прямой CV executor отвергнут для performance default

Изолированное CPU component окно,5 повторов после warmup, original expert
weights четырёх экспертов. GPU не использовался. `codex-m31-port5-cpu-sweep-v2`
rc0, frozen probe SHA256
`65e9e549bbd66bf77c3c310c43a632a6b70fea39ca04d33a249c3fe6ee410202`.

| Потоки | Слой | Исходный graph, мс | Direct CV executor, мс |
|---:|---:|---:|---:|
|16|3|1.162|3.107|
|32|3|0.645|2.937|
|48|3|0.451|3.732|
|64|3|0.401|1.851|
|64|10|0.460|2.225|
|64|52|0.431|2.034|
|64|58|0.908|2.037|
|64|59|1.010|2.571|

Это warm component probe, не om_bench и не конечный TG. Прямой путь
в текущей реализации медленнее; **M31_CPU_DIRECT оставить0** в основном
performance preset. Нельзя обещать экономию от удаления graph ops без замера.
CPU trait nrc исключён как причина: исходный `mul_mat_id_one_chunk` также
вызывает vec_dot с nrc1, IQ3_XXS/IQ4_XS.nrows1. Рабочая гипотеза — CV wake/
mutex herd против spin ggml pool; отдельного количественного proof пока нет.

Первая sweep-v1 завершилась rc139 внутри повторного reference graph: probe
загружал inputs/IDs единожды, хотя gallocr scratch мог их переиспользовать.
Каждый compute теперь загружает входы заново, как настоящий Engine.
Исправленный sweep-v2 rc0. Оба запуска сохранены в append-only
`/mnt/ai/logs/m31-port/port5-cpu-sweep-v1.log`, разделены ELF SHA.

## Следующая попытка — fused ggml pool

`cpu_fused_decode.hpp` готовит один `ggml_custom_4d` FFN node на существующем
CPU backend/pool, с пятью исходными ggml_barrier фазами. Quant, dots, scalar
OAI, scales и исходный rank order сохраняются; второй CV pool не нужен.
Есть borrowed accessor weighted per-rank contributions для будущего hybrid.
`codex-m31-port5-cpu-fused-build-v1` rc0: те же пять реальных слоёв при16
потоках побитно CPU graph, affinity test PASS. Frozen ELF/source/manifest
сохранены в `/mnt/ai/logs/m31-port/port5-cpu-fused-v1/`, ELF SHA256
`ace3dcd1ce725e29c15a1c51f1df22eec0b92f9a05669173041646e01f1c5d4c`.
Изолированное component окно `codex-m31-port5-cpu-fused-speed-v1` rc0,
только64потока,5слоёв×5повторов, все результаты побитны CPU graph:

| Слой | Graph, мс | Fused, мс | Снижение времени |
|---:|---:|---:|---:|
|3|0.383821|0.316972|17.4%|
|10|0.476277|0.407215|14.5%|
|52|0.442286|0.365758|17.3%|
|58|0.888333|0.821065|7.6%|
|59|1.007423|0.854768|15.2%|

Это warm expert component, не реальный TG. Агент dense_decode добавляет
`M31_CPU_FUSED=1` в следующий engine candidate, отдельно от отрицательного
CV direct пути. Integrated code128/decode и итоговый TG проверяет координатор.

## Parallel pack в pinned ring

`cpu_pack_spans.hpp` использует существующий bounded GpuHostExecutor,
не создавая новой реализации scheduler. `M31_GPU_PACK_THREADS=1..16`,
default1; кандидаты4/8. Все3role spans одного эксперта копируются одним
job, с независимыми1MiB stripes и синхронным завершением перед enqueue H2D.
Весами и pinned staging helper не владеет; новых больших allocations нет.
Валидация всех descriptors/overlaps выполняется до первой записи.

`codex-m31-port5-cpu-pack-build-v1` rc0:25MiB,3role spans, threads1/4/8,
полные bytes/guards/reuse/empty/overlap rejection PASS. Это byte gate,
не bandwidth measurement. Интеграцией pool ownership в GPUExperts и
двумя upload rings владеют gpu_cache/stream_prefill. Результата PP для
parallel pack пока нет.

Отдельное изолированное DRAM copy окно
`codex-m31-port5-cpu-pack-speed-v1` rc0: источник1GiB,40 rotating segments
(больше aggregate L3), два destination slots по25MiB, три role spans,
128 проходов/3.355GB полезных bytes на каждое плечо:

| Потоки | GB/s |
|---:|---:|
|1|21.12|
|4|86.15|
|8|134.51|

Для candidate выбран `M31_GPU_PACK_THREADS=8`: +56% к4 потокам,
6.37× к serial. Это pageable CPU component, не pinned H2D или PP.
Actual worker logs показывают1/4/8 потоков, inherited affinity192CPU,
для parallel плеч132 jobs с warmup. ELF/source сохранены в
`/mnt/ai/logs/m31-port/port5-cpu-pack-bandwidth-v1/`, ELF SHA256
`d7d7bf0a273eadbe247b7c95be944f472e02523707928da24b4529d7dbb9a30c`.
Raw log — `/mnt/ai/logs/m31-port/port5-cpu-pack-speed-v1.log`.

GPUExperts получил caller-owned `forward_into`, а Engine использует
`host_reads.moe` для повторного выходного буфера: прежний local result
создавал≈73/201MB zero-initialized vector на каждом слое при2968/8192
токенах. Арифметику и исходный route histogram этот API не меняет.
Новая engine-интеграция и её gates/PP — ответственность координатора;
их результат не выводится из component copy скорости.

## Runtime worker configuration

В старых correctness jobs обнаружено имя
`M31_GPU_CPU_QUANTS_HOST_THREADS=32`; код читает `M31_GPU_HOST_THREADS`.
При отсутствии правильной переменной GpuCpuQuants намеренно использовал1
поток, поэтому OAI8.16s/quant4.38s/reduce3.95s из большого diagnostic gate
были serial. Подготовленные server speed arms использовали правильное имя;
им не приписывается эта ошибка. Старые immutable jobs не переписывались.

`gpu_host_options_from_environment()` теперь отвергает эту опечатку и
`M31_GPU_SORT_ROUTES`, сообщая правильные имена. Constructor каждого
persistent host executor печатает фактические threads/workers, threshold,
grain и CPU affinity. Исходник GPU profile также получил actual threads и
counters serial/parallel jobs для следующего candidate build, чтобы worker
configuration была видна до teardown.
`codex-m31-port5-cpu-runtime-build-v1` rc0: typo rejection, validHOST32,
explicit/inherited affinity и caller restore проверены.

## Canonical CPU build после v3 source freeze

`build.sh` теперь явно задаёт CPU_QUANTS_CUDA=ON для CUDA и OFF для CPU;
default jobs24. Старый implicit cache больше не определяет наличие exact CUDA.

`codex-m31-port5-canonical-final-cpu-v1` завершён rc0. Snapshot wrapper
использовал shared build lock, чтобы не блокировать независимый GPU correctness
gate; measurements в это время не запускались. Source SHA до/после build
совпали. Canonical inspect/generate/serve/expert_check собраны; comparator4/4,
tiny NumPy1/7/29 maxabs0.00011285, arena/workspace/pool/residual bit parity,
server14pass/8 optional vocab skips из22. CPU ELF и manifest сохранены в
`/mnt/ai/logs/m31-port/port5-canonical-final-cpu-v1/`.

`codex-m31-port5-canonical-final-cuda-v1` также завершён rc0: explicit
CPU_QUANTS_CUDA=ON, CUDA17/sm120a, pinned/WMMA headers, canonical targets,
comparator и tiny/math/arena/pool checks. После oracle/tokenizer build
отдельный CPU-only unit `codex-m31-port5-vocab-final-check-v2` прошёл все8
real vocab/tokenizer checks за3.721s. Вместе с14 protocol checks это закрывает
все22 доступных проверки. Первый vocab wrapper v1 неверно назвал unittest
class и завершился до запуска helper; исправленный v2 и старый error log
сохранены раздельно.

После согласованного guard для известного неверного QUERY_TILE>0 и правки
comments выполнен quick canonical CPU+CUDA Engine TU rebuild/relink,
`codex-m31-port5-canonical-final-relink-v2` rc0. Численная q0 ветка не менялась.
Final Engine SHA256:
`8811d150292a0581a55d1782aa788773d4de5be8ed15dbe6c6fb88f50996a526`.
Canonical GPU model inference отдельно не запускался: численные gates и
изолированные speed windows используют согласованные immutable candidate
ELFs координатора. Во время полного om_bench окна дополнительных ELF copies
или hashes не выполнялось.

## Fused pool64/80/96

`codex-m31-port5-cpu-threads-speed-v1` rc0, isolated window, только ggml
и fused (без отрицательного CV direct),5 реальных слоёв×5 повторов:

| Потоки | Среднее fused по5слоям, мс |
|---:|---:|
|64|0.342903|
|80|0.292626|
|96|0.867050|

Все15случаев побитны CPU graph.80 потоков быстрее64 на14.66% в этом
warm component.96 здесь медленнее64 в2.53раза; причину не локализовали.
В ggml_barrier нет отдельного перехода80→96; CPU placement/power/load
не измерялись, поэтому ими регрессия не объясняется как факт.
Предложен candidate `--threads 80`, реальный TG ещё должен его подтвердить.
M31_CPU_FUSED не имеет ограничения64; оно относится только к CV direct.
Affinity default не менялся.

Frozen ELF/source находятся в `/mnt/ai/logs/m31-port/port5-cpu-threads-v1/`,
SHA256 executable `c92785524a3720ee500cd3f62378364110475169c2ee54aa28e53ffada4190ee`;
raw log `/mnt/ai/logs/m31-port/port5-cpu-threads-speed-v1.log`.

## Итоговая конфигурация принята по M1

Frozen v4, CPU80/HOST32/PACK8/pinned, final full20 unit завершён rc0:
20/20,28723 позиции, top1=100%, mean/max KL0, `M1-complete-logits`.
`/mnt/ai/logs/m31-port/port5-gpu-v4-final-gate20-v1/compare.json`.
End-to-end итоги двух4K/16K/32K и actual CPU counters сведены отдельно
в `port5-cpu-final.md`: TG27.51/26.385/25.44, PP215.05/231.3/237.9.
Генерация выиграла16–18%; prefill остаётся ниже reference, M2 частичен.
