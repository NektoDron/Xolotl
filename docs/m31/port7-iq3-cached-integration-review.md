# Port7: независимый review IQ3 cached integration

Read-only review DSpark агента; runtime/source владельца не менялись,
сборок и GPU запусков в рамках review не было. Проверяется интеграция
проверенных stage1/stage2 в exact cached worker, не итоговая модельная
приёмка. Результат source review: блокирующих ошибок в рассматриваемом
модельном диапазоне E6144/F3072 не найдено. Compile/component/model proof
после интеграции обязателен.

`M31_GPU_CACHED_IQ3_IMMA=1` считывается при создании worker. Новая ветка
работает только для IQ3_XXS, count1 и original_tiled=false; IQ4/NVFP4,
count>1 и tiled сохраняют `GpuDeviceDots::multiply`. При unset/0 worker
не резервирует новый scratch и вызывает старый multiply через тонкий
wrapper. Default ordinary path без explicitly configured exact cache
вообще не создаёт этот worker. GU concat — независимый
`M31_GPU_CACHED_GATE_UP=1`; default0 сохраняет две матрицы.

Q8_K квантование вызывает уже существующий
`gpu_quantize_cpu_compatible(GGML_TYPE_Q8_K)`, тот же block_q8_K292B,
argmax-first, знаковый inverse и round-to-nearest. Перед IMMA нет
переквантования/half scale/abs(d). Stage1 сохраняет8 I32 CPU lanes
на каждый K256 и weight row, odd scales применяет после sub32 integer MMA;
stage2 идёт по K256 в прежнем порядке, затем отдельная .25 после lane
reduction. Подробная арифметическая проверка ранее описана в
`port7-iq3-lane-imma-review.md`.

Область корректности — нормальные производственные CPU Q8_K quants
qs[-127,127]. Произвольный raw int8−128 не покрывается: stock AVX2 PSIGNB
имеет другую семантику. NaN/Inf и патологические reciprocal overflow у
activation не дают дополнительной гарантии из этого review. Никакой
новый raw Q8_K вход в worker не добавлен: он принимает F32 и сам вызывает
прежний quantizer. Compile flags должны сохранить precise division,
subnormal/FMA semantics isolated proof; старый production NVCC command
не содержит --use_fast_math, CUDA defaults precise, FTZ=false. Новый
combined receipt всё равно нужно проверить, а не полагаться на имя TU.

В constructor сначала создаётся private nonblocking stream, затем общий
quant scratch, затем новый I32 scratch. Для E6144/F3072 reserve_iq3_n1
берёт columns=max(E,F)=6144, rows=max(E,2F)=6144:4,718,592B. Этого хватает
на gate/up отдельно, joined GU и down. `scratch_bytes()` учитывает эту
ёмкость, worker.device_bytes включает её; lazy Gpe worker добавляет
полную сумму в decode_scratch_bytes. В штатных вызовах этого диапазона
worker не делает realloc между kernels. Shape validation и64-bit byte
calculation защищают scratch sizing; произвольные экстремальные размеры
вне модели/допустимого CUDA grid этим review не принимаются.

Quant→stage1→stage2→D2H идут по одному worker stream. Reuse quant и
partials безопасен: последующие kernels в том же stream видят порядок;
объект не допускает overlapping begin. Grow/free дожидается prior CUDA
users, destructor сначала drain. Перед kernel submission pending=true;
launch error выбрасывается, worker catch синхронизирует stream до done.
finish rethrows только после завершения worker; outer lease drain
срабатывает и при исключении CPU callback/размере результата.

Веса читаются из leased VRAM pool: base+layer.role_offset+slot*stride,
не напрямую из pageable RAM arena. `active_read_keys` исключает eviction
до завершения реального raw stream. Gpe destructor уничтожает exact
workers до pool/uploads/registered arena, поэтому borrowed pointers не
переживают owner. Source arena registration новым kernel не меняется.

GU join допускается только при одинаковом type, contiguous [gate;up]
роль span, up_pointer=gate_pointer+raw_gate_bytes, 2*raw<=slot_stride,
с проверками overflow. Одно n1 input квантование обслуживает две группы
rows; row-local accumulation неизменный. После D2H gate/up scales
применяются отдельно, CPU OAI прежний, down scale и router weight остаются
двумя отдельными multiply, merge сохраняет original rank order.
Mixed-role layers fallback на старую пару dot calls. GROUPED1 сейчас
не использует joined GU; это допустимая текущая граница feature.

Новый helper Stats имеет другой размер: Gpe/helper/worker header/объект
надо собирать согласованно. Нельзя подменить только CUDA object в старом
несогласованном worker ELF. Для принятия нужны actual wrapper/FFN proof,
модельная ordinary parity и clean same-ELF speed. Отсутствие замечаний
к исходникам не означает35т/с или достижение1,4TB/s.

Reviewed source SHA256:

- `gpu_iq3_lane_exact.cuh`: `bef9fec7639e5bad128913c22a35ba0af4c3508db9ed04b9f8136ed98180220b`
- `gpu_iq3_lane_exact.hpp`: `fdf4718bc7c3f3f80dcc244e1df168271e5c6479cce729e2c6be856d88f1b730`
- `gpu_cpu_quants.cu`: `0aba42ee8aa7fc06a4132c23cf5f78c442dd3ed90ec0234eac806e8f60ac6b5b`
- `gpu_resident_dot.hpp`: `a6768ffbc3035a82160a4101f76147328e21c3d6e36ba95d843b2902e3767857`
- `gpu_cached_exact.cpp`: `5339d7232106cbecf86eb61df7ae112b69eca5dd54484da95cde172314815087`
- `gpu_cached_exact.hpp`: `9ef6dd686183fccd2fed4c8958a777d07927e89b065a4677be57b22dbe2f18a8`
- `gpu_cached_decode_impl.hpp`: `0d6681468244deb55c55d18d9d043e4919fa60ed015dd4185381766efd3b5d99`
- `gpu_cached_batched_impl.hpp`: `488604cb6f180639d2cf726637d1102ea7b64c950cf6f180ebd87d4780f0654b`


## Интеграционный component: независимый audit PASS

После source review собран coherent snapshot
`build-port7-cached-imma-gu/v2/source`, включая production GPU dots,
IQ3 headers и actual cached worker. ELF
`33b8afd12ae5c1c9b382135c790d929cce58ec738662d2885fe92d2ecfa2655a`
заморожен отдельно в job; оба GU0/GU1 запуска используют один ELF.
Независимо проверены SHA каждого source в build receipt, самого ELF,
замороженной копии ELF, копий build/freeze receipts и fixture.json.
Записанные compile flags: NVCC O3/ftz=false/fmad=true/prec-div=true/
prec-sqrt=true; CXX no-fast-math/frounding-math. Оба command rc0.

`/mnt/ai/logs/m31-port/port7-cached-imma-gu-component-v1/` содержит
`component-GU0.json`, `component-GU1.json` и независимый
`independent-iq3-gu-audit-v1.json`. Оба варианта60cases/1,474,560 weighted
F32 побитно равны CPU n1 original-rank terms: всего120cases/2,949,120F32.
Исходник harness делает memcmp expected CPU terms до timing; это не
сравнение только двух GPU реализаций. Рабочие веса original IQ3 gate/up
и down IQ3/IQ4/NVFP4, реальные128×6144 L3 activation rows, слои3/10/52,
K1..8. Последние rows переиспользуются как fixture для разных слоёв;
это не captured activations каждого текущего слоя модели.

Из60cases на каждый GU вариант20 действительно использовали grouped
путь,6 проверяли128B gap между gate/up. GU0 нигде не concat-ит GU;
GU1 concat-ит только27eligible cases с GROUP0/no-gap. GROUP1 и gap
сохраняют fallback; metadata cases симметричны между двумя arms. В
GROUP0 actual IQ3 IMMA counter>0 для каждого случая; суммарно в timed
вызовах7020 IMMA calls при GU0 и4700 при GU1. Count>1 и other-type
fallback дополнительно проверены source dispatch; отдельный счётчик
каждого fallback типа не добавлялся.

Во всехcases дополнительный device scratch ровно4,718,592B, pinned
storage прежний2,162,688B. GROUP0 base2,169,720→candidate6,888,312B;
GROUP1 base2,218,860→6,937,452B: отличие базового grouped quant/metadata
reserve учтено, это не утечка. Каждый timed job имеет2joins, прежний
CPU scales/OAI/weight/rank merge. Ни weight layouts, ни Q8 lattice не
менялись. Current actual extra scratch входит в stat/workspace владельца.

N1 GROUP0/no-gap whole-helper median speedup (capacity-thrash fixture):
GU0 range1,489–1,662×, median1,538×; GU1 range1,306–1,725×,
median1,607×. Это не bandwidth/TG, не обещание ускорения ordinary engine.
8physical replicas и minimum reuse distance превышают actual96MiB L2,
но аппаратные hit/miss counters не измерялись. Модельная ordinary greedy
сверка и clean same-ELF A/B/A идут следующими отдельными gates;
этот документ их заранее не принимает. Произвольный raw Q8−128/NaN/Inf
и полный all-layer динамический activation domain по-прежнему не
покрываются component proof.


## Same-ELF ordinary A/B/A: numerical PASS, speed winner отсутствует

`/mnt/ai/logs/m31-port/port7-p7v11-ordinary-imma-rank-aba-v1/`
завершён rc0; независимый `independent-aba-audit-v1.json` проверил manifest,
job/scorer/profile hashes, frozen v11 ELF
`ade1d45b6a3d9590efa5857f1a312f98f12c229fba9ecea8e475068724455c0b`,
source timer span и все три immutable logs. Код не менялся в рамках audit.

OFF/ON/OFF — один обычный greedy128 на каждом arm, prefix3324 одним
actual chunk[[0,3324,3328]], ctx40000/chunk32768/80CPU threads. Без
sidecar, capture, Engine.verify, batched MSA/ops, lookup, GU и GROUPED.
ON включает exact cached subset+deferred hot290+IQ3 IMMA+rank-outer merge.
Это сравнение пакета, не отдельная изоляция IMMA или rank-outer.
После нормализации output path все argv одинаковы; ENV delta ограничен
этим пакетом и его sealed seed contract; OFF2 profile совпадает сOFF.
Полная inherited runtime ENV attestation не выдумывается.

| Arm | Decode ms | TG |
| --- | ---: | ---: |
| OFF1 | 4888,91 | 26,1817 |
| ON | 4894,06 | 26,1542 |
| OFF2 | 5107,91 | 25,0592 |

Консервативный ON/max(OFFs)=0,99895; выигрыш не доказан, порог+10%
не пройден. OFF drift−4,29% запрещает трактовать+4,37% к позднемуOFF как
устойчивое ускорение. Все128 generated IDs каждогоarm совпали, SHA
`414644292d608e255614fa935779e21beded07fb3e6f63db587a9383db452f3f`.
Это локальный ordinary proof, не gate20/GSM/tools/fullom acceptance.

Actual ON counters:10020 IQ3 IMMA calls/39,647,232 rows,
2806 rank-outer calls и hybrid forwards; GU counters0. Lookup29184,
resident3567 (12,2225%), seed1/6,159,384,576B/108,198ms. OFF counters0.
Primer входит в serial span128forward+logits: его не вычитали. Этот
standalone timer считает prompt-anchor как первый output и вычисляет
последний unused next sample; конвенция одинакова для всехarms.

Главный оставшийся предел виден без сложения пересекающихся интервалов:
last96decode rows имеют dense wall12,490/12,468/12,386ms дляOFF1/ON/OFF2;
вложенный dense read11,624/11,626/11,604ms практически неизменен.
Routed critical wall22,738/21,980/25,060ms: ON немного уменьшает первую
оценку, позднийOFF показывает CPU drift. Head~0,72ms сread остаётся
похожим; shared late wait~0,06–0,10ms, поэтому именно поздний shared join
не объясняет основной бюджет.

GPU exact worker484,745ms за весьON, CPU misses876,479ms только у
hit layers, hybrid join55,147ms иsubmit19,615ms — вложенные/перекрытые
счётчики. Их нельзя прибавлять к routed critical илиdense wall. Нельзя
называть разницу routed total−GPUworker «чистым CPU» без timeline.
Primer108ms соответствует0,845ms на128 измеренныхforward и поглощает
часть небольшого steady выигрыша; вычитать его для заявленной скорости
не разрешено. Текущий hit rate12,2% на этом corpus ограничивает долю
CPU traffic, на которую влияет kernel ускорение. И helper1,5×, и этот
A/B/A вместе не доказывают35т/с; default флаги остаютсяOFF.
