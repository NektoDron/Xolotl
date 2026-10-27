# Port7: логическая полоса и цель +10% TG

Расчёт по переданным root числам, без новых замеров. Единицы десятичные: GB=10^9bytes, TB/s=10^12bytes/s. JSON с арифметикой и actual partial-header intersection — `/mnt/ai/logs/m31-port/port7-logical-bandwidth-ceiling-v1.json`. Logical bytes/time — эффективная полоса приложения, не hardware DRAM/HBM counters: в ней смешаны cache hits, unpack/dot, occupancy и ожидания.

| Категория | Логические bytes/token | Время | Эффективная полоса | Идеал при1.792TB/s | Максимальная экономия по этой модели |
|---|---:|---:|---:|---:|---:|
| Routed CPU |5.117GB |21.500ms |238GB/s |к CPU HBM-предел неприменим |— |
| GPU quant GEMV |9.554GB |6.225ms |1.535TB/s |5.331ms |0.894ms |
| GPU F32 projection/router |1.123GB |0.976ms |1.151TB/s |0.627ms |0.349ms |
| Прочие GPU kernels |— |3.840ms |— |не определён bytes |не оценена |

Даже мгновенно довести обе GPU weight-категории до теоретической HBM полосы даёт лишь≈1.243ms/token. При35.6ms baseline это+3.62% TG. **+10% TG требует уменьшить latency на9.09%, то есть3.236ms**, остаётся ещё≈1.994ms сверх такого идеального GEMV улучшения. Значит, обещание+10% только от приближения уже быстрых weight kernels к HBM peak не обосновано: нужна дополнительная экономия CPU, kernel/launch latency, устранение чтений или перекрытие. Сумма приведённых категорий32.541ms; разницу до35.6ms нельзя целиком назвать sync без matched interval/overlap trace.

CPU250GB/s при тех же bytes означали бы20.468ms (−1.032ms),290GB/s —17.645ms (−3.855ms). Последнее арифметически достаточно для+10%, но raw memory-bandwidth reference240–290GB/s не гарантирует такую скорость quantized routed FFN: vecdot/dequant/FMA, barriers и cache traffic сохраняются. Измеренный private skew/spread дал небольшой GU эффект и не устранил tail; это не доказательство достижения250/290GB/s в модели.

## Что видно по новому кванту без догадок о невидимых слоях

Снимок836 доступных UD-Q2 headers покрывает routed layers3–41. Сопоставление **тех же tensors** с полным UD-IQ3_S:110,717,042,688→90,540,343,296 routed bytes, экономия18.22% в этом известном участке. При4 из128 экспертов это **630,521,856bytes/token меньше только на39 установленных слоях**; layers42–59 не экстраполируются. При условном сохранении238 logical GB/s получилась бы экономия≈2.649ms. Это memory-only потенциал, не прогноз: новый IQ2_XS unpack/vecdot может изменить effective bandwidth и compute cost.

Из доступных dense matrices известна ещё экономия1,147,471,872bytes/token на полном scan соответствующих projections/head —≈0.640ms при1.792TB/s. Embedding storage уменьшился на163,252,224bytes, но **не включён** в этот decode scan: token embedding читает одну строку, а не весь словарь. F32 indexer/router, K/V и часть down сохраняют типы, поэтому их стоимость автоматически не исчезает.

Сумма этих двух условных byte-only оценок≈3.29ms показывает, почему новый квант может быть более сильным резервом, чем polishing старого GEMV. Она не доказывает+10%: отсутствуют полный header audit, runtime поддержка всех opt-ins, новая effective CPU/GPU полоса и качество. Plain stock compatibility уже есть по source, а M3 CPU allowlists/exact GPU cache и Q4/Q5 batch-dense marker требуют отдельного решения. GSM8K/tools/oracle старого UD-IQ3_S на новый UD-Q2 не переносятся; полный размер нового GGUF и качество здесь не предсказываются.

## Обновление после появления четвёртого header

В14:37UTC стали доступны все1188 headers; payload shard4 ещё не завершён. Это даёт фактический полный byte count без экстраполяции: `port7-ud-q2-vs-iq3-header-full-v2.json`. Routed payload163,754,016,768→133,857,017,856bytes (−18.2573%), то есть **934,281,216bytes/token меньше** при4/128. Memory-only оценка при прежних238 logical GB/s:≈3.926ms. Dense full-matrix scan без embedding экономит1,480,132,608bytes/token,≈0.826ms@peak. Суммарные≈4.75ms показывают достаточный арифметический резерв для+10%, но effective IQ2 dot rate, stock-wrapper overhead и quality по-прежнему не измерены; прогноз TG из этих чисел не делается.

Теперь известен полный declared tensor payload134.018948GiB, из него routed124.664063GiB. Это header facts, не peak RAM/VRAM и не подтверждение завершённого download. Основной новый тип —IQ2_XS gate/up3–58, protected59 остаётсяIQ3; down52/58/59 теперьIQ4 вместоNVFP4. Доработки совместимости и отдельные quality гейты остаются обязательными для новых fast opt-ins.

Последний download-status15:11UTC: все4 logical extents совпали с headers, *.aria2 отсутствуют; `port7-ud-q2-download-status-v3.json`. Payload checksum/runtime/quality не проверяли, новую модель не запускали. Это меняет статус файлов, а не доказательность memory-only оценок.
