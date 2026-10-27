# Port7: online LRU / last-token cache

Read-only causal replay реальных8×127 decode forwards (по128 samples/case). Использованы прежние6 training /2 held cases; static rank обучен только на training. Cache сбрасывается между cases; нет переноса состояния training→held и нет доступа к будущим routes. Static290 контроль побитно по counters воспроизвёл прежние held11052/57912hits=19.0841%. Никакой Engine/GPU/сборки/замера скорости.

Все политики имеют capacity≤290 slots. LRU начинает cold либо со static290 seed, после потребления слоя защищает текущие hits и допускает каждый miss; все uploads считаются мгновенно готовыми. Last-token удерживает прошлые228 максимум ключей(layer,expert). Дополнительный вариант pin static62 плюс прошлые token routes занимает≤290.

Held, steady с пропуском первых8 forwards каждого case:

| Политика | Route hit | Полностью hit слой | H2D GB/token | H2D floor при50GB/s | Только H2D ceiling |
|---|---:|---:|---:|---:|---:|
| static290 | 19.28% | 0.494% | 0 | 0 | — |
| lru290_cold | 20.56% | 0.435% | 4.068 | 81.36ms | 12.29t/s |
| lru290_staticseed | 20.56% | 0.435% | 4.068 | 81.36ms | 12.29t/s |
| last_token228 | 20.56% | 0.435% | 4.068 | 81.36ms | 12.29t/s |
| last_token_plus_static62 | 24.22% | 0.966% | 3.885 | 77.69ms | 12.87t/s |

LRU290 здесь практически равен last-token: бюджета хватает на прошлый token working set, а reuse через пропущенный token в основном имеет слишком большую stack distance. Cold и static-seeded LRU после warmup сходятся. Желаемых50–70% hits нет; даже дополнительный pinned static62 даёт лишь24.22%.

Miss streaming требует3.885–4.068GB/token. При35t/s это136–142GB/s H2D, заметно больше PCIe5×16. Даже разрешённые идеально перекрытые50GB/s означают77.7–81.4ms/token — ещё до dense/CPU/GPU compute. При измеренном лучшем raw arena57.6GB/s floor всё ещё67.4–70.6ms/token. Поэтому загрузка каждого CPU miss ради следующего token в текущем бюджете не ускоряет25–27t/s и не ведёт к35–40.

Static seed стоимость указана отдельно в JSON (около6.16GB/case для290), а steady runtime H2D=0. Dynamic расходы — raw оригинальных gate/up/down payload из прежнего tensor-header budget. Дополнительно записана верхняя оценка full-slot copies; мелкое scale/alignment отличие реального загрузчика не воспроизводилось. Мгновенная ready-residency делает оценку hit rate оптимистичной: реальное ожидание DMA может только уменьшить доступные hits.

Это реальные ordinary n1 routes. Спекулятивные неверные ветви могут иметь другую локальность; качество и actual exactGPU arithmetic не проверялись этим симулятором. Уровень raw weight traffic не равен hardware DRAM counters. Отдельный JSON хранит per-case, train/held, all/steady counters, admissions, evictions, H2D и исходные SHA.

Результат: `/mnt/ai/logs/m31-port/port7-online-cache-sim-v1.json`. Воспроизводимый генератор: `tools/m31/port7_online_cache_sim.py`.
