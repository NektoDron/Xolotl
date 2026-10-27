# Port7: оценка multiple-K persistent cache

Источник: завершённый диагностический V5 sweep20arms, prefix3324,
ctx40000, shared_inputs1/batch_dense1/cache0/fusionOFF. Новый runtime или
замер не запускался. Числа получены суммой уже записанных build_ms и
reconfigure_ms; это не om_bench и не прогноз итоговой скорости.

Артефакт: `/mnt/ai/logs/m31-port/port7-multik-evaluation-v1.json`.

| depth / p_min | K в verify | build+reconfigure | Доля decode wall |
|---|---|---:|---:|
| 1 / 0 | 2 | 49,3 ms | 0,9% |
| 1 / 0,5 | 1,2 | 717,9 ms | 10,4% |
| 1 / 0,8 | 1,2 | 1229,4 ms | 16,4% |
| 3 / 0 | 2,4 | 111,7 ms | 1,7% |
| 3 / 0,5 | 1,2,3,4 | 885,5 ms | 13,4% |
| 7 / 0 | 2,8 | 158,7 ms | 1,8% |
| 7 / 0,5 | 1,2,3,4,5,7 | 1124,1 ms | 15,2% |

Для текущего лучшего fixedK/p_min0 это вторичный рычаг. Для confidence arms
частая смена K съедает заметное время. Даже заведомо недостижимое вычитание
всех717,9ms из depth1/p0,5 даёт20,7т/с против реально23,59 у depth1/p0.
Расчёт не включает повторную warmup/capture/instantiate стоимость в
compute_ms; полный выигрыш может отличаться. Сохранение состояний также не
устраняет первую подготовку каждой K/геометрии и смену PAD256 bucket.

Возможный bounded design: отдельные state по K для одного uniform nk bucket,
общая CPU executor отдельно, лимит суммой реальных allocator bytes с GPU
reserve. При переходе nk bucket или mode flags все owners освобождаются
согласованно; mixed-boundary nk vectors держать отдельно. На ctx40K несколько
состояний могут помещаться в свободные~6GB после draft, но на64K запас меньше.
Нужно измерить actual V7 shared_graph workspace до выбора лимита.

Любой Graph destructor очищает ВСЕ CUDA captures backend. Поодиночная LRU
эвикция одного K сбросит captures остальных K; её стоимость нельзя считать
локальной. Все shared consumers освобождать до main norm-owner, graphs до
AttentionInputs/hidden buffers. Planner обязан учитывать сумму всех
retained states. Эти условия пока описаны, реализация не добавлена.

## Дополнение: повторные CUDA capture warmups

Root справедливо указал, что build/reconfigure исключают warmup/instantiate
в compute_ms. Второй разбор использует только uniform nk3584 хвоста V5:
median compute у age≥3 после последнего reconfigure становится reference
для каждой K. Для age0/1/2 суммируется max(0,compute−reference). Это отделяет
дорогой первый повтор после geometry rebuild от обычной GPU работы.

Reference medians: K1=14,105ms (393samples), K2=21,544 (97), K3=29,091 (61),
K4=36,568 (51), K5=46,755 (42), K7=62,044 (только2), K8=69,853 (31).
Неподдержанные K6 нельзя оценить этим методом. На fixedK K3 первая
стабильная геометрия повторяется с compute100,4ms против steady29,1;
на K4 повтор132,3ms против36,6. Поэтому одних build_ms действительно мало.

| depth / p_min | build+reconfigure | Доп. compute age0..2 | Условный потолок t/s |
|---|---:|---:|---:|
| 1 / 0,5 | 718 ms | 674 ms | 23,23 |
| 1 / 0,8 | 1229 ms | 1012 ms | 24,31 |
| 2 / 0,5 | 967 ms | 1049 ms | 24,50 |
| 2 / 0,8 | 1397 ms | 1092 ms | 25,68 |
| 3 / 0,5 | 886 ms | 924 ms | 26,65 |
| 3 / 0,8 | 1305 ms | 1011 ms | 26,74 |

Условный потолок вычитает также обязательные первую подготовку и смену PAD
bucket, игнорирует drift CPU/GPU frequency и разницу числа steady samples.
Это оценка причины задержки, а не обещание результата. Для confidence arms
multipleK может оказаться существенным после view/packing/overlap; для
текущего лучшего fixedK/p_min0 всё ещё мало geometry flips.

Полный разбор: `/mnt/ai/logs/m31-port/port7-multik-warmup-evaluation-v1.json`.
Runtime cache по-прежнему не реализован.
