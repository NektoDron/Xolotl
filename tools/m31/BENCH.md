# Замеры M2

`bench_m31.py` готовит план и сводит сохранённые результаты. Он не запускает
модель или внешние команды. Замер остаётся в `~/om_bench.py`, проверка качества —
в `~/eval_duty_1023.py`, как у референса.

```bash
python3 tools/m31/bench_m31.py --out /mnt/ai/logs/m31-port/bench-plan-v1.json \
  plan --arm m31-xolotl-v1 --port 8179 \
  --directory /mnt/ai/logs/m31-port/bench-v1
```

План содержит argv, environment, SHA256 клиентских скриптов/корпуса/REPORT,
контракт API и ожидаемые выходы. Каталог результатов должен быть новым:
`om_bench.py` дописывает TSV, `eval_duty_1023.py` перезаписывает JSONL.
Все команды исполняются последовательно координатором из aictl worker под
card/build locks, systemd-run User=nekto с логом в `/mnt/ai/logs/m31-port/`.
Прямой запуск helper не разрешает занимать карту; aictl управляет моделью.

Сначала smoke 3 запроса, затем warmup и два круга 4K/16K/32K, затем качество.
Speed выключает thinking, temperature=0, max_tokens=256, ignore_eos=true,
cache_prompt=false. Tokenizer самого сервера режет тот же корпус; случайный
префикс исключает reuse. prompt_n включает chat template и инструкцию,
поэтому он больше номинальных 4096/16384/32768. Серверу нужны `/tokenize`,
`/detokenize`, `/v1/chat/completions` и реальные `timings` в ответе.

Качество — первые 100 GSM8K плюс 20 tools и 3 продолжения; max_tokens=6000,
temperature=1.0/top_p=0.95, thinking по умолчанию модели. Явные EVAL_TEMP/EVAL_TOPP
существенны: комментарий существующего eval скрипта не соответствует этим
параметрам референса, а его температура по умолчанию равна 0. EVAL_TOPK/MINP
план очищает. Tools проверяются как JSON, реальные команды не исполняются.

```bash
python3 tools/m31/bench_m31.py --out /mnt/ai/logs/m31-port/bench-summary-v1.json \
  summarize --speed /mnt/ai/logs/m31-port/bench-v1/speed.tsv \
  --quality /mnt/ai/logs/m31-port/bench-v1/m31-xolotl-v1.jsonl
```

Summary сохраняет обе исходные строки каждого плеча и медианы, отношения к
одно-карточному референсу: TG4K/16K/32K 23.33/22.63/21.92, PP4K/32K425/433.
Пропуски глубин/повторов и неполный quality отражаются `complete*=false`;
плохие timings, не256 токенов, ранний EOS и повторённые quality IDs отклоняются.
Сравнение PP16K отсутствует, потому что его нет в итоговой таблице референса.
Качество референса96/100 и19/23; один стохастический прогон не доказывает
статистически значимое изменение. M2 приёмку helper не назначает.

Для отчёта рядом сохранить: commit/dirty diff и конфиг сборки, аictl preset,
backend flags (включая nofusion), arena/cache budget, KV/context/chunk/threads,
server log и peak VRAM/RSS. Если реализация использует F16 KV, а референс q8_0,
это прямо отметить: F16 увеличивает память и уменьшает экспертный кеш.
M1 nofusion/op_offload=false oracle — эталон logits, не speed baseline.
