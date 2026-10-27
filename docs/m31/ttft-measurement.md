# Разбивка TTFT M31

`tools/m31/ttft_session.py` запускается координатором через собственный
`vision_card_launch.py` job с `measurement=true`: GPU0 под card lock, весь
замер под exclusive `/mnt/ai/logs/m31-port/.build.lock`. Скрипт сам карту
не резервирует, aictl/config не меняет. Его CPU-only проверки:
`python3 tools/m31/ttft_test_session.py`.

Пример команды внутри уже разрешённого card worker:

```bash
python3 tools/m31/ttft_session.py --output /mnt/ai/logs/m31-vision/ttft-baseline-v1 \
  --engine /path/to/instrumented/m31_serve --adapter /path/to/m31_vision_adapter \
  --env M31_GPU_PREFETCH_HISTORY=0 --env M31_PROMPT_CACHE_MIB=0 --raw-probe
```

Для кандидата задаются те же inputs/profile и другие явные ENV. Нельзя
перезаписывать уже использованный output directory. `--repeat` задаёт число
полных cache-hit повторов и текстовых запросов; default2.

Baseline и history-only arm явно отправляют `cache_prompt=false`. Для
отдельного whole-prompt KV arm нужны **оба**
`--env M31_PROMPT_CACHE_MIB=1024` и `--allow-prompt-cache`; тогда обычные
запросы отправляют `cache_prompt=true`. После точных повторов каждой медиа
добавляется тот же prompt с `cache_prompt=false`: он обязан вычислить один
полный prefill, не иметь cache hit и дать те же content/reasoning, stop reason
и число generated tokens, что непосредственно предшествовавший подтверждённый
exact hit. Ключ prompt payload отдельно хешируется без управляющего cache flag.
Новый вопрос обязан оставаться miss даже в KV arm.

Текстовый baseline строится через принятый PR vocab helper: естественный
corpus excerpt в user message плюс просьба ответить `OK`. Длина **всего**
отрендеренного chat prompt доводится до515 и3863 token IDs. После запроса
server `prompt_n` обязан подтвердить эту длину. Это контроль длины и затрат
engine, а не семантически одинаковая с картинкой задача. Опциональный raw
probe отправляет ровно те же515 IDs через `/v1/completions`, чтобы отдельно
увидеть стоимость chat template/parser.

PNG и видео используют те же prompts, порядок text/media, temperature0,
effortlow, max1200 и1fps, что сохранённый reference client. Default sequence:

1. Два text515 запроса: первый native warm-up и повтор.
2. PNG с холодным encoder worker.
3. **Тот же body** PNG с уже загруженным worker, но без full-prompt artifact.
4. PNG с новым вопросом той же token length: возможный KV cache miss.
5. Два точных PNG повтора с full-prompt artifact.
6. Аналогичные video запросы; worker к этому моменту прогрет PNG.
7. Два text3863 запроса и optional raw515 probe.

`--first-media video` меняет порядок медиа для отдельного холодного video
worker замера. Нативный LLM по обычному контракту сервера сбрасывает request
state. Если включён opt-in whole-prompt KV cache, точные повторы могут быть
его попаданиями; это отдельно отмечается в ENV/trace и не выдаётся за
ускорение нового промпта.

По умолчанию diagnostic session задаёт `embedding_cache_mb=0`: сохранены
загруженные веса/scheduler encoder, но повторное encode действительно
выполняется. Завершённый собственный full-batch entry переносится в
`cache-archive/` между cold и warm-worker запросами; sources и модельные
файлы не меняются и ничего не удаляется. `--embedding-cache-mb 256` воспроизводит
обычный memory cache и требует учитывать его фактические hits. Архивные
manifest сохраняют исторический `source_manifest`; файлы массивов находятся
рядом с архивным manifest и не зависят от этого информационного пути.

Каждому запросу присваивается `X-M31-TTFT-ID`; опциональная серверная
инструментация пишет `M31_TTFT_TRACE` JSONL. `results.json` сохраняется после
каждого запроса, включая ошибку. Результат требует SSE `[DONE]`, первого
generated delta, правильной длины prompt и совпавшего trace ID.

`breakdown.critical_path_ms` — непересекающиеся интервалы по milestones:
HTTP/body/JSON → prepare/queue → native RPC → первый native token → первый
generated delta. Остаток client/server start offset и транспорта учитывается
отдельно. Вложенные media, encoder и native leaf timers объясняют интервалы,
но **не добавляются** к их родительским суммам повторно. Число
`native_unattributed_before_first_token_ms` показывает расхождение полного
native TTFT с суммой явно измеренных native leaf stages.

Первый generated token получает logits из последней строки prefill; обычный
однотокенный decode forward идёт **после** его отправки. Поэтому его timer
не включается ещё раз в TTFT. Если parser удержал первый native control token,
этот decode может лежать в уже измеренном интервале native token → первый
generated delta. Reasoning до первого видимого content вынесен
в отдельный интервал. Полные responses/events и native timings сохранены,
чтобы различать compute, скрытые control tokens и отложенный parse.

Документ описывает методику и не утверждает результаты GPU-замеров. Итоговые
числа и ограничения публикуются в `REPORT-ttft.md` и `.final.md` координатора.

`--prewarm-encoder` передаёт opt-in `prewarm=true`: worker загружается после
основного LLM до HTTP readiness. Затраты входят в `server_startup_s`; сама
первичная медиа помечена configured-prewarm, а фактическое состояние берётся
из trace `details.encoder.worker_started`. Это перенос cold init в startup,
а не исчезновение работы.

Старые сохранённые прогоны можно разобрать без перезаписи исходных файлов:

```bash
python3 tools/m31/ttft_summarize.py /mnt/ai/logs/m31-vision/ttft/baseline-v1 \
  --output /mnt/ai/logs/m31-vision/ttft/baseline-v1/breakdown.json \
  --markdown /mnt/ai/logs/m31-vision/ttft/baseline-v1/breakdown.md
```

Выходные файлы должны быть новыми. JSON сохраняет SHA исходного results,
флаг завершения session, сырые вложенные stages и дизъюнктный breakdown
каждого запроса; Markdown даёт краткую сравнительную таблицу.
