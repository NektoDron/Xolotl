# M31: измерение TTFT по стадиям

Диагностика включается только переменной `M31_TTFT_TRACE=/abs/trace.jsonl` в
описании собственного aictl job. Клиент передаёт `X-M31-TTFT-ID`; это значение
попадает в `id` единственной JSONL-записи запроса и не меняет OpenAI wire.
Без переменной native request не содержит `ttft_trace`, дополнительные timing
поля в native/HTTP ответах отсутствуют. Текст промпта и байты медиа не пишутся.

`points_ms` использует монотонный `perf_counter_ns` и отсчитывается от входа
`Handler.do_POST`, после разбора HTTP заголовков. Клиент отдельно измеряет весь
запрос, включая соединение, передачу тела и получение первого SSE delta.
Разность client TTFT и server `first_generated_delta` покрывает транспорт и
предшествующий входу в POST разбор заголовков; это не отдельный GPU интервал.

- `http_body_parsed`: тело прочитано и JSON разобран.
- `prepared`: media ingestion, шаблон и encoder закончены, mixed manifest готов.
- `native_rpc_start`: native request отправляется в pipe.
- `native_prefill_received`: native закончил prefill/head/первый sample.
- `first_native_token`: Python получил первый токен до detokenize/PEG.
- `first_generated_delta`: первый content или reasoning SSE delta.
- `first_content_delta`: первый видимый content SSE delta.
- `native_done`, `request_finished`: завершение; обычный reasoning до content
  измеряется разностью first_content_delta − first_generated_delta.

`stages_ms` содержит HTTP body/JSON, ожидание inference lock, media prepare,
chat-template RPC, encoder total и prepare total. Это вложенные интервалы:
`prepare_total_ms` уже включает media/template/encoder, их нельзя суммировать
повторно с родителем. `details.media` разделяет base64, source read/store,
PIL verify и ffprobe. Настоящий decode картинки выполняется stb в mtmd;
PIL здесь лишь проверяет допустимость/размер загруженного изображения.

`details.encoder.stage_ms` разделяет выбор backend, hash медиа, cache lock,
cache lookup/validation, startup worker RPC, request IPC+compute, сборку F32
файлов и публикацию manifest. `worker_started`, `cache_hit` и backend явно
различают холодный процесс, прогретый encoder с новой работой и готовый batch.
`startup_stages_ms` отдельно сообщает vocabulary/backend init и загрузку
encoder context. Disk cache hit не запускает worker и не выполняет encode.

`adapter_timings_ms` содержит wall интервалы media read/decode, mtmd tokenize
с preprocessing/video frame decode, сериализацию chunk metadata, cache copy,
ленивую загрузку дополнительных CPU contexts и encode с файлами/cache.
Публичный mtmd API объединяет ffmpeg frame decode и preprocessing внутри
`mtmd_tokenize`; эти два подпроцесса здесь не выдаются за независимо измеренные.
Compute, F32 write, validation/cache-copy также даны как `*_thread_sum_ms`:
для одного CUDA worker это последовательные интервалы, для нескольких CPU
workers сумма потоков может превышать wall time и не прибавляется повторно.
Последний manifest write и полный process wall включены в ответ encoder RPC.

`details.native_timings.ttft_stages_ms` и тот же блок SSE done.timings:
JSON parse, manifest read/validation, request setup, reset, prefill planning,
все `forward` prefill, output head, первый sample (включая grammar handshake),
первый decode forward/head. Первый sample относится к prompt_ms, а первый
следующий decode не входит в TTFT. `actual_chunks` хранит реально выполненные
позиции/размеры и forward wall; для media prompt это позволяет подтвердить
единственный полный prefill. Интервалы native — wall, не CUDA event kernel time.

CPU regression: `python3 tools/m31/test_ttft_trace.py` проверяет настоящий HTTP
SSE, correlation ID, native→reasoning→content порядок и отсутствие native flag
или trace wire при выключенной диагностике. Численные гейты и замеры скорости
проводит координатор в своих aictl/exclusive окнах.

## Загрузка encoder до готовности HTTP

Опциональное поле `prewarm: true` в vision-config обоих серверов вызывает
`VisionEncoder.start()` после ready основного Engine, до создания HTTP server.
В auto-режиме VRAM probe поэтому учитывает уже загруженный LLM. False/отсутствие
оставляет прежний ленивый запуск. Поле строго boolean и удаляется до передачи
параметров конструктору encoder; математику и cache identity оно не меняет.
`start()` на работающем worker не перезапускает его. Загрузка переносится в
server startup, отдельно пишется `M31 vision prewarm` со временем/backend и
vocab/context startup stages в server.log. Первый media-запрос всё ещё делает
новые preprocessing/encode и LLM prefill, если нет соответствующих cache hits.

`test_vision_prewarm.py` проверяет порядок EngineReady→WorkerStart→HTTPReady,
отсутствие eager worker при false, отклонение не-boolean до любого RPC и
сохранение уже живого worker при повторном start. Это CPU mocks, не GPU замер.
