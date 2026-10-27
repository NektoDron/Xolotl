# Port6: изолированный prototype общего M31 serve

Новые файлы находятся в `tools/m31/serve/`. Рабочий standalone adapter `tools/m31/serve_m31.py`, общий `serve/server.py` и C++ engine в этой волне не менялись. Это opt-in prototype следующего этапа, не конфигурация замеров PP/TG и quality port6.

## Реализованная граница

`engine_adapter.py::M31Engine.generate(ids,max_new,sampling,cancel,embeddings=None)` реализует Python Engine interface общего сервера непосредственно поверх существующего JSON-lines subprocess. JSON не подаётся в `StrataEngine`, ожидающий GEN/GENI. Общие HTTP/SSE и OpenAI/Anthropic response builders берутся из `serve/server.py`.

`PreparedIds` содержит IDs и неизменяемую копию request metadata. На этапе prepare рендерит один vocab-only PR helper. Перед генерацией второй helper повторно рендерит request под inference FIFO и проверяет точное равенство IDs. Поэтому подготовка следующего HTTP-запроса не сбрасывает активную grammar/PEG. Piece/tokenize/detokenize используют render helper; grammar и output parser используют generation helper.

Controller владеет sample/grammar и token/continue ACK; при отмене или grammar failure отправляет отрицательное подтверждение, потребляет terminal event и только потом освобождает канал. Повреждённый транспорт или неудачный drain инвалидирует RPC. Bounded read/write transport переиспользован из исправленного standalone adapter. Тихое ожидание выдаёт heartbeat; отмена во время prefill ограничена доступной границей события движка.

В общем Service `parser.finish()` вызывается уже после освобождения FIFO. Поэтому финальный PEG parse вычисляется в controller до unlock и сохраняется в parser; finish не обращается к потенциально уже сброшенному helper. Thread-local parser context отделяет ожидающие запросы. Ветка stop string → cancel → negative grammar ACK → terminal error тоже финализирует parser.

Native M31 потребляет EOG внутри subprocess. На успешном stop controller выдаёт common Service представительский EOG ID, чтобы finish/usage считали последний sample так же, как native engine. Timings переименованы в поля общего server, reused всегда0. Sampling top_k передаётся без ограничения64; top_k0 и temperature0 сохраняются. Неподдержанные repetition_penalty/repeat_penalty !=1 и любые явно заданные penalty_last_n/repeat_last_n отклоняются до RPC. PR additional_stops добавляются к stop matching общего сервера. Token-count endpoint использует PR chat IDs, а не Qwen literal-tag encoding.

## Запуск

Config с явными argv:

```json
{
  "engine_argv": ["/path/m31_serve", "--model", "/path/model.gguf", "--backend", "cuda"],
  "tokenizer_argv": ["/path/m31_tokenizer", "/path/model.gguf"],
  "model_name": "m31-xolotl-common"
}
```

Либо без дублирования принятого runtime recipe:

```json
{
  "runtime_profile": "/checkout/tools/m31/profiles/native-gpu.json",
  "engine": "/path/m31_serve",
  "model": "/path/model.gguf",
  "tokenizer_argv": ["/path/m31_tokenizer", "/path/model.gguf"],
  "model_name": "m31-xolotl-common"
}
```

`runtime_profile` может быть относительным к config. Resolver не запускает процесс; CLI применяет unset_environment, затем environment до старта helper/engine. Model/executable в профиле не зашиты. Смешивание engine_argv с runtime_profile отклоняется. Требования к сборке из profile README остаются обязательными.

Команда: `python3 tools/m31/serve/serve_m31.py --config CONFIG --port 8181`. Bind только loopback. CUDA требует `M31_CARD_SESSION=1` и существующую aictl/card-lock сессию; shim не занимает карту самостоятельно. Это не production preset.

## Явные ограничения

Vision/GENI, raw continuation IDs, prefix cache, disk sessions, VRAM resize, forced Qwen tool openings, reasoning-budget continuation и structured grammar requests не подключены. Они отклоняются вместо имитации функциональности. Prototype не делает автоматический restart умершего engine: требуется явный перезапуск процесса. Установленная в `make_service` фабрика parser process-local, как model shim GLM; запускать разные model shims внутри одного Python процесса не предполагается.

HelperTemplate поддерживает generation-prompt render; history-only/assistant-prefill renderer PR helper пока не экспортирует. Streaming выдаёт reasoning/content постепенно, завершённые tool calls — в финале, как принятый standalone adapter.

Прежде чем менять рабочий сервер, нужны native-engine HTTP gate, tools23/GSM и измерение overhead двух helper/re-render. CPU prototype не является такой приёмкой.

## Проверки

`test_common.py` проверяет общий HTTP OpenAI/Anthropic в stream/nonstream, настоящий JSON subprocess на двух последовательных запросах с ранним close, grammar failure/cancel/drain и reuse, отсутствие top_k64 cap, native EOS, ID replay/mismatch, immutable metadata, render-helper isolation, финальный parse до unlock, PR additional stop+grammar cancellation, portable profile resolver и Anthropic auto/parallel choice. Отдельные CPU vocab-only integration cases проверяют реальные IDs и финальный PEG output для Unicode/reasoning и nested tool XML против прямого pinned PR helper.

Все **19/19 CPU тестов PASS**, без пропусков, 0.934 с; `py_compile` PASS. Финальный лог и source receipt: `/mnt/ai/logs/m31-port/port6r2-common-shim-check-v4.log` и `.receipt.json`. Ни модель на GPU, ни GPU performance в этой волне не запускались.

Независимый аудит integration: EOS считается один раз, последовательные запросы не переносят last/timings, все penalty aliases отклоняются до RPC. Receipt: `/mnt/ai/logs/m31-port/port6-profile-common-audit-v2.json`.
