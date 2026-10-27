# Port7: дамп реального Engine.verify

`tools/m31/port7_verify_dump.cpp` — отдельный инструмент приёмки. Он использует тот же Config/ENV, что передал runner; запускается на CUDA только при `M31_CARD_SESSION=1`. Сам карту не занимает и профиль не меняет. CMake target и сборка ожидают координатора.

Вход — исходные prompt `.i32`, обычно 20 случаев длиной64–3000. Основные параметры совпадают с `m31_generate`; добавлены `--generate 128` (по умолчанию128) и `--require-bitwise 1` (по умолчанию1). Для приёмки: chunk2048, context4096, head1. Head/rows/decode_tail инструмент устанавливает под этот контракт самостоятельно. Захват residual features выполняется для отводов4/18/32/46/60.

Сначала ordinary forward генерирует128 жадных токенов, каждый вводится в Engine и сохраняет исходный F32 row после него. После reset prompt воспроизводится теми же фактическими prefill chunk, затем инструмент вызывает реальный `Engine.verify`. Чередуются шесть пар `(K,commit)`: `(1,one),(2,all),(3,half),(4,one),(7,all),(8,half)`; последний вызов ограничивается остатком. Это шесть пар, не полное декартово произведение. Logits и features копируются до rewind; сохраняются только committed строки. Все проверенные строки дополнительно сравниваются с ordinary F32, включая speculative строки, которые позже отбрасываются.

`run::Dump` содержит `engine=xolotl-m31-verify-tail-gate`, combined IDs `prompt +128`,129строк logits на позициях`P−1..P127`. Anchor`P−1` предсказывает первый выданный токен; row`P+i` — следующий токен после введённого `t_i`. Последняя строка предсказывает невыданный129-й токен. Порядок совпадает с oracle teacher forcing `--rows last:129 --head-tail 1`. `actual_chunks` оставлен пустым, поскольку перекрывающиеся verify вызовы не образуют последовательный prefill.

Отдельный `verify-metadata.json` содержит фактические prefill boundaries, verify/commit/rewind calls, формы, пути `serial-logits.f32`, `serial-features.f32`, `verified-features.f32` (все129строк, anchor включён), побитовые F32/неfinite/maxabs/top1 счётчики. `nk` явно помечен padding-derived: публичный API не отдаёт runtime nk. Python runner сверяет его с runtime evidence и сравнивает только одинаковую prefill геометрию; ordinary tailN1 и verifyK намеренно различны. Старый bounded-chunk comparator для этой пары использовать нельзя.

При несовпадении сначала завершаются полный dump/metadata, затем default strict mode возвращает2; исключение возвращает1. `--require-bitwise 0` оставляет оценку допуска внешнему валидатору, не объявляет mismatch успешной приёмкой.

Это проверка корректности teacher forcing/частичного commit/rewind, не замер скорости и не проверка draft-sidecar acceptance. Отбрасываемые токены сейчас правильные ordinary continuation; математическое покрытие ошибочной future branch требует отдельного случая. Независимый controller fixture покрывает такую ветвь на игрушечной модели. Качественные GSM8K/tools и actual DSpark generation проверяются отдельно на финальных ELF.
