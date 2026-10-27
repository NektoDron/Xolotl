# Port7: точный кеш полного prompt, opt-in

Принятая v8 остаётся без кеша. Кандидат включается отдельным флагом
`serve_m31.py --prefix-cache`, который передаёт worker `--prefix-cache 1`.
`cache_prompt=false` проходит HTTP → JSON worker и всегда требует cold reset.
По умолчанию новый флаг к argv старой v8 не добавляется: adapter сохраняет
совместимость с её frozen ELF.

Первая версия кеширует только **полный одинаковый text prompt, вычисленный
одним actual prefill chunk**. Worker хранит исходные token IDs, рецепт
`(position,n,padded_keys)` и неизменяемую raw logits row до sampling.
Перед reuse сверяет весь prompt, наличие исходного KV prefix и результат
чистого `prefill_chunk_at(0,requested)` с записанным n. Sampling parameters
и seed применяются заново; sampled token и отфильтрованные logits не кешируются.

Один chunk позволяет проверить совпадение с очередным fresh проходом: planner
читает тот же текущий бюджет удерживаемых buffers, а до единственного chunk
нет промежуточных пропущенных allocations. Multi-chunk и частичный LCP
отключены: для них надо отдельно доказать последовательность изменения
allocator high-water, а не округлять LCP к256. Другой prompt, max_tokens0 без
вычисленной головы, ошибка и остановка через interactive ACK очищают кеш.

Если `draft_feature_layers` не пуст, reuse запрещён: первая версия не хранит
DSpark feature ring. Кеш KV/head без необходимых draft features не применяется.
Сервер принимает только text: embeddings/vision в этот механизм не входят.

Общие новые Engine API: `rewind(position,error)` и
`prefill_chunk_at(position,requested,error)`. Первый сохраняет prefix main KV
и indexK, очищает hidden/head/dense/shared graphs, captured draft features
и active chain flags. Будущие stale finite rows маскируются причинно, как
после штатного reset. Второй не меняет position и читает текущий бюджет
удерживаемых buffers; он не обещает неизменность budgets после будущих
allocations. Изначальный `prefill_chunk` не менялся.

HTTP ответ с кешем содержит `usage.prompt_tokens_details.cached_tokens`,
`timings.cached_n`, `timings.prompt_eval_n` и `timings.cache_hit`.
Вычисленный PP использует только реально вычисленные токены. При полном reuse
вычислений0 и PP0; logical prompt count остаётся полным. Default OFF сохраняет
прежние поля ответа. Внутренний worker дополнительно сообщает actual recipe,
KV position и количество действительно поданных generated tokens; последний
sample без следующего forward не становится KV row.

## CPU проверки

- Helper:15/15 PASS, включая независимость raw sampling copy и отказ от
  непроверенной multi-chunk геометрии.
- Actual tiny-model worker protocol:21/21 PASS. Проверены default OFF,
  одинаковый prompt/greedy output, `cache_prompt=false`, max_tokens0,
  другой prompt, ошибка/отмена, отказ от partial/multi-chunk и fed KV count.
- HTTP/API protocol:4/4 PASS, включая cached usage и вычисленный PP.
- Existing server suite:32 tests,24 PASS и8 real-vocab SKIP без fixture env;
  повторён после изменения counters.

Receipt: `/mnt/ai/logs/m31-port/port7-prefix-cpu-check-v4.json`.
Это не приёмка качества production модели.

Первый build прошёл helper, затем link failed: Engine уже требовал новые
symbols, а tooling линковал старый CPU archive. Исправлено компиляцией current
GpuExperts вместе с Engine. Вторая сборка compile/link прошла, но fixture
указала ctx64 при model maxctx32 и верно была отклонена. После исправления
fixturectx32 runtimev3 прошёл. Эти вспомогательные ошибки не являются
отрицательными model numerics.

CPU ELF был собран до последней правки cold timer и fail-closed DSpark guard.
Основной cached алгоритм тот же, но финальный combined ELF требует повторения.

## GPU приёмка ограниченного exact-cache пути

Job `codex-m31-port7-prefix-gpu-v1` завершён rc0, обе команды — rc0.
На immutable combined V2, context 40000, cache 7GiB, CPU80 и тех же 48 ENV:

- Raw proof: 9 уникальных позиций, сохранённых в 3 прогонах cold/reuse/cold replay.
  Prompt head и восемь decode rows совпали побайтово, maxabs 0. Каждый файл
  `[9,200064]` занимает 7202304B; SHA256 всех трёх:
  `4ff45b0d2348b98330cffb71fa4c108f91bcc436fb4384770279c83d7a45af8d`.
- Actual native worker smoke: 16/16 PASS. Repeat читает 0 новых prompt tokens
  и отмечает cached4352; `cache_prompt=false` действительно вычисляет 4352.
  Следующий request после cache_false тоже cold, затем кеш снова заполняется.
  Отмена после одного sample оставляет 0 fed rows и очищает metadata.
- Independent audit: 29/29 PASS. Проверены ELF/profile/ENV, весь recipe,
  одинаковые 8 greedy tokens всех полных ответов, 7fed rows и KV4359; после cancel
  KV4352. Дополнительный environment — только CARD_SESSION1.

Input4352 = 17×256; первый decode переходит к nk4608, а rewind оставляет stale
future slots4353…4359 в том же бакете. Raw proof проверяет значения логитов после
такого возврата, а не только равенство коротких строк ответа.

Receipts в `/mnt/ai/logs/m31-port/port7-prefix-gpu-v1/`:
`raw/result.json`, `raw/independent-byte-audit.json`, `native-worker.json`,
`independent-audit.json`, `port7-prefix-gpu-v1-card-job.result.json`.
Manifest содержит pre-launch frozen plan; фактический result хранится отдельно.

Это cache parity с тем же нативным путём, **не новое oracle-сравнение, не full quality
и не замер скорости**. В correctness run повторный prompt занимал 10.3ms против
3.1–3.4s cold; эти наблюдения не подменяют выделенный performance benchmark.
Длинные multi-chunk prompts и частичный LCP по-прежнему не используют reuse.
GSM97/100 + tools19/23 и скорость принятой v8 не выдаются за приёмку этого кандидата.

Новые файлы: `server_prefix_cache.hpp`, worker integration patch
`port7-prefix-worker.patch`, CPU helper/protocol/API checks. Root применил
worker patch к production source, но default остаётся OFF. Исследовательские
GPU proof tools проверяют только указанную область и не подменяют финальные
quality/performance gates.
