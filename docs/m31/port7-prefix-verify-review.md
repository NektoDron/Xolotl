# Независимый readonly review Engine::verify (prefix agent)

По прочитанному коду ошибок causal KV/indexer masking или rollback, способных
нарушить правильный accepted prefix, не найдено. Это source audit, а не
доказательство численного совпадения GPU. Нужна actual serial/verify приёмка.
Production assumptions: finite KV/indexK, DISABLE_FUSION1, неизменённый
однокарточный backend и n1 sparse gather.

Для каждой позиции `first+row` verifier строит собственные `positions`, mask,
bias и `nk=PAD(first+row+1,max(256,block))`; attention вызывается сn1 и
indexed KV write. Все SET_ROWS стоят в явном graph node order до packed
outputs. Ранний query может физически читать уже записанный futureKV внутри
своего padded bucket, но causal -inf прибавляется **до block max**, а выбранные
блоки снова получают per-cell mask при sparse FA gather. Future blocks с-inf
могут попасть вtop_k ties, однако их все клетки остаются masked. Несовпадение
физического содержимого таких finite future rows не меняет block score.

Контрпример, который искался: `first=255,n=3`. Получаются nk256/512/512.
Query255 не видит строк256/257 даже как часть view; query256 видит строк257
только под-inf. При `first=127,n=3` все nk256, но local block bias переходит
с block0 наblock1 отдельно на каждой позиции. Геометрия совпадает с serial,
а не с одним batch n3, который имел бы nk512 для всех query. В графе нет
заимствованных temporary rows от предыдущего chunk.

Driver коммитит `matched+1` строк: anchor плюс принятые кандидаты. Следующий
corrective token берётся из logits последней committed строки. Draft ring
получает только feature prefix; затем main Engine rewind исключает первый
rejected KV row. Future stale rows остаются, но следующий forward/verify
ограничивает view новым nk и маскирует будущие позиции. Rewind очищает
captured rows/features и head/hidden/dense/shared captures; driver заранее
скопировал features, поэтому после rewind их чтение из Engine не требуется.

Graph::compute синхронизирует backend, Graph::read возвращает собственный
CPU vector, CPU graph compute также blocking. Per-layer Graph уничтожается
после чтения norm/shared/residual/IDs/weights и CPU FFN. Head graph читается
до выхода из scope. На успешном пути dangling graph-owned tensor pointers
не остаются. После входной проверки verifier ставит poison=true до изменения
KV/owners; исключение не выдаёт частично вычисленный pass за валидный.

Ограничения API, которые следует учитывать:
- Возвращённый `verify(..., logits, ...)` рассчитан per-row n1 head. Если после
  него вызвать обычный `Engine::logits` с несколькими rows, тот снова построит
  обычный batch head и не обязан дать тот же последний бит. Нынешний DSpark
  driver правильно использует уже возвращённый vector.
- `verify` захватывает все n feature rows, даже если Config feature_max_rows<n;
  сейчас default4096 иn<=8, поэтому driver не затронут. Если этот API станет
  общим, контракт лимита следует явно определить.
- Finite stale rows безопасны; NaN/Inf poisoning не изолируется маской, поскольку
  `NaN + -inf` остаётсяNaN. Это существующее свойство reset/causal masking,
  а не новое обещание, что можно сохранять произвольные невалидные KV bytes.
- Source order SET_ROWS→reads проверен. Не следует включать новый fusion или
  отдельные concurrent streams для этих aliasing cache writes без нового proof.

Предлагаемые реальные GPU regressions: prompt lengths255/256/257 и
4095/4096/4097 сdepth7; после полной verification8 откат кfirst+1 иfirst+4,
затем другая ветка токенов и сравнение raw rows сcold serial. Для длинных
контекстов —16383 и32767. Нужны хотя бы один early reject, accept-all и
p_min→нулевой draft (anchor-only). Проверка лишь top1 первых128 позиций не
закрывает crossing256, stale futureKV и long-context geometry.
