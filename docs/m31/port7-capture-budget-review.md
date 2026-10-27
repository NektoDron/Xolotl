# Независимая проверка capture budget/window в V7

Проверены capture fields/function/getter/reset/window, cold worker lifecycle,
DSpark session и retained-draft standalone flow. Это readonly review; новых
тестов и изменений Engine по его результатам не сделано.

CaptureHost на M31 держит scratch 4096×6144×4 = 96 MiB и token-major backing
4096×30720×4 = 480 MiB в host RAM. CUDA scratch выделяется из backend host
buffer type; backend default CUDA device allocator для него не используется.
Эти 576 MiB не следует прибавлять к VRAM planner. Возможные небольшие driver
метаданные pinned allocation не равны размеру pinned host region. При cfg capture
на init выделение предшествует resident free snapshot; при runtime setter
остаётся обычный запас planner. Физическое выделение/first touch происходит до
PP timer и повторяется только при изменении geometry.

Capture graph появляется только на n1 decode_chain: input embd×F32,
residual additions и cont; source shared/residual/output заимствуются из dense
owner. Для пяти выбранных слоёв это несколько embd-sized F32 buffers на граф,
порядок сотен KiB суммарно, а не residual matrix в сотни MiB. n>1 prefill_chain
читает существующий device hidden непосредственно и не создаёт эти графы.
Каждый reset/rewind/release/shape switch освобождает capture graph перед dense
owner. Поэтому n1 capture graphs не сосуществуют с последующим cold PP и
не требуют нового persistent-VRAM члена planner. Точный allocator peak здесь
не измерялся.

При 32K/64K окно [max(0,P−4096),P) включает последнюю prompt row P−1.
Последний chunk сохраняет min(chunk_rows,4096,rows_after_min) строк; его first
равен end−rows. Полностью ранний chunk возвращает rows0/first=end/empty values;
observer оставляет предыдущие segments. Последующие непустые сегменты идут
непрерывно и завершаются в P; session перед init draft проверяет это и выставляет
logical_fed=P. Таким образом нужный residual anchor context не теряется на
границе chunk или при хвосте17 после4096. Ограничение features — только host
capture, KV/index state и planner key geometry оно не меняет.

Standalone с retained drafter после reset сообщает Engine актуальный
resident_bytes() перед новым PP: weights, ring и retained append/draft gallocr
суммируются. Engine вычитает external resident bytes из resident free snapshot;
полный KV/index для выбранного context уже учтён при этом snapshot. Рост draft
scratch в decode попадёт в resident_bytes следующего cold PP.

HTTP helper evict-ит drafter до cold PP и выставляет external budget0. После
lazy init target больше не вызывает prefill planner до finish/eviction; поэтому
HTTP не требуется удерживать число2.1GiB в external budget во время decode.
В текущем helper это число после init не устанавливается. Это безопасно для
существующего worker flow; при будущем добавлении PP с живым draft caller обязан
сначала обновить external resident budget. Verify выделяет свою private state при уже resident draft, не получает
ложного PP credit. Его отдельный peak этим readonly review не измеряется.

Prepared finish откатывает physical speculative position к logical fed boundary,
release снимает metadata getter; bounded VIEW lease сохраняет host backing
и epoch только до blocking lazy append, после чего явно отпускается до verify.
Затем drafter удаляется при finish. Если stop/EOS/cancel
произошёл до lazy prepare, speculative rows ещё нет: Engine сохраняет только
достоверные prompt features. Session уничтожает seed, следующий запрос делает
cold reset+setter. Rewind/reset/release снимают logical host validity; verify
явно invalidate_capture перед собственным legacy vector writer. При exception
worker уничтожает session и reset-ит Engine. Старые future/host capacity rows
не экспортируются getter.

Новой реальной ошибки budget/window/lifetime при этом review не найдено.
V9 GPU split byte proof завершён (см. port7-capture-view.md): полный
accumulated window и after-release lease bitexact. Paired model PP32K и
actual HTTP VIEW cancel/next остались отложены; их прохождение не заявляется.
