# Port7: независимое ревью HTTP M3 controller

Read-only, 2026-10-27. Проверены live `m31_dspark_session.hpp`, `m31_serve.cpp`, `server_dspark_policy.hpp`, `server_sampling.hpp`, `serve/engine_adapter.py`. Ошибок native queue/stop/fed-position и порядка seed sampling при условии корректных target logits не найдено. Это ревью управления, не доказательство численной эквивалентности нового CUDA verifier, не GPU smoke и не качество модели.

## Native worker / session

`choose()` остаётся в worker: token0 после prefill, далее ровно один вызов на каждой фактической итерации. Session не вызывает sampler, не использует `std::mt19937_64` worker и не сэмплирует наперёд. Temperature>0 и grammar переключают request на ordinary target, сохраняя тот же sampler/RNG flow. Greedy `sample()` проверяет все logits на finite и выбирает первый/меньший ID при равных максимумах через строгий `>`. Draft tie-breaking не задаёт target sample: cached row принимается лишь когда очередной уже sampled token равен draft input. Численное совпадение target rows всё равно требует отдельного model proof.

Пусть prompt имеет P токенов. При первом `advance(t0)` буфер получает `[t0,draft...]`, `buffer_first=P`, target физически исполняет K входов. Возвращается row0 — logits ПОСЛЕ t0, затем `consumed=1`, `logical_fed=P+1`. На совпадении t1 возвращается row1, logical_fed увеличивается на1. На несовпадении `flush_features()` дописывает в draft ring только consumed features, target rewinds до logical_fed, затем новый verify начинается с реального sampled token. Неиспользованные speculative features в ring не попадают. При исчерпании буфера аналогично flush всего consumed span, без лишнего replay уже fed токенов.

`finish()` отбрасывает speculative suffix до logical_fed, освобождает verify workspace и draft, очищает очередь. Worker проверяет `position=P+max(0,predicted−1)`. Это правильно и для EOS: predicted включает sampled EOS, который не отправляется и не вводится; последний предшествующий токен уже fed. Для max_tokens M без EOS fed только M−1; для cancel сразу после emitted token i он ещё не fed, значит fed i предыдущих. Для maximum0 position=P. Cancel/EOS до первого advance и max_tokens0/1 не загружают sidecar; policy дополнительно отключает M3 для max≤1.

Перед первым advance draft загружается лениво после token0 и ACK, сохранённый seed ending position сверяется с target.position. Сегменты prefill подрезаются до последних4096 строк с правильным `skip*width`; discontinuity начинает новый непрерывный сегмент. Append принимает ненулевой first position и записывает абсолютные позиции ring. Захваты4/18/32/46/60 соответствуют entering residuals, подтверждённым прежним math review. При exception worker очищает prefix metadata, reset target и возвращает error; локальный session RAII уничтожает draft до следующего request.

При configured M3 policy fallback cold request включает пустой список capture layers; M3 requests не используют prefix cache, потому что cache не хранит draft feature ring. При смене policy накопленный prefix cache очищается. Для M3 всегда требуется persistent CPU pool и expert threshold>8, чтобы K≤8 не ушёл в обычный routed GPU prefill path.

## Standalone HTTP — текущий выбранный сервер

Владелец prefix подтвердил, что финальный M3 HTTP запускает `tools/m31/serve_m31.py`, а common adapter пока отдельный opt-in prototype. Дополнительно прочитан `Application.events`: incremental decoder и `clipped_text` удерживают неполный stop suffix; stop decision выполняется ДО отправки continue ACK (`serve_m31.py:375–380`). При совпадении stop отправляется continue=false, поэтому обнаруженный stop token не fed. При обнаруженной отмене до ACK finally также отправляет continue=false. После ACK native forward может перекрываться с PEG; отмена, обнаруженная позднее, дренируется на следующем token boundary. Это ограничивает протокольный backlog, но не отменяет исполняемое CUDA ядро. Drain читает только до terminal done/error, ограничен30с/16кадрами; при неисправности канал invalidate, следующий request на нём не запускается. Ошибка grammar разблокирует sample handshake reject+emptyallowed, затем ожидается terminal error/reset; повторный grammar-helper при drain не вызывается.

`request_parameters` проверяет seed как uint64 либо -1/None для nondeterministic default, исключает bool и не подменяет явно переданный seed. Worker создаёт один mt19937 на request и использует прежний sampler. Top-k/p/min-p/temperature остаются на стороне worker, session их не применяет. Standalone не использует `M31Engine.last`, поэтому найденная ниже статистическая проблема common prototype не относится к выбранному HTTP launch path.

## Common Python adapter — отдельный prototype

Отмена ограничена drain/deadline, но не является отменой уже исполняемого verify. `engine_adapter.py:205–208` отправляет ACK до yield token, чтобы forward перекрывался с frontend PEG parsing. Если frontend затем обнаружил stop string или отменил запрос, worker может уже исполнять следующий advance; `_drain` дочитывает кадры и посылает continue=false, а finish убирает speculative tail. Ошибка/timeout drain инвалидирует RPC, такой framed channel не переиспользуется. Cold next request исключает продолжение старой ветки. Это существующая tradeoff overlap/cancellation, её нельзя описывать как отсутствие GPU работы после cancel.

Найдена небольшая проблема статистики: `M31Engine.generate()` не очищает `self.last` в начале request, а `_drain()` при terminal error (`engine_adapter.py:122–125`) только завершает parser и возвращается. Cancel во время grammar может вызвать deliberate reject с пустым allowed, worker отвечает error/reset. Тогда `last` остаётся от прошлого request вместо cancelled/current stats. Вывод токенов/Engine KV от этого не портится; следующий request транспортно безопасен. Передано владельцу prefix и root; код без ACK не менялся. Возможное исправление: обнулить request stats до write и явно записать cancel/error outcome в error drain, с fixture previous-success→grammar-cancel→next-success.

## Границы доказательств

Controller fixture уже проверяет causal queue, все/частичные/нулевые acceptance, p_min, suppression, cancel/max, длинный trimmed seed, lazy single load и смену future branch на независимой игрушечной модели. Новый `port7_verify_dump` будет проверять настоящий target/rollback/features на teacher-forced ordinary continuation, но его текущая future ветвь правильная. Neither proof подменяет actual sidecar final greedy parity/quality и native HTTP/SSE smoke.

Исходные SHA256 этого read-only review:

- `m31_dspark_session.hpp`: `2d4e877375593bc4f177cc0c6d23c483e7d656ce04b7e92c606535a9dd139f9b`.
- `m31_serve.cpp`: `6a29adea99a28c110c4307230f9bddc819f0f9e699de49edc3ed4b8431489e1e`.
- `server_dspark_policy.hpp`: `91c473ad0ed5679915f609eeb38575745c4b189d4299e854549a39dafc22cec4`.
- `server_sampling.hpp`: `a0e91d10bf1e70680e05f33db1cad9a0bc90144c94a8bf91f1d130cd3dc35764`.
- `serve/engine_adapter.py`: `151c6d979c375f345d0864e03a47b652b5962e33b5d0e020b6ae9245e0621824`.
- `serve_m31.py`: `9c96c0f077a559b941c0ae7ff4b021b81ca8810f54537d29b59295994d61f07e`.
