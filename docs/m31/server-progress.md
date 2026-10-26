# M31 server/eval — port4

Направление server_eval. Коммитов и GPU-прогонов агент не делал.
Системный aictl config не менялся.

- Добавлена настоящая lazy tool grammar из аудированного PR: `server_grammar.hpp`
  использует llama grammar sampler, `common_sampling_add_*` и reasoning-budget.
  Никакого regexp-парсера аргументов или подмены GBNF. Regex используется только
  для оригинальной семантики trigger patterns PR.
- Изолированный tokenizer/helper проверяет предложение токена. При отказе
  возвращает допустимые ID всей vocabulary; m31_serve пересэмплирует исходные
  logits тем же top-k/top-p/min-p/temperature на допустимом множестве.
  Дополнительный handshake работает только при наличии tool grammar.
- Авто-режим разрешает обычный ответ; grammar выключена во время thinking.
  Новый chat сбрасывает грамматику; disconnect/cancel дочитывает handshake и
  отменяет запрос на следующей token boundary, следующий запрос делает reset.
- Проброшен `--cpu-decode-cache` из CLI адаптера в native server; default false.
- `server_make_session.py` подготавливает временный `m31-xolotl` через имеющийся
  card_session, с уникальными stem, бэкапом и точной rollback командой.
  Настоящие argv om_bench и eval_duty сохраняются вместе с SHA входных скриптов.
  Исправлен default benchmark port8180 (8179 занят worker health).

Проверки:

- `codex-m31-port4-server-eval-build`: rc0; приватный `build-server-eval` собирает
  только vocab helper против готовых PR archives; server C++ syntax-only check.
  `/mnt/ai/logs/m31-port/server-build-eval.log`.
- `codex-m31-port4-server-grammar-tests`: rc0, **15/15**,0.603с, CPU vocab only.
  `/mnt/ai/logs/m31-port/server-grammar-tests.log`.
  Включены plain auto ответ, tool trigger внутри thinking, nested object/array с
  Unicode, два вызова, запрет неизвестной функции + возврат allowed IDs, reset,
  protocol resample и cancellation во время proposal.
- `py_compile` и `git diff --check` по файлам направления проходят.
- Подготовленные JSON:
  `/mnt/ai/logs/m31-port/server-eval-prepared-v2/manifest.json`,
  `m31-port4-server-v2-card-job.json`, `m31-port4-server-v2-http-session.json`.
  Это smoke на8192/F16/chunk128, ещё НЕ запускался.

Осталось: реальный HTTP gate нового grammar handshake, затем tools23/GSM8K100 и
полный speed suite. Время до23.33t/s и425..433PP не объявляется улучшенным.
Full suite требует VRAM gate на>=40000/F16 context; Q8 KV пока отсутствует.

## Второй этап: реальный tools HTTP и tools23

Подготовлены `server_tool_smoke.py` и `server_tools23.py`.
Native sampler считает `grammar_checks`/`grammar_rejections` и возвращает их в
`timings`, чтобы HTTP gate доказал использование C++ proposal/grammar handshake.
Smoke проверяет обычный auto-ответ391, настоящий get_weather(Казань,celsius),
SSE вызов, отключение клиента во время thinking и следующий успешный вызов.

`server_tools23.py` AST-загружает исходные определения eval_duty_1023.py до final
with, запускает только tools(), без GSM8K/div0 и изменения исходника. Вне сети
проверено ровно20+3 записей на stub chat; sourceSHA
bca423fc5b57609337aff1cff5d931a328d728fa8fe0a199f4fa7f0e1222d286.
Пресет-генератор принимает modes toolsmoke/tools23 и --dense-decode-cache.
Реальный запуск ещё ожидает свежую build-cpu-decode/m31_serve.

### Живой HTTP gate принят (19:14 UTC)

`codex-m31-port4-tools-http-v1` rc0: **38/38** checks,
`/mnt/ai/logs/m31-port/server-tools-http-v1/http-tool-smoke.json`.
Native worker: `build-cpu-decode/m31_serve` с CPU+dense decode caches,
CUDA dense / CPU routed experts, ctx8192/F16/chunk128/threads64.

- Auto plain:391, без вызова,11 настоящих C++ grammar checks.
- Nonstream и SSE: get_weather({"city":"Казань","unit":"celsius"}),49checks
  на каждый ответ; корректный tool_calls finish и ID.
- Отключение клиента во время thinking с активными tools; следующий вызов
  успешно выполняется после reset/recovery, ещё49checks.
- Grammar rejections0: на этих ответах модель сама предложила допустимые токены.
  Наличие rejection/resampling ветки проверено protocol/vocab тестами, а не
  утверждается по данному HTTP smoke.
- card_session завершился, down+rollback выполнены, карта возвращена майнеру.
  Никаких реальных действий инструментов не выполнялось. Это correctness,
  серверные timings этого smoke не считаются speed benchmark.

Tools23 job подготовлен в server-tools23-v1 и ожидает очереди координатора.
CPU vocab проверка20 исходных prompts:849..874токена, +max6000 укладывается8192.

### Разделение серверного времени и greedy fast path

`m31_serve` возвращает `sampling_ms`/`sampling_calls`, `grammar_wait_ms`,
`interactive_wait_ms`/`interactive_waits`. Счётчики суммарные за запрос, включая
первый sample и последний interactive ACK; прежние границы prompt/predicted
таймеров сохранены. Sampling — только собственный C++ draw; grammar wait —
передача proposal и ожидание helper, interactive wait — ожидание token ACK.
Это позволяет отдельно измерить прежнюю необъяснённую задержку между engine
stages и API; доля пока не измерена.

Для temperature<=0 используется один проход по logits без vector200k и
max_element. Строгое сравнение `>` сохраняет первый минимальныйID при ties;
проверяются все nonfinite, включая excluded/внеallowed. Стохастический путь
перенесён в server_sampling.hpp без изменения арифметики/chain/RNG.

`codex-m31-port4-server-frontend-build` rc0, собственный build-server-eval/m31_serve
линкуется только к свежему build-cpu-decode/libm31_engine.a и готовым ggml/CUDA.
1416 regression checks/128 настоящих code128logitrows: ties/excluded/allowed/
empty/nonfinite и отсутствие расходования RNG совпали со старым greedy.
Лог server-build-frontend.log. Затем9CPUprotocoltests прошли,6real-vocab skipped
в этой сборке (ранее все15 пройдены с настоящим helper).

Tools23 на новом frontend подготовлен в server-tools23-v2, в очередь не поставлен:
сначала координатор проводит отдельное окно dense speed.

`--profile` адаптера дополнительно сохраняет полные done.timings в stderr строкой
`M31 request timings {JSON}` после native predicted timer; по умолчанию false.
Это важно для окна: исходный om_bench.py пишет компактный TSV и не сохраняет
sampling/interactive поля ответа. Сам reference client не изменялся.
Пресет-генератор теперь указывает frozen `card_launch.py`, а correctness/quality
jobs явно ставят measurement:false; speed/full сохраняют exclusive lock.

### Python ACK/nonstream после старта tools23

Tools23-v3 стартовал19:28UTC через immutable card_launch с measurement:false,
старым уже загруженным Python-кодом; его процесс в середине оценки не менялся.
До него повторный38/38 HTTP tool smoke принят.

В новой версии (для следующего старта):

- Nonstream не вызывает промежуточный PEG, только итоговый parse после done.
  piece/UTF8/clipped_text и решение customstop остаются на каждом токене.
- Stream отправляет ACK после clipped_text, до PEG RPC, перекрывая parse с
  native compute. Stop=false/true решается до ACK. При PEG exception finally
  дожидается следующей token boundary и отменяет, не оставляя stale stdin;
  следующий request reset. Дополнительный token наружу не выдаётся.
- Top-level reasoning_effort повторяет server-common.cpp: nonempty значение
  переопределяет kwargs, none выключает thinking и удаляет effort. Тело запроса
  не мутируется. Неправильный тип отклоняется.
- Bench plan исправлен на точный reference OM_OFF={"reasoning_effort":"low"}.
  Прошлые speed окна с enable_thinking=false — диагностические. Исходный quality
  ref не задавал effort, поэтому tools23 идёт defaultadaptive без low.

`codex-m31-port4-server-ack-tests` rc0, **21/21** tests,0.850с,
`server-ack-tests.log`. Добавлены ACK-before-parse, customstop, PEG error drain,
nonstream final-only, cancellation, effort precedence/type и реальные vocab
stream/nonstream finalmessage/finish/timings comparisons для Unicode/think/stop
и toolcall (автогенерированные IDs нормализованы).
Живой smoke новой Python-версии ожидает завершения tools23.

Подтверждённая диагностика предыдущего окна4K: sampling.347/.339ms на token,
interactivewait2.847/3.431ms. Это был adapter RPC/parse overhead, а не sampling.
Строгий speed замер с новым ACK и reference effort=low ещё не выполнен.

### Tools23 завершён

`codex-m31-port4-tools23-v3` rc0: **20/23** против reference19/23,
453.891с. Temperature1/top_p.95/max6000, defaultadaptive thinking без effort,
как исходный reference quality launch. Это один стохастический прогон;
статистически значимое улучшение не заявляется.

`/mnt/ai/logs/m31-port/server-tools23-v3/`:

- `m31-port4-tools23-v3.jsonl`: все20cases+3follow;
- `.raw.jsonl`: исходные messages/tools/choice/usage по каждому запросу;
- `.summary.json`: sampling/sourceSHA/полнота/итог;
- `http-tool-smoke.json`: повторный38/38 gate до качества;
- frozen runner0444, launcherSHA, server/client/telemetry logs.

0 length,0 service-marker leaks. Ошибки проверки:

- tool11: run_shell вместо send_message;
- tool18: systemctl status вместо restart;
- tool_follow1: математически верное103,38 с запятой вместо ожидаемой строки103.38.

Все вызовы получены как валидныйJSON; ни одна команда или сообщение инструмента
не исполнялась. GSM8K не запускался. Carddown, точечныйrollback и возвратмайнера
подтверждены sessionlog, unitinactive/ExecMainStatus0.

После качества новая Python-версия направлена в самостоятельный HTTPgate:
`codex-m31-port4-ack-http-v1`, measurement:false,109generic+38tools smoke,
карта только через immutable card_launch/aictl/flock. Результат ещё ожидается.
22/22 CPU/vocab tests прошли (3.827с), добавлено совпадение stream/nonstreamfinal
на всех обрезанных tokenprefixes think+XMLtool, а не только на полном сообщении.

### Новый Python HTTP gate принят

`codex-m31-port4-ack-http-v1` inactive/rc0:
**109/109** generic HTTP/SSE checks + **38/38** real tools checks.
Отчёты `server-ack-http-v1/http-smoke.json` и `http-tool-smoke.json`.
Stop/zero-token/Unicode/NFC, stream/plain, toolcalls, cancellation/recovery и
usage/timings сохранились. Down+rollback+возвратмайнера подтверждены sessionlog.

Диагностические ACKwait из этого correctness-run (не speed benchmark):
plainnonstream0.126–0.155ms/token, toolsnonstream~.099ms/token,
SSEtool~.048ms/token; отдельный grammarRPC~.31ms/token. Прежние2.8..3.4ms/token
на повторные partialPEG/RPC устранены в этом пути. Полный4K matchedrefbody low
должен проверяться отдельным окном координатора; цифры smokeнеподменяютbenchmark.

Работа server_eval закончена: code/tests/docs готовы в общем дереве, без коммитов
агента, пушей/PR и сохранённых testpresets. Сейчас активных юнитов направления нет.
