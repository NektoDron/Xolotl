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
