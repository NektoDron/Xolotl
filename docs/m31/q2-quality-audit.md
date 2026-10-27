# UD-Q2_K_XL: независимый аудит GSM8K100 и tools23

Волна закрыта как **NOT_ACCEPTED**: baseline Q2 получил 97/100 и 17/23,
а cached-E3 не прошёл численные long gates и не запускал новое качество.

Это аудит новых результатов Q2. Значения 97/100 и 19/23 принятой IQ3_S V8
не наследуются. Native и reference проверяются отдельно. Оба используют
ревизию GGUF `b17b7f5459c0b8f127ea12ee774409b282521d6a`; payload SHA256
четырёх шардов ранее сверил пользователь, аудит не перечитывает 146 ГБ.

## Метод и текущий результат

Исходный клиент `/home/nekto/eval_duty_1023.py`, первые 100 задач
`/mnt/ai/eval/gsm8k_test.jsonl`, затем 20tool questions и 3follow-ups.
Фактические параметры: `EVAL_TEMP=1.0`, `EVAL_TOPP=0.95`, max_tokens6000;
TOPK/MINP/MAX/OM_OFF явно сняты, thinking по умолчанию модели. Старый docstring
клиента упоминает temperature0, но запуск задаёт 1.0 через environment.

| Конфигурация | GSM8K100 | Tools23 | length | recorded leak | Аудит |
|---|---:|---:|---:|---:|---|
|llama.cpp PR31192, E3/Q2, T64/TB96/64K/Q8KV/ub2048|97/100|19/23|0|0|FINAL PASS|
|Xolotl baseline canonical q2-v1, T80/40K/F16KV/chunk32768|97/100|17/23|0|0|Evidence PASS; tools threshold FAIL|
|Xolotl cached-E3, cache9.88/384/40K/chunk6656|не запускался|не запускался|—|—|0 quality попыток; numeric FAIL|

Reference получен заново:100GSM+20tool+3follow=123 уникальных строк,
GSM ошибки с нуля `[12,73,85]`; tool ошибки `[5,9,11,18]`, follow3/3.
Поле err пусто во всех строках; ложные ответы отличаются от ошибок выполнения.
Финальные HTTP и card-job result имеют rc0 и точные argv/spec.
Все 8 заявленных reference SHA совпали до и после прогона.

Reference receipt:
`/mnt/ai/logs/m31-port/q2-reference-e3-quality-independent-final-v1.json`.
Native receipt:
`/mnt/ai/logs/m31-port/q2-native-quality-independent-final-v1.json`.

Native также имеет 123 уникальные упорядоченные строки: GSM ошибки
`[12,73,89]`, tool ошибки `[9,10,11,13,18]`, follow ошибка `[1]`.
Все 11 заявленных native SHA совпали до и после; HTTP и card-job rc0,
argv/spec/profile совпали. Поля err/length/leak пусты. Порог GSM≥95 пройден,
порог tools≥19 **не пройден**; Q2 native не получает полной quality acceptance.
Сторож завершился rc0/inactive после обоих запусков; итог
`/mnt/ai/logs/m31-port/q2-quality-independent-watch-v2.summary.json`.

В сохранённых native ошибках виден выбор инструментов:9/10 — run_shell с find
вместо search_files,11 — поиск контактов через shell вместо send_message,
13 — вычисление через shell вместо ответа без инструмента,18 — status вместо
restart. Follow1 вернул корректные `103,38 EUR`, но fixture требует строку
`103.38`; по неизменному критерию это false. Все JSON разобраны, ошибки сервера
здесь не выявлены. Это объяснение результата, а не пересчёт 17 в другой балл.
У reference ID5 неверен, а 10/13/follow1 верны; итог отличается на 2.

## Что проверяет reducer

`/mnt/ai/logs/m31-port/q2_quality_audit.py` не запускает HTTP, модели,
инструменты или сообщения. Он проверяет:

- точный набор и порядок IDs: `gsm8k0..99`, `tool0..19`, `tool_follow0..2`,
  отсутствие дублей, пропусков и посторонних строк;
- gold из привязанного корпуса и numeric score из записанного pred/gold;
- q/want из bound client CASES, повторное применение predicate к сохранённым
  имени и аргументам первого tool call;
- ошибки, finish=length, leak flags и видимую часть follow-up answer;
- реальные server model/ELF/tokenizer argv и environment native profile;
- original client/corpus/profile/ELF SHA, HTTP spec/client result и полный
  outer card-job result с соответствием выполненных и плановых commands.

Большие SHA берутся только с shared `.build.lock`, после завершения speed
окон. Final reducer ждёт outer card result до 60 секунд после HTTP result.
Сторож `codex-m31-q2-quality-independent-watch-v2` заканчивается после обоих
baseline quality runs; сам он не держит карту и не вызывает aictl.

Клиент и корпус в actual argv используют исходные пути, а не frozen копии;
их содержимое связано с подготовленным manifest SHA и повторно сверено.
Reference ELF frozen SHA256:
`05189b53f19049105b4cde9680bbf2f70b8add55fcc242c8bd311dc5c73afef0`.
Native serve ELF frozen SHA256:
`2e2a4de354ad3b4fb1e12a343f42425a08401b8b20d3a2edfc423a5bd58a7a90`.

## Границы независимого пересчёта

GSM JSONL содержит только извлечённые pred/gold, не raw content; заново
проверить извлечение числа из полного ответа или leak regex по всему ответу
невозможно. Числа и score пересчитываются, recorded leak flags учитываются.
Tool JSONL сохраняет только первый parsed call. Follow answer обрезан 300 знаками,
а дополнительные tool calls не сохранены; полный follow predicate независимо
не восстанавливается. Видимые противоречия выявляются, сохранённый ok не
подменяется неподтверждённой оценкой.

`full_audit_pass` означает целостность новых evidence, отдельно от
`quality_thresholds` GSM≥95/tools≥19. Один stochastic run не доказывает
статистическую эквивалентность квантов. Scope quality ограничен проверенным
ELF/profile; последующие opt-in guard/candidate builds его автоматически
не наследуют. Имена baseline Q2 и IQ3_S V8 разделены во всех результатах.


## Cached-E3: отдельная конфигурация без новых quality данных

Profile SHA `fd78776df17eae2ee1c53ca545e722b7d48b4b0aa4b238e2a81f66d7f99b8acb`,
существующий ELF2e2a4d..., cache9.88GiB/384 slots, seed всех 128 экспертов
слоёв 10/11/12, maxchunk6656/adaptive, ctx40K. Совпадение ELF с baseline
не разрешает переносить 97/17 на этот profile: placement и PP geometry другие.

Контроль code128 совпал побитно, но чистый обычный resident-E3 oracle
(PP≤2048, stock last-FFN59 pruning) не совпал с native PP≤6656/full-last-FFN:

| Проверка | top1 | mean KL | max KL | Итог |
|---|---:|---:|---:|---|
|code128|1,00000|0|0|PASS|
|code3000|0,87500|0,226658|1,152634|FAIL|
|16K|0,84375|0,213821|2,298108|FAIL|
|32K|0,68750|1,980432|13,538279|FAIL|

Источник: `/mnt/ai/logs/m31-port/q2-cached-e3-crossbatch-final-summary-v1.json`.
Batch и last-FFN policy различаются явно, поэтому это отказ численной приёмки,
а не изолированное доказательство ошибки какого-либо ядра. Старые OOM и
ресурсно загрязнённые попытки не используются как численные результаты.

Из-за gate FAIL cached-E3 om_bench и GSM/tools не запускались: **0 новых
скоростных прогонов и 0 quality попыток**. Prepared quality eval directory пуст;
prepared single-profile watcher не запускался. Baseline97/17, reference97/19
и исторические IQ3_S V8 scores сохраняют собственный scope. Итоговый machine
summary: `/mnt/ai/logs/m31-port/q2-quality-independent-final-summary-v1.json`.

Будущая source-гипотеза — layer-major проход с 2048-token microbatches для
плотной части/индексера/FFN и удержанием весов экспертов слоя между этими
microbatches. Нужно воспроизвести KV/MSA visibility и ordinary last-FFN
pruning/source-quantization policy. Простое уменьшение Engine chunk с новой
загрузкой всего набора весов на каждый chunk этому не эквивалентно.
Реализации, замера скорости и quality proof этой гипотезы здесь нет.
