# UD-Q2_K_XL: завершённый референс llama.cpp PR #31192

27.10.2026. Проверенный основной режим E3 дал **26,11 / 25,21 / 24,37 ток/с**
генерации на4K/16K/32K, префилл **484,0 / 532,8 / 479,1 ток/с**.
Полная оценка выбранного режима: **GSM8K97/100, tools19/23**,
обрывов поlength0, утечек служебных тегов0. Это самостоятельная приёмка
референса Q2; результаты на Xolotl проверяются отдельно.

## Конфигурация и артефакты

Исходный немодифицированный PR worktree `/mnt/ai/llama.cpp-m31pr-1026`,
HEAD `6b95a9908d2c7b242817a447708aec38e937e2c0`, status пуст.
Существующий принятый бинарь llama-server заморожен для каждого job:
SHA256 `05189b53f19049105b4cde9680bbf2f70b8add55fcc242c8bd311dc5c73afef0`.
Модель: четыре шарда `/mnt/ai/models/MiniMax-M3.1-Flash-GGUF/UD-Q2_K_XL/`,
ревизия `b17b7f5459c0b8f127ea12ee774409b282521d6a`; полнота и SHA подтверждены
Андреем до волны. Полный135ГБ checksum во время замеров не повторяли.

Параметры совпадают с лучшим одно-карточным ref из
`/mnt/ai/logs/m31-ref/REPORT.md`: GPU0, context65536, KVq8_0,
batch/ubatch2048, threads64/threads-batch96, n_gpu_layers999,
эксперты слоёв10/11/12 вCUDA0, остальныеCPU, load-mode none,
flash-attention on, DSpark отсутствует, stockCUDAgraphs/fusion включены.
Пресет m31 не меняли. Рабочий порт8183/localhost; точный argv в manifest.

Для проверки VRAM дополнительно испытан один режим E4: слои10/11/12/13
наCUDA0 при тех же прочих параметрах. Он вместился, но генерация медленнее
на всех трёх глубинах. Выбран E3. E5 не запускали: после E4 оставалось
1652МиБ VRAM, дополнительный экспертный слой весит2214МиБ, ещё до scratch.

## Скорость

|Режим|PP4K/16K/32K|TG4K/16K/32K|GSM100|tools23|
|---|---|---|---:|---:|
|llama.cpp IQ3_S, прежний ref26.10|425/—/433|23,33/22,63/21,92|96|19|
|llama.cpp Q2_K_XL E3, выбран|484,0/532,8/479,1|26,11/25,21/24,37|97|19|
|llama.cpp Q2_K_XL E4|489,4/539,0/483,8|25,22/25,02/23,46|не запускали|не запускали|

Для Q2 каждый режим имеет один проход оригинального неизменённого
`~/om_bench.py`: warmup+4K/16K/32K, 256выходных токенов на глубину,
temperature0/ignore_eos/cache_prompt=false/reasoning_effort=low.
Корпус прежний `q38bench-corpus.txt`. Фактические prompts E3:
4349/16635/33022; E4:4348/16636/33019. Случайные UUID и выходы различаются:
это одинаковая методика, не побайтово парные запросы/логиты.
Старый IQ3-ref исторический, не новый same-day запуск llama.cpp.

E4 против E3: TG−3,41%/−0,75%/−3,73%, PP+1,12%/+1,16%/+0,98%.
PooledTG для равных3×256 токенов: E3 25,2100, E4 24,5409 (−2,65%).
Выбор лучшего из двух испытанных вариантов не является исчерпывающим тюнингом
или доказательством причины изменения скорости. PeakGPU E3 27993МиБ,
E4 30459МиБ; modelGPU15415,48→17629,48МиБ,
computeGPU6459,21→6715,21МиБ, KV5100,06МиБ.

Дополнительно по просьбе диспетчера независимо проверен новый same-day
контроль защищённой XolotlIQ3_S V8: PP1382,6/1539,0/1254,9,
TG25,97/24,46/23,12, pooled24,4615; один проход. FrozenV8 ELF/adapter/
tokenizer/runner, old48ENV и2unsets совпали с принятой V8.
Новый GSM/tools на этом контроле не проводился; прежние97/19 не являются
новыми результатами. Это другой движок/квант/adapter против Q2-ref,
не парный по logits опыт.

## Полная оценка качества выбранного E3

Оригинальный `~/eval_duty_1023.py`, первые100GSM,
20инструментальных запросов+3продолжения, temperature1/top_p0,95/max_tokens6000,
размышления по умолчанию. Фактические stock sampler defaults top_k40/min_p0,05
проверены отдельным sampler audit. Инструменты проверяются какJSON;
команды/сообщения не исполняются. Никаких перезапусков/выборки лучшего
качества: один непрерывный исходный клиент,123строки.

Итог97/100 и19/23, все3follow верны. Ошибки GSM0-based:[12,73,85],
tools:[5,9,11,18], follow:[]. Length0/leak0. Всего26654completion tokens,
1148,2с суммарного времени ответов, GSM~1032с/tools~116с.
Одно стохастическое измерение не доказывает улучшение качества против
исторических96/100 и97/100 других режимов.

Это качество только выбранного HTTPq8/fusion/resident-E3 режима.
Основной numerical oracle Xolotl использует прежний acceptedGPU-offload
harness сF16KV, all-CPU expert storage+op_offload1, fusionOFF.
Дополнительный resident-E3 F16 code128 diagnostic готовит q2_compat.
Качество между этими режимами, E4 и финальным Xolotl не переносится.

## Проверки и управление ресурсами

SpeedE3/E4: card/session/client/server rc0, все256×3выходов,
pre/post source/frozen/client/corpus SHA стабильны. Quality: те же rc0 и SHA,
полные идентификаторы0–99/0–19/0–2 без пропусков,123строки,
GSM≥95/tools≥19. Каждый reference job завершён, собственный сервер остановлен,
свой временный preset восстановлен обработчиком перед следующим job.
После reference quality auto-chain перешёл к Xolotl quality; это не
утверждение, что карта сейчас свободна. Финальный общий resource audit
выполняет координатор. Vision worktree/юниты и GPU1 не трогали.

Запуск только через aictl/card_launch и общий card lock. Speed имеет
exclusive build lock; quality shared. Frozen controller KillMode=mixed,
TimeoutStopSec600; worker владеет всеми модельными потомками.
Именованный down --if-owned выполняется после завершения up;
rollback восстанавливает только свой ключ, защищённый m31 не меняет.
Бэкапы/исторические файлы не удалялись, пуша/PR нет.

Основные пути относительно `/mnt/ai/logs/m31-port/`:

- `q2-reference-speed-e3-v1/{manifest.json,summary.json,memory-summary.json,speed.tsv}`.
- `q2-reference-speed-e4-v1/{manifest.json,summary.json,memory-summary.json,speed.tsv}`.
- `q2-reference-e3-e4-selection-v1.json` — честный выбор одного из двух проходов.
- `q2-reference-quality-e3-v1/{summary.json,quality-post-audit-v1.json}`.
- `q2-reference-quality-e3-v1/eval/m31-q2-ref-e3.jsonl` — все123сырые строки.
- `q2-wave-iq3-v8-control-v1/{reference-independent-preflight-audit.json,freshV8summary.log.json}`.

Исходные reproducibility helpers: `tools/m31/q2_ref_prepare.py` и
`tools/m31/q2_ref_summarize.py`. Prepare не запускаетGPU и требует нового
каталога; submit только через queuedcard_launch. У reference helper нет
правок core/вендора/моделей. Итоговые локальные коммиты делает root.
