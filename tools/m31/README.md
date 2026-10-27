# MiniMax-M3.1: M2 принят, port6

Финальная canonical CMake сборка V8 прошла все критерии. Медианы двух
кругов om_bench на 4K / 16K / 32K: PP **1391 / 1535 / 1247 ток/с**,
TG **27,45 / 27,02 / 25,31 ток/с**. Сырые TG32K: 24,61 и 26,00;
один запрос ниже 25, медиана выше 25. GSM8K100: **97/100**, tools23: **19/23**.

Полный gate20 (28 723 позиции) и ещё 768 выбранных long/direct позиций
совпали с GPU-offload llama.cpp PR #31192: top-1 100%, KL/maxabs 0.
В длинном oracle физическая ёмкость KV была уменьшена при сохранении
фактических chunks, позиций и RoPE; отличие явно записано в receipts.
Strict CPU — отдельный режим проверки, включаемый через `M31_GPU_EXACT=1`.

Подробности: [MINIMAX_M31](../../docs/MINIMAX_M31.md), `HANDOFF.md`,
`/mnt/ai/logs/m31-port/REPORT-port6.final.md`. Полное машинное доказательство:
`port6r4-final-v8-acceptance-v1/acceptance.final-v1.json`, все 16 проверок PASS.

Принятый профиль находится в [profiles/native-gpu.json](profiles/native-gpu.json).
Он хранит 48 ENV и CLI параметры без локальных путей модели/executable.
[profile_plan.py](profile_plan.py) печатает готовые argv/ENV для просмотра,
ничего не запускает; применение описано в [profiles/README.md](profiles/README.md).

Основной GPU FFN — stock grouped ggml MMQ, с native activation quantization.
GPU prefill chain и зарегистрированная арена включены в принятом профиле,
fusion выключен; BANKS/GU/MASK остаются отдельными экспериментальными опциями.
Exact-debug использует `EXACT1/CHAIN0`, возвращает промежуточные значения в RAM
и требует compiled CPU-compatible kernels. `CPU_QUANTS=1` сам режим не выбирает.

Проверенная конфигурация: GPU0 RTX5090, CPU80/HOST32/PACK8, context40000/F16 KV,
maxchunk32768, arena170GiB/cache7GiB/seed0. Это не тот же KV/placement профиль,
что у speed reference llama.cpp Q8_0. HIP/SYCL и контексты больше 32K не проверены.

Все GPU jobs запускаются только через immutable aictl launcher с новым именем:

```bash
python3 tools/m31/card_launch.py /mnt/ai/logs/m31-port/NEW.json --unit codex-m31-NEW
```

Замораживаются runner/worker/preset helpers. Controller использует mixed stop,
worker — control-group; up завершается до named down/rollback. Для скорости
нужен `measurement:true`, который берёт исключительный build lock.
На aiserver установлены opt-in aictl guards `up --no-switch` и
`down NAME --if-owned` под `/run/aictl.lock`. Патч: `card_aictl_guard.patch`.
Системный бэкап: `/usr/local/bin/aictl.bak-m31-port6-guarded`. Откат:

```bash
sudo cp -a /usr/local/bin/aictl.bak-m31-port6-guarded /usr/local/bin/aictl
```

Новый opt-in общий HTTP bridge — [serve/serve_m31.py](serve/serve_m31.py).
Он прошёл CPU и native GPU smoke OpenAI/Anthropic, отмену с новым запросом и nested tools.
Полный GSM/скорость относятся к standalone adapter; bridge отдельно их не проходил.
Prefix reuse и engine vision пока не подключены. DSpark не запускался.
GPU1, production, up*/hxq3k и постоянный preset m31 не менялись.

Ниже исходная последовательность M1 для справки.

## После завершения референса

Сначала прочитать `/mnt/ai/logs/m31-ref/REPORT.md`. Сборка и проверки отдельными
systemd unit, последовательно (они могут скачивать исходники ggml):

```sh
sudo systemd-run --unit=codex-m31-port-build-cpu -p User=nekto \
  /bin/bash /mnt/ai/Xolotl-m31/tools/m31/build.sh cpu
# После успешного окончания CPU-плеча:
sudo systemd-run --unit=codex-m31-port-build-cuda -p User=nekto \
  /bin/bash /mnt/ai/Xolotl-m31/tools/m31/build.sh cuda
```

Логи: `/mnt/ai/logs/m31-port/build-*.log`. Скрипт проверяет ref unit перед
configure/build/tests, GPU не занимает; synthetic fixture checks идут на CPU.
Конфигурация gcc-14/CUDA13/sm_120a и compatibility header взяты из build-up5.
Используется собственная FetchContent-копия ggml, не изменяются up*/ref worktree.
Отдельный oracle связывается с уже готовыми библиотеками PR (static/shared),
исходный SHA проверяется. Ошибки компиляции/фикстур исправлять до реальных запусков.

## Инструменты

- `m31_inspect ANY-SHARD.gguf`: только заголовки, полная карта тензоров и байты;
  обязательны все четыре index tensors на плотных слоях и scales у NVFP4.
- `m31_generate -m MODEL --cases ID-DIR --out NEW-DIR --backend cpu|cuda
  --ctx 4096 --chunk 128 --threads 32 --rows all --taps none|layers|all`:
  прямой проход, без sampling/server. CPU backend по умолчанию. CUDA требует
  `M31_CARD_SESSION=1`, который выставляет только запуск через aictl под card lock.
- `m31_oracle`: тот же dump CLI, отдельный процесс PR #31192. CUDA: dense GPU0,
  routed tensors CPU, F16 caches и flash attention; один sequence.
- `m31_oracle tokenize MODEL TEXT-DIR NEW-ID-DIR`: vocab-only, CUDA скрыта;
  полученные `.i32` подавать ОБОИМ движкам. Для каждого `.txt` опциональный
  `.length` задаёт длину префикса. Тексты не являются benchmark-ответами.
- `--decode-tail 32` у обоих движков: префикс chunk128, последние32 токена
  chunk1. Проверяет gather/decode после включения выбора MSA, сохраняя все logits.
- `make_prompts.py NEW-DIR`: 20 фиксированных RU/EN/code/math промптов,
  64..3000 токенов; NFC/decomposed Unicode и границы MSA.
- `compare_logits.py REF-DIR NATIVE-DIR --require-suite --out NEW.json`:
  сверяет ID, модель, позиции, размер/finite logits, считает F64 KL(reference||candidate).
  По умолчанию top1>=.99, meanKL<1e-5, maxKL<1e-3. При `--require-suite`
  требует все позиции всех двадцати промптов. Без него результат — diagnostic.
- `compare_taps.py REF-CASE NATIVE-CASE`: первый расходящийся `l_out`;
  затем `--all-common` — attention/MSA/router/FFN taps. Для локализации rerun
  с одинаковым chunk и `--taps layers` / `all`.

Dump содержит `tokens.i32`, `logits.f32`, manifest (пишется только при успехе),
опционально `taps.jsonl` и binary taps в логическом GGML порядке (dim0 contiguous).
Файлы little-endian; все реальные запуски на x86_64 aiserver.
Выходной каталог должен быть новым, незавершённые результаты не принимаются.

На полном наборе ~27 тыс. позиций ×200064 logits: нужны десятки ГБ для пары
dump; taps запускать только на первом расходящемся кейсе. Для первичной
диагностики есть `--rows stride:32`, но это не полная приёмка M1.
Проверить отдельно chunk1 и chunk128: decode использует gather, prefill — mask.
Сравнение raw logits проверяет математику; перед сервером отдельно проверить
NFC BPE, chat template, stop/think/tool markers и потоковый parser.

## Допуски и следующий шаг

NumPy fixture — независимая проверка маленькой F32 архитектуры и split-prefill,
с F16 cache/FA-query rounding и F32 indexer score. Она не заменяет PR на настоящем
IQ3 GGUF. CPU gate пройден: chunk1/7/29, maxabs logits=0.00011285; 4/4 проверки
dump/comparator, loader negative cases. Реальный M1 принят: [../../docs/m31/M1.md](../../docs/m31/M1.md).

Реальная модель на GPU запускается только **из aictl preset**, под
`flock -w 86400 /mnt/ai/logs/.xolotl-card.lock`, с trap down ПОСЛЕ возврата up.
`M31_CARD_SESSION=1` — защита от случайного запуска, не разрешение обходить aictl.
Preset пока не установлен: сначала сборка/CPU fixtures, затем собственный launcher,
backup config.json, rollback-команда и снятие тестового preset в конце M1.
Не использовать занятый чужой preset для резервирования карты.

Собственный batch launcher: `card_session.sh JOB.json`, запускать через
`sudo systemd-run --unit=codex-m31-port-... -p User=nekto /bin/bash ...`.
JSON содержит `{"commands": [["/absolute/executable", "arg", ...], ...]}`.
Сам worker и модели стартуют из временного `m31-xolotl-check` через aictl.
Контроллер держит card/build locks, пишет session log и backup/rollback-команду,
ждёт возврата up, result-файла, затем down и откат только своего ключа.
JOB должен быть новым, лежать в `/mnt/ai/logs/m31-port/`; команды выполняются
последовательно до первой ошибки, их логи и `.result.json` сохраняются рядом.

Oracle M1: dense CUDA, routed CPU, `op_offload=false` (CPU placement сам по себе
не запрещает ggml копировать экспертные операции на CUDA). Native M1 использует
те же CPU kernels/activation quantization. Output head вычисляется целым chunk,
как в oracle; однострочный MMV и батчевый MMQ дают разное округление активаций.
CPU-only FA перед накоплением переводит V из F16-кеша в F32: stock ggml CPU FA
иначе округляет накопитель после каждого key даже с `GGML_PREC_F32`.

**Обязательная конфигурация текущего M1:** `GGML_CUDA_DISABLE_FUSION=1` для
native И oracle (worker выставляет её сам). Включённая fusion в этой конфигурации
даёт неверные logits в обоих процессах; точный fused kernel ещё не локализован.
Замена gallocr на scheduler не помогла и отменена. На nofusion исправлены F32
indexer scores и pad256 для main/index KV, как в PR: 64RU и 3000code (все строки)
совпали побитно, top1=1/KL0/maxabs0. Полный gate20 и decode-tail пройдены побитно.
Padding128 математически маскирует те же строки, но меняет FA kernel/tiling:
на chunk128 разница attention ~1e-4 относительных усиливалась по60 слоям.
Не подменять этот oracle speed-референсом: M2 сравнивается с REPORT.md
(23.33/22.63/21.92 TG, 425/433 PP), с обычным op_offload и KV q8_0.

При подключении общего Ds4MoeTier нужны OAI во всех kernels, больший CPU scratch,
NVFP4 scales и корректный blob stride. Затем cache/MMQ; RAM arena уже проверена.
Все approximations, route penalties, expert dropping и requant выключены.

## M2 RAM arena (подготовка тиринга)

`m31_generate ... --arena-gib170` — opt-in точная перепаковка7296 экспертов
в pageable RAM. `[gate|up|down|NVFP4-scale-tail]`, aligned stride; CPU ggml
использует custom nb[2]. Бюджет должен вмещать полную арену, MemAvailable
reserve16GiB. Default0 — mmap; oracle этот флаг не принимает.
CPU/CUDA tiny packed/mmap побитно. Реальный code128:152.51GiB, pack73.160с,
все logits побитно совпали с PR. GPU cache/streaming/server пока не подключены.
Подробнее и дальнейшие действия: [../../docs/m31/M2.md](../../docs/m31/M2.md).
