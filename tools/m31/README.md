# MiniMax-M3.1: M1 принят, M2 выигрыш не получен

26.10.2026, port4. Exact GPU + async прошёл20/20/28 723 к CPU strict
oracle побайтово, KL0. Stock CUDA отдельно прошёл full20 к GPU-offload
oracle; контракты квантизации активаций не смешивать.

Последний best native CPU median: TG22.155, PP183.3; ref23.33/425,
ref PP32K433. Exact GPU46.2PP/21.375TG, stockGPU95.35PP/20.75TG.
M2 не принят, M3 не запускался; full16K/32K иGSM100 ещё не измерялись.
Tools20/23vsref19/23; M4 CPU embeddings PNG/12frames приняты, native
vision ещё нет. Все результаты в [M2](../../docs/m31/M2.md),
`HANDOFF.md`, `/mnt/ai/logs/m31-port/REPORT-port4.final.md`.

Последние source CPU/CUDA builds обновлены и прошли checks. Все изменения
opt-in; production пути не менялись. Reproducible frozen ELF/source/env
и TSV/JSON находятся в `/mnt/ai/logs/m31-port/`.

GPU jobs только через immutable launcher с НОВЫМИ job/output именами:

```bash
python3 tools/m31/card_launch.py /mnt/ai/logs/m31-port/NEW.json --unit codex-m31-NEW
```

Runner0444+SHA, systemd User=nekto, card flock/aictl/backup/down/rollback.
measurement:true(default) — exclusive build lock для скорости; false —
shared для correctness. GPU0 only, nofusion. GPU1/prod/up*/hxq3k/m31
не трогать. Не редактировать живые shell.

Exact GPU: STRATA_M31_CPU_QUANTS_CUDA=ON и M31_GPU_CPU_QUANTS=1.
Stock GPU: M31_GPU_CPU_QUANTS=0, другой placement oracle. Async uploader
M31_GPU_PREFILL_ASYNC=1, явные arena/cache budgets. Env HOST_THREADS,
EXPERT_SORT_ROUTES, REUSE_HOST_READS, CPU_SHARED_OVERLAP и seed receipt
описаны в direction docs. CPU decode остаётся CPU при min_tokens32;
shared overlap только CUDA+n1+CPU-routed, без taps и с graph caches.
Большую THP арену не форсировать;64K memory budget ещё не замерен.

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
