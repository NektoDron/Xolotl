# MiniMax-M3.1: экспериментальный M1

**Код написан, но ещё не собран и не проверен.** На 2026-10-26 референсный
`codex-m31-ref` работает; до его завершения запрещены сборка и inference.
Проект M0: [../../docs/m31/M0.md](../../docs/m31/M0.md).

Модуль включается `-DSTRATA_M31=ON` и не меняет пути других моделей.
Это собственный граф M3.1: GQA с per-head RMS/partial NeoX RoPE, MSA на каждом
слое, sigmoid router и `swigluoai`. CPU ggml вычисляет исходные routed tensors;
GPU держит исходные dense/shared weights, F16 main/index KV. BF16/Q6_K автоматически
не переквантываются. RAM-арена, VRAM-кеш, сервер, драфт и vision ещё не подключены.
Transient graphs и копирование промежуточных активаций оставлены для M1, не для M2.

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
с F16 cache/query rounding. Она не заменяет PR на настоящем IQ3 GGUF. Текущие
тесты не являются доказательством правильности, пока их не запустили.

Реальная модель на GPU запускается только **из aictl preset**, под
`flock -w 86400 /mnt/ai/logs/.xolotl-card.lock`, с trap down ПОСЛЕ возврата up.
`M31_CARD_SESSION=1` — защита от случайного запуска, не разрешение обходить aictl.
Preset пока не создан: сначала сборка/CPU fixtures, затем собственный launcher,
backup config.json, rollback-команда и снятие тестового preset в конце M1.
Не использовать занятый чужой preset для резервирования карты.

После M1 адаптировать Ds4MoeTier (OAI во всех kernels, больший CPU scratch,
NVFP4 scales и корректный blob stride), затем подключать RAM arena/cache/MMQ.
Все approximations, route penalties, expert dropping и requant выключены.
