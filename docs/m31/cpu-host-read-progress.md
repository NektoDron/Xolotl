# Persistent host readback vectors, opt-in

26.10.2026, port4. Менялся только `m31_engine.cpp`; Config/header ABI, ggml ops,
GPUExpert API, model tensors и арифметика не менялись. GPU source freeze сообщён
GPU-агенту после CPU checks; длинный gate20 работает на прежнем frozen binary.

## Что сделано

`M31_REUSE_HOST_READS=1` включает режим только при `reuse_allocations=true` и
отсутствии taps. Иначе безопасный fallback на старый ownership; default OFF.
Флаг фиксируется при init Engine, не читается на каждом слое.

- Новый `Graph::read_into` делает те же contiguous F32 проверки и тот же
  `ggml_backend_tensor_get` с теми же байтами, но resize/read в переданный vector.
- В Impl живут отдельные host vectors input/residual/shared/routeweights/ids/moe.
  Размеры обновляются для каждого chunk; ёмкость сохраняется по слоям и reset.
- CPU routed path имеет внутренний `routed_cpu_into` и читает в persistent MoE
  vector. Default wrapper по-прежнему возвращает owned vector.
- GPUExperts.forward не менялся: его result остаётся прежним owned vector и
  действует до завершения текущего слоя. Нового borrowed результата нет.
- Hidden dense/residual readback также может писать в уже принадлежащий Engine
  hidden vector. Logits API и caller-owned output vectors не менялись.
- Все readback происходят после compute/synchronize, до graph destruction;
  host vectors не становятся tensor views и не получают pointers на scratch.
- При error/reset буферы сохраняют capacity, но следующая операция перезаписывает
  все логические rows. Старые logits caller продолжают владеть собственной памятью.

Одна исходная read allocation/zeroing на первом нужном размере остаётся. Resize
после смены размера также может zero-initialize выросшую logical size внутри
имеющейся capacity. Основная цель — убрать fresh allocation/zeroing на каждом
из57слоёв при неизменном размере prefill; pinned staging здесь не добавлялся.

## CPU checks и доказательство reuse

`codex-m31-port4-cpu-host-read-build-v1` **rc0**. Свой `build-cpu-host-read`,
CPU-only link с существующими vendor archives. Log:
`/mnt/ai/logs/m31-port/cpu-host-read-build-20261026T200224Z.log`.
Тест не резервирует и не инициализирует GPU; скорости из test timers не заявляем.

На synthetic F32 MSA model сравнивались независимые baseline/env1 Engine runs:

- В одном Engine chunks128→2048→1→7.
- Reset, затем1→128→2048→1.
- Invalid token → poisoned state → reset → valid forward.
- CPU residual on/off; persistent CPU/dense decode graphs.
- Taps и reuse_allocations=false: режим выключается, logits и выбранные taps
  побитно baseline; fallback явно проверяется по диагностике.
- Все возвращённые logits побитны; сохранённый caller output не меняется после
  следующих forward/reset.

При `--profile 1` строки `M31 host-reads` показывают реальные capacity changes.
Для fixture D16/used2:

| Chunk | Reads | Capacity growths | Retained bytes |
|---:|---:|---:|---:|
| Первый128 |13|6|34816|
| Затем2048 |13|6|557056|
| Затем1 |13|0|557056|
| После reset,2048 |13|0|557056|

Последующие layers/steps используют старую capacity; test автоматически проверяет
initial growth, shape growth, zero growth на decode и reuse2048 после reset.
`reuses` считает события prepare IDs и float readback, поэтому это не один к одному
с `reads` (IDs подготавливаются также на leading dense слоях).
Retained bytes включают только шесть новых buffers; уже существующий hidden
принадлежал Engine и раньше и в эту сумму не входит.

Воспроизведение:

```bash
sudo systemd-run --unit=codex-m31-host-reads-NEW -p User=nekto \
  /bin/bash /mnt/ai/Xolotl-m31/tools/m31/cpu_host_read_build.sh
```

## Что ещё нужно

1. Incremental main candidate build после source freeze, затем реальный
   code128/3000tail32 against CPU oracle с env0/1 (GPU0 только после текущего
   длинного gate и через aictl/card_launch).
2. В окне без сборок — API4K и per-stage dense_read/routed_read, RSS и minor faults.
   До этого улучшение скорости не заявляется.
3. При подтверждённом остатке pageable D2H staging рассмотреть bounded pinned
   buffers; нынешнее изменение не делает pageable memory pinned и не устраняет
   полную передачу F32 hidden/shared/residual между CPU и GPU.
