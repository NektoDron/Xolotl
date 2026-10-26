# Независимый review shared FFN / CPU MoE overlap

26.10.2026. Только чтение кода, backend contracts и сохранённых небольших
артефактов; GPU/сборки/замеры в этом review не запускались.
**Blocking lifetime/race/default regressions не найдено.** К завершению review
реальный3000/decode-tail32 численный gate прошёл; доказательство фактического
перекрытия остаётся отдельным.

Source freeze `tools/m31/m31_engine.cpp`:
`c797b6e94f4fd961486354461f54857488dce586d145c66f1021535c249216c4`.

Проверенные участки:

- **Guard/default:** `m31_engine.cpp:884–890` требует explicit
  `M31_CPU_SHARED_OVERLAP=1`, CUDA backend, reuse allocations, CPU residual,
  dense decode cache и отсутствие taps. На571–573 добавлены n=1/routed layer
  и отказ overlap, если данный MoE идёт на GPU. CPU/prefill/taps/default
  сохраняют совместный исходный граф со shared FFN.
- **Input/scratch ownership:** на575–586 shared FFN имеет собственные
  context/gallocr/input/output, а не borrowed attention workspace/views.
  Он использует только постоянные dense weights и свой F32 input. Нормальный
  norm readback завершён до enqueue; tensor_set на588 копирует CPU vector.
  В вендоренном `ggml-cuda.cu:844–850` set_tensor выполняет H2D плюс
  `cudaStreamSynchronize(cudaStreamPerThread)`: host vector после вызова
  не является borrowed GPU input.
- **Join/readback:** `SharedDrain` создан на736–739 до enqueue. Inflight flag
  ставится до async submission на590–592, включая submission-error случай.
  CPU MoE завершается до drain на781; shared tensor_get на784–786 выполняется
  после join. Residual на799–800 сохраняет порядок `(moe+shared)+residual`.
- **CPU graph/cache interaction:** CPU MoE использует отдельный CPU backend
  и workspace. Его Graph destructor вызывает cache_clear с CPU backend,
  но `ggml-cuda.cu:5419–5422` немедленно возвращается для не-CUDA backend.
  Поэтому CPU MoE не очищает CUDA captures, пока shared FFN в полёте.
  Dense temporary graph уничтожен ещё на727, до запуска shared на743.
  При активном overlap dense cache обязателен, residual CPU и GPU MoE
  исключён guard: промежуточного CUDA graph-cache-clear на этом пути нет.
- **Reset/bucket/error/destruct:** clear_dense_decode_cache на595–602 сначала
  drains. При смене attention bucket на961 shared graphs остаются валидными:
  они вообще не ссылаются на mask/positions/KV. Reset935–937 освобождает и их
  только после drain; destructor337–354 drains перед graphs/weights/state.
  Guard на738 выполняется и при исключении CPU MoE/после launch; forward1002
  помечает Engine poisoned. Caller должен reset перед повтором, как прежде.
  Одновременный reset/forward одного Engine не поддерживается; этот patch
  не добавляет такую модель использования.

Прочитанные доказательства, без повторного запуска:

- `cpu-shared-overlap-build-20261026T204414Z.log`: CPU fallback/taps/no-dense
  прошли, async shared launches на CPU отсутствуют, прежний tiny math/arena
  regression gate пройден.
- `job-cpu-shared-overlap-correct-v1-00.log`: сохранены CUDA PASS для4 modes:
  overlap, taps fallback, no-dense fallback, GPU-routed decode fallback.
  Runner `cpu_shared_overlap_check.cpp:31–35,68–72` также проверяет injected
  post-launch error, poisoned state и reset/recovery побитно с baseline.
- К завершению review появились `cpu-shared-real-base-compare-v1.json`,
  `cpu-shared-real-overlap-compare-v1.json`, `cpu-shared-real-compare-v1.json`:
  baseline и overlap против CPU oracle и друг друга — все3000rows,
  top1=1, mean/maxKL0, per-case maxabs0. Прочитан job result: rc0, все6
  commands rc0. Настоящая модель, chunk128/prefix2968 + decode-tail32,
  threads64/F16/ctx4096/nofusion; исходный CPU oracle не подменён.

Неблокирующая граница доказательства: host `submit/cpu_begin/cpu_end/joined`
timestamps показывают порядок вызовов, но сами не доказывают, что CUDA kernels
реально шли одновременно с CPU MoE. Для такого утверждения нужен отдельный
Nsight/CUDA event trace (`job-cpu-shared-overlap-events-v1.json` подготовлен).
В timing label joined_ns дополнительно входит shared readback после drain.
Этот review и численный gate не являются заявлением ускорения; speed требует
отдельного изолированного окна.

Финальная отметка координатора: rolling M2/HANDOFF/README синхронизированы после завершения окон. Stock GgmlGPUoracle full20 принят отдельно; shared-overlap real3000/d32 и CUDA/NVTX proof завершены, benchmark ниже ref. Все окончательные числа/границы — REPORT-port4.final.md и port4-final-summary.json, незапущенные GSM100/16K/32K явно перечислены.
