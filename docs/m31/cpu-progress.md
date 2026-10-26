# CPU decode: port4, первый этап

Работа отделена от GPU cache. Цель 18.13 → ≥23.33 ток/с пока **не измерена**.
Один NUMA узел на машине; interleave между узлами не даст резерва.

Добавлен opt-in `Config.cpu_decode_cache` / CLI `--cpu-decode-cache 1`:
однотокенные CPU routed graphs сохраняются отдельно для каждого слоя. Их
исходные ggml ops, порядок F32 суммирования, типы весов, IDs/weights сохраняются;
при каждом вызове обновляются input/ids/weights. Смена prompt через reset()
сохраняет графы, но не активации. Batch n>1 и taps идут прежним путём. Default false.
Один cached graph имеет 1MiB metadata и собственный gallocr; отсутствует повторная
постройка графа/alias-plan на следующих single-token шагах. Нет предположения, что
разные transient graphs с одинаковым числом nodes имеют одинаковый alias-plan.

`--profile 1` теперь выводит stage timings prepare, dense, routed setup/compute/read,
residual и отдельные head setup/compute/read. Дальше нужен реальный профиль: сам
статический аудит **не доказывает**, что graph construction определяет потерю TG.
Общий forward timer не включает prepare и head; они печатаются отдельно.

Добавлен `Config.route_observer(layer,n,ids,weights)` для калибровки рейтинга.
IDs/weights token-major `[n,used]`; default callback пустой.

Отдельный link-only build `build-cpu-decode` через `tools/m31/cpu_build.sh`
и `cpu_build/CMakeLists.txt`: использует готовые static vendor libs из
`build-m31-cuda`, vendor не пересобирает. Shared `.build.lock`, `-j24`.

Проверка v2 rc0: старые CPU tiny/NumPy/arena tests и новый `cpu_test_decode.py`.
Новый тест: chunk1, chunk7+decode_tail11, taps fallback; два разных prompts с reset
на одном Engine; все logits cache0/cache1 побитно одинаковы. Затем добавлены head
и prepare timers; v3 пересобирает финальный source и повторяет те же tests.

Активные units:
- `codex-m31-port4-cpu-decode-build-v3` — финальная build/tiny; лог
  `/mnt/ai/logs/m31-port/cpu-decode-build-*.log`.
- `codex-m31-port4-cpu-decode-correct-v1` — ждёт v3 rc0, затем card_session
  `/mnt/ai/logs/m31-port/job-cpu-decode-correct-v1.json` (aictl/lock/trap).
  Code128 и code3000+decode_tail32 against исходные M1 oracle datasets.
  Результаты `cpu-decode-code128-compare-v1.json` и
  `cpu-decode-long-compare-v1.json` в logs/m31-port.

После correctness: отдельное окно без сборок/CPU нагрузки, CPU stage/perf profile,
затем om_bench TG4K на live server с cache0/cache1. Серверному направлению передан
запрос пробросить CLI флаг (serve_m31.py и m31_serve.cpp принадлежат ему).
Без замера нельзя заявлять ускорение или M2 acceptance.
