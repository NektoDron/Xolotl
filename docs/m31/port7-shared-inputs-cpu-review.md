# Shared verification inputs: свежий независимый review

Read-only review текущих `m31_dspark_verify_state.hpp`, `.inc`, batched graph
и базовых Graph/AttentionInputs/constplanner. Runtime/model parity не запускался.

Блокирующей ошибки alias/ordering не найдено при сериализованном использовании
Engine. Hidden matrix F32[D,K] имеет row leaf aliases по row*D*sizeof(float),
с отдельной n1 contiguous metadata. ПриD6144 stride24576 кратенCUDAalignment.
Graph::borrow дублирует shape/type как новый leaf INPUT, привязывает исходный
buffer/data и не переносит producing graph. Так шестьдесят графов ссылаются
на общий input, не вычисляют и не владеют им повторно.

Attention positions/mask/bias заполняются один раз перед слоями, затем только
читаются. Hidden matrix перезаписывается после завершения предыдущего GPU
слоя и CPU FFN. ggml_backend_graph_compute всегда вызывает backend synchronize,
включая error return. Синхронный tensor_set на CUDA использует свой
cudaStreamPerThread и затем stream synchronize, поэтому переход между ним
и backend graph stream упорядочен. Финальная head получает отдельную
завершённую загрузку hidden. Внутри графа hidden INPUT не перезаписывается.

clear_geometry освобождает head и layer graphs до attention/hidden owners,
после чего освобождает shared buffer и metadata. Impl уничтожает verify_state
до KV/weights/backend. Ошибка частичного prepare_shared оставляет Engine
poisoned; reset уничтожает частично созданные owners. Повторное использование
незавершённого shared state запрещено тем же preflight guard.

workspace_bytes суммирует каждый layer gallocr, head gallocr, один hidden
buffer и K attention buffers. Borrowed aliases отдельно не прибавляются;
gallocr get_buffer_size возвращает собственные buffers, не external leaf inputs.
Shared mode и batch-dense mode входят в geometry key вместе с per-query nk;
K задаётся длиной geometry. Размеры/causal dtype принадлежат неизменному Engine.

Два неблокирующих края:

- Если менять PERSISTENT_VERIFY1→0 внутри живого процесса, прежний
  p.verify_state остаётся, пока строится transient state. В фиксированном
  профиле такого переключения нет; для runtime sweeps стоит явно освобождать
  старый persistent owner либо документировать фиксированный флаг.
- Constplanner добавляет verify bytes к retained_dense, затем max-ит всю
  сумму с прогнозом нового dense stage. Это не независимая сумма Vstate и
  нового dense scratch. Нынешний ordinary forward освобождает Vstate до
  allocation, поэтому реального coexist OOM в этом переходе нет. Если
  будущая политика сохранит Vstate при prefill, нужен отдельный budget owner.

Numeric pilot всё равно должен проверить K4/8, повторные rounds без
reconfigure, переход через PAD256, rewind/reconfigure и16K/32K. Данный review
не заменяет эти результаты. Production-код reviewer не менял.
