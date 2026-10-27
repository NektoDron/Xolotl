# Port7: один routed metadata D2H — отдельный proposal

По поручению владельца DSpark подготовлены только новые файлы:

- `tools/m31/m31_dspark_packed_routes.hpp` — graph packing, layout, один get,
  unpack без арифметики над F32 norm/residual/weights.
- `tools/m31/port7_packed_routes_check.cpp` — standalone CPU/CUDA компонент.

На этапе отдельного компонента Engine, frozen V6/V7, общая CMake и profiles
не менялись. После CPU/GPU PASS владелец DSpark интегрировал default0 путь
для P7-v8; модельная проверка выполняется root/validation.
Standalone build rc0: `build-port7-packed-routes/v1`; CPU240cases/2 892 168F32/2520IDs bitproof PASS, включая NaNpayload/signedzero/strides и5invalid unchanged checks. GPU runtime proof PASS: те же240cases/2 892 168F32/2520IDs побитно, включая edge values и5invalidunchangedchecks. Default integration должна быть flag0,
например `M31_DSPARK_PACKED_ROUTED=0`, сохраняя прежние Graph::read вызовы.

## Формат и арифметика

Входные matrices: norm/residual F32 `[D,N]`, weights F32 `[U,N]`, IDs I32
`[U,N]`, N1..8. Формат final F32 tensor `[W,N]`, W=2D+2U; внутри строки token:

`norm[D] | residual[D] | weights[U] | numeric_cast(ids)[U]`.

Три concat по axis0 и один I32→F32 cast, без изменений router/weights/norm/FFN.
Стандартные входы уже contiguous; helper добавляет cont только для strided
view. При D6144/U4/N8 W12296, размер393472 B. Предыдущие четыре metadata reads
переносят ровно столько же байт; экономятся три synchronous API transitions.
Shared196608 B остаётся отдельным после shared-overlap/CPU FFN. Вместо пяти
синхронных reads всего два (packed metadata + поздний shared).

Для0..127 numeric I32→F32 и обратный cast точны. CPU unpack использует memcpy
для F32 полей, сохраняя все биты; IDs проверяются finite,0≤id<experts и целыми
перед любым изменением output. Скалярный IDs cast не относится к model math.
Тензорные данные norm/residual/weights не проверяются на finite и не проходят
через FP операции: это перенос исходных битов. GPU concat behaviour будет
проверен компонентом, включая signed zero/subnormal/Inf/NaN payloads.

CUDA support проверен в локальном ggml источнике:
`build-m31-cpu/_deps/strata_llamacpp-src/ggml/src/ggml.c:3666` — ggml_cast=CPY;
`ggml-cuda/cpy.cu:609` — I32→F32 и contiguous/strided implementations;
`ggml-cuda/ggml-cuda.cu:5928` — supports_op принимает эту пару;
`ggml-cuda/concat.cu:7` и:69 — axis0 копирует каждую token row.
Компонент также вызовет backend supports_op для каждого реального узла.

## API и условное подключение (не применено)

`dspark_pack_routes(c,norm,residual,weights,ids,experts=128)` возвращает layout
и packed tensor. `dspark_read_packed_routes(pack,reusable_scratch,host)` делает
ровно один ggml_backend_tensor_get, затем unpack. Host поля — отдельные vectors
norm/residual/weights/I32ids, совместимые с текущим CPU callback. После unpack
можно move поля в прежние local vectors; либо хранить capacities в state.
`dspark_unpack_routes` отдельно пригоден для raw readback validation.

КРИТИЧНЫЙ lifetime при shared_overlap: mainGraph должен по-прежнему отмечать
исходный norm как OUTPUT: `b.allocate({packed_metadata,packed_norm})`. Shared
Graph.borrow(norm) не виден liveness главного gallocr; если оставить только
packed_metadata, storage norm может быть переиспользован поздними concat
узлами до асинхронного shared compute. Packed копия norm на CPU этого не
исправляет — shared FFN использует исходный device norm. Нельзя заменять его
на стянутый из packed non-contiguous view без отдельной проверки FFN layout.

Порядок: compute mainGraph → packed get/unpack → enqueue async sharedGraph →
GPU cached exact enqueue/CPU misses либо обычный CPU FFN → shared join/read →
original-rank FFN output + shared + residual. На shared callback ошибках
сохранить существующий SharedDrain. Packed read/CPU unpack входят в read_ms;
отдельно желательно pack extra compute и CPU unpack diagnostic counters.

Три concat добавляют GPU launches и временные buffers. Не обещать чистый
выигрыш9ms: измерять saving read_ms против добавившихся compute_ms и total.
При N8 сумма трёх concat output размеров1 180 032 B + IDs cast128 B, до
возможного reuse gallocr; persistent57-layer owner sum может увеличить VRAM
на десяткиMB. Это нужно включить в фактический workspace sum, не в max.

## Подготовленный компонент

Standalone CLI: `port7_packed_routes_check cpu|cuda REPORT.json`. CUDA требует
M31_CARD_SESSION=1, значит только через существующую очередь aictl/cardlock.
Замороженный GPU ELF SHA256 `189d5bda500d2005263997be55ef3ab7d6ee1478beb357b377ed3ab54e13b67d`; NEW job `port7-packed-routes-component-v1` передан validation, сам не запускал.

Матрица240 кейсов: D1/31/256/257/6144, все N1..8, U1/2/4, contiguous и strided
источники. Все IDs0..127 покрываются. F32 значения включают ±0, subnormal,
±Inf, максимальное конечное, qNaN/sNaN payloads. Gallocr используется по
настоящему; norm помечен output и читается после packing для lifetime check.
Один get вызывает helper, дополнительно тест читает retained norm только для
проверки. Unpack сравнивает все F32/I32 побитно с исходными token-major rows.
Пять неправильных buffers/IDs должны отвергаться без изменения предыдущего
host output. Это проверка переноса данных, не модели и не performance.

После component PASS: same-ELF flag0/1 raw rows + greedy, особенно N3/5/7/8
с shared_overlap0/1 и GPU cached0/1; затем чистое окно component/model timing.
До этих проверок default0 и никаких изменений frozen артефактов.

## После component PASS

Владелец DSpark применил Engine integration под default0; cached agent сделал
read-only review includes/state/verify. Geometrykey содержитflag, приoverlap1
mainGraph удерживает normOUTPUT; sharedGraph уничтожается доmainGraph.
Reused hostvectors не alias hidden, следующийunpack идёт только послевсехjoins.
Приflag0 прежнийпорядокget norm/res/shared/weights/ids сохранён. Дополнительные
concat buffers находятся вmainGraphgallocr и входят вworkspace SUMодинраз.
FrozenV7 не менялся; новаясогласованнаясборкаP7-v8 проверяется root/validation.
Это source review ACK, не ещё model/quality/TG acceptance.
