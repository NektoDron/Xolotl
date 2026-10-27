# Port7: default worker/shim timing, source-only review

Исходники не менялись, новые модели/сборки/profiling не запускались.
Сравнение — accepted V8 gitfaeb1332 против currentHEAD dd4bc35b и
canonical-v2/source-tree/tools/m31/m31_serve.cpp. Последние два C++
исходника совпадают. Machine receipt:
`/mnt/ai/logs/m31-port/port7-default-worker-timer-source-audit-v1.json`.
Это проверка семантики wrapper, не вывод о регрессии скорости. Решение
о сравнительной скорости должно опираться на fresh V8→canonical control.

Actual canonical default om запускал Python shim SHA
`e87e70e1700cc6a1caf7e6355228ffdc3787b13b0640253967d6e9807666f4bf`,
который **побайтно совпадает с V8 serve_m31.py**. Current HEAD shim имеет
новые prefix/DSpark параметры, но этот om их Python реализацию не
использовал. Server argv не содержит --prefix-cache/--dspark-model.
Поэтому Python additions не являются объяснением slowdown этого замера.

В C++ default session=null. Source comparison проверил механическое
сведение новой token loop: заменить if(dspark_session)/else на ordinary
forward+logits и убрать prompt_cache.clear только из ветки cancellation.
После этого loop совпадает сV8 с точностью до whitespace. Последовательность
choose→decode_end→predicted++→stop check→token send→ACK→forward+logits
осталась прежней. Порядок serialized ACK и единственный forward/head
на fed token не менялись; дополнительных RPC/JSON per-token exchanges нет.
Новая работа в штатном loop — проверка null unique_ptr. Capture, feature
getter, release_prefill_workspace, verify, sidecar init не вызываются.

Python token branch (piece/decode/clipped stop/ACK/PEG mapping) уHEAD также
текстуально совпадает сV8. Cache metadata разбирается только на prefill/done,
usage_counts — после окончания engine timer. Политика M3 вычисляется один
раз на запрос, при пустом dspark_model сразу отключена. Default cache
пуст/выключен, холодный engine.reset по-прежнему вне PP timer.

Первый choose относится кPP, decode_start идёт сразу после него. Дальнейшие
choose входят вTG. decode_end фиксируется после последнего choose, перед
последним token send/ACK. TG=1000*(predicted_n−1)/predicted_ms в обеих
версиях. Для length256/ignoreEOS/interactive это256 sample и256 ACK,
255 forward+head. ACK/transport между первым и последним sample входят в
обе TG метрики; последний transport/ACK не входит в обе. Final JSON и
prefix/DSpark finish metadata не меняют этот default timer.

При prefix=false локальный prefill_event содержит те же поля, что старый
временный JSON. Никакой новой initial hidden/head работы после decode_start
в default worker не найдено. Это не означает побитную или performance
эквивалентность всех Engine/backend/build изменений: clocks/NUMA/page
state, backend kernels, library flags, graph caches и allocator layout этим
review не измеряются. Worker/shim source audit не заменяет paired control.
