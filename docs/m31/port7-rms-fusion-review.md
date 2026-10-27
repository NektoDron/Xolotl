# Независимый review RMS-only fusion / empty-map fastpath

Readonly review `ggml-rms-only-fusion.patch` SHA256
`b9dc502a…` и apply logic после STABLE владельца dense. Проверены новые
atomic/map ветки, setter/free, matcher/graph optimizer, Engine policy и
существующий DSpark DraftFusionGuard; runtime не редактировался и новые
jobs/сборки этим агентом не запускались. Backend compile/model correctness и
speed остаются отдельными receipts владельцев dense/validation.

В рассмотренной схеме не найдено ошибки map lifetime, graph-capture policy
или некорректного расширения fusion matcher. Требование backend API прежнее:
caller сериализует setter/compute/free для одного backend. Mutex защищает
карту, но не делает одновременную смену policy и compute допустимой.

## Empty-map и ownership

Atomic count начинается0; acquire load0 возвращает inherited mode−1 без mutex.
Setter сначала проверяет текущий mode, при реальном изменении drain/clear
CUDA graph cache выполняется ДО смены карты и ВНЕ её mutex. Insert/erase и
release store нового map.size() происходят под одним mutex. Backend free
делает erase+store до delete context. Count не меняется отдельно от карты.

При documented сериализации same-backend состояние опубликовано до следующего
compute/getter. Параллельный другой backend либо видит count0 и собственный
inherited policy, либо ждёт mutex и ищет свой pointer; уничтоженный pointer не
разыменовывается. Pointer reuse не наследует старую entry после корректного
free. Порядок cache clear вне map lock избегает рекурсивного mutex при cleanup.
No-op setter не сбрасывает совпадающие captures.

Default OFF теперь платит atomic load вместо ранее найденного обязательного
mutex/map lookup. Это не буквальное отсутствие instruction overhead. При
наличии override хотя бы у другого backend В ТОМ ЖЕ процессе count>0, и getter
default backend снова берёт mutex — fastpath именно global empty-map.

## Mode2 и graph lifetime

`fusion_disabled` трактует mode2 как enabled. Новый early branch допускает
только stock `RMS_NORM → MUL` через прежний `ggml_cuda_can_fuse` и stock
`ggml_cuda_op_rms_norm_fused`; при несовпадении сразу return0. Broad matchers
RMS+MUL+RoPE/ADD, FFN/weighted reduction/topk и прочие не исполняются.
Те же topology/type/memory-range/output правила сохраняются внутри can_fuse.
Final MUL fence остаётся output, промежуточный RMS не объявляется лишним output.
Exact-MMV marked node по-прежнему отклоняется ДО mode2 ветки.

Graph optimize при mode2 получает disable_fusion=true: broad allocator rewrite
и reorder не подменяют selective matcher. Смена2↔0/1/−1 сбрасывает backend
capture cache до смены policy; stale CUDA replay под новым mode не сохраняется.
Сами tensor graphs могут жить дольше — новая запись захватывается при следующем
compute с актуальной policy. Engine destructor возвращает borrowed backend
previous mode после release собственных graphs/workspace.

## Phase и DSpark

Найденный scope caveat принят координатором и передан владельцу: Engine выбирает
policy по n==1, а не семантической фазе запроса. Поэтому n>1 forward unfused,
но prompt1 или хвост prefill chunk1 получает тот же RMS2, что и generation.
Нельзя писать «любой prefill выключен»; корректно «single-token forward».
Actual4K/16K/32K recipes в текущей серии имеют chunks>1. Владелец сообщил,
что комментарии исправлены; отдельный phase API сейчас не добавляется.

DraftFusionGuard читает current2, временно задаёт0, и restore(previous2)
проходит новый setter диапазон−1…2. При уже0 или inherited disabled он не
переключает режим. Явного пропуска restoration для mode2 не найдено.
Engine verify также использует target single-token policy. Совмещение M3 с
новым RMS2 этим source review НЕ получает model numerical/speed приёмку:
переключения0↔2 дополнительно сбрасывают captures и могут быть медленными.
DSpark в этой волне не расширяется; production profile не должен включать
непроверенную комбинацию только потому, что API её корректно восстанавливает.

При первом чтении no-CUDA compiled Engine молча игнорировал RMS ENV, так как
проверки находились внутри M31_ENGINE_CUDA. Владелец после сообщения добавил
явный CPU-only reject. CUDA build с CPU backend уже отклонял его require(cuda).
Это guard invalid configuration, а не изменение default math.

## CMake / canonical четыре patches

Порядок подтверждён source review:

1. `tools/xolotl/ggml/xolotl-ggml.patch` — общий vendor patch.
2. `tools/m31/ggml-decode-fusion.patch` — API/side map.
3. `tools/m31/ggml-exact-mmv.patch` — marker/dispatch API.
4. `tools/m31/ggml-rms-only-fusion.patch` — mode2/atomic empty-map.

`port7_canonical.py PATCHES` содержит именно эти четыре пути и на private
чистом pinned checkout выполняет git apply --check, затем apply для каждого.
Таким образом canonical не полагается только на reverse-check чужого дерева.
Top-level CMake применяет общий patch до M31 helper; managed helper применяет
последовательно2→3→4. External vendor при недостающем patch fail-closed и не
мутируется. STRATA_M31=OFF возвращает из helper без дополнительных patches.

Ранний reverse-check последнего patch необходим на повторном configure:
последний patch изменяет header/setter текст первого, поэтому прежний reverse
check первого на финальном дереве уже не обязан проходить. Final reverse
успешен → helper выходит, не пытаясь применить старый patch повторно.
Fresh managed, API+marker без RMS, финальное повторное configure проходят
логические ветки корректно. Владелец включил их проверку в private backend-v2.

Ограничение: последний reverse-check подтверждает все RMS hunks и часть
prerequisite context, но не гарантирует byte identity КАЖДОГО раннего API hunk
в произвольно вручную изменённом external vendor. Комментарий «proves
prerequisite API/marker context» верен именно как context, не как полный
integrity audit. Canonical fresh-base apply+vendor hashes закрывают нужный
нам reproducible scope; это не блокер canonical freeze.

Материальных новых blockers после принятого phase caveat не найдено.


Последующая provenance проверка canonical v2 подтвердила применение всех
четырёх patches, source/vendor stability, CPU/CUDA compile и CPU-backend final
fixtures (49 stages rc0). Это не отдельная mode2 GPU math/performance приёмка;
selected V11 RMS/lookup model diagnostics ведутся владельцем dense отдельно.
