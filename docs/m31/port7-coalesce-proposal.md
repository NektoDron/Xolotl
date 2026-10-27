# Port7: full-width row coalesce

Чистый helper `m31_dspark_coalesced_rows.hpp` прошёл CPU и GPU proof:
360 сценариев, 2 052 864 F32 побитно, 26 unsafe metadata rejects. Результат:
`/mnt/ai/logs/m31-port/port7-coalesced-rows-component-v1/component-result.json`.
После PASS root разрешил DSpark owner интегрировать defaultOFF flag
`M31_DSPARK_COALESCE_INPUTS` для будущего P7-v10. Frozen P7-v9 не менялся.

Helper принимает 2–4 соседних n1 F32 view полной ширины одного contiguous
root. Работают и `[D,N]`, и фактический RMS/MUL `[D,1,N]`. Проверяются
`owner.ne[0]==D`, `nb[0]==4`, `nb[1]==D*4`, общий view_src, смещения по D*4
и bounds в owner. data pointers не читаются: component проверяет predicate
до allocation, когда все data=null. Возвращается contiguous `[D,count]`
`ggml_view_2d`, сохраняющий graph edge к producer; при отказе — nullptr без
allocation, затем прежний cont/concat fallback.

Важна отдельная producer-проверка: view_src показывает storage root, но
src[0] chain может включать inplace MUL/SCALE/CPY. Такой writer нельзя
пропускать прямым view root. Разрешена только metadata-only цепочка
VIEW/RESHAPE/PERMUTE/TRANSPOSE до общего owner. Проверены direct и nested
RMS/MUL producers; inplace arithmetic и view-over-write отвергаются.

Partial packed QKV с шириной меньше owner.ne0, другие parents, gaps, reverse,
неверные strides/bounds, один row и count>4 отвергаются. Component сравнивает
старые cont+concat и новый view на GPU, включая signed zero, subnormal, Inf
и NaN payloads, а также реальные RMS/MUL вычисления. Статически удалены
720 CONCAT и 1080 CONT nodes в матрице360 тестов. Это не измерение скорости.

Read-only review подключения owner: Graph bool defaultfalse, flag0 прежний
join body, flag1 попытка helper и прежний fallback при nullptr. Main/shared/head
получают policy; geometry key содержит flag, требуется VIEW1. Shared norm
OUTPUT/lifetime и output fences сохранены. Header SHA256
`4809f19341430002121af78c40dcd53d57a7852b35a072ec4ec332820ceb91c0`
совпадает с GPU proof. Profile-only node counts не меняют graph.

Модельная проверка flag0/1 и timing ещё требуются. Сокращение nodes может
перевести graph ниже границы chunk512 и изменить число cudaGraphLaunch;
эффект не равен сумме длительностей удалённых kernels.
