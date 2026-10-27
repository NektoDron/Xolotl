# Readonly review persistent DSpark verifier (prefix agent)

При прочтении `m31_dspark_verify_state.hpp` и `.inc` dangling graph leaf aliases
или неверного reuse geometry не найдено. Это source audit; numerical runtime
и VRAM budget должны проверяться отдельно.

Ключ — vector каждого query `nk=PAD(position+1,max(256,block))`. Размерvector
кодируетK, значения учитывают переходы256 отдельно для всех rows. При смене
K/nk сначала уничтожаются старые head иlayers, затем появляется новое состояние;
несколько геометрий одновременно не удерживаются. `position` иmask/bias остаются
inputs: обновляются перед каждымcompute. Константы `start` используются в
attention дляnk/views; при совпавшемnk не остаётся другого скрытого start-offset.
RoPE читает обновлённый positions input. Top_k/FA shapes также определяютсяnk.

Leaf tensors заимствуют immutable dense weights и mainKV/index buffers, живущие
до концаEngine. Другие скрытые состояния приходят через собственный input
каждого layer, а не как view чужого временного graph. QueryInputs vector содержит
только pointers на metadata собственногоGraph. Rewind очищает обычные graph/head
результаты иcaptured features, но не сохраняет borrowed hidden tensors вverify.
Его KV aliases остаются валидны; causal arguments рассмотрены в предыдущемreview.

Каждый backendcompute blocking; CPU graph compute иGPUread также завершены до
возврата. CPU executor сохраняет pointers на последний input vector, но между
вызовами они не используются: следующийforward заменяет их доcompute, destructor
не разыменовывает. Reset/release/destructor уничтожаютverify_state доdense weights,
KV/state buffers, backend иborrowedCPU pool. Graph destructor очищаетCUDAcaptures
перед освобождением metadata/buffer. Обратный порядок уничтожения членовLayer и
DSparkVerifyState также не освобождает tensor metadata раньшеGraph owner.

Две замеченные integration границы:

1. Обычный `Engine::forward` изначально оставлял persistentverify_state, а
   PrefillBudget не включал егоVRAM. Смешанный verify→rewind→forward удерживал
   60verify allocations одновременно сdense scratch. DSpark agent исправил:
   после успешногоforward preflight, доallocations вызываетсяverify_state.reset.
   Поэтому initialfree planner снова соответствует будущему compute без этого
   отдельногоowner. Rewind по-прежнему сохраняетverifycache для следующегоverify.

2. Verify не выбирает `set_dense_fusion` явно. Если optional decodefusion1
   включён, parity послеserial наследуетmode1, аspec сразу послеprefill —mode0.
   Это история arithmetic policy, а не danglingcapture. Рекомендация передана
   root/DSpark/dense owners: явно выбрать n1 policy или запретитькомбинацию.
   Owner исправил: verify вызывает `set_dense_fusion(true)` до работы сgraphs;
   при default fusionoption0 это no-op. Runtime сочетания всё ещё требуетгейтов.

`workspace_bytes()` корректно складывает все отдельные layer иhead gallocr
buffers. CPU executor/quant vectors — RAM, неVRAM. Однако это telemetry, а не
предварительный memorybudget. Long paritydriver сохраняет прежние prefill
allocators, тогда как specdriver вызываетrelease_prefill_workspace. На16/32K
persistentparity может упереться вVRAM из-за одновременной суммы; после сохранения
reference logits иrewind можно освободитьprefill scratch передverification.
Старые иновые geometry buffers не складываются: reconfigure освобождаетстарые.

Default persistent0 строит per-layer graphs с прежними n1 operations иосвобождает
каждыйowner после его слоя; head также по-прежнему собирается из отдельныхn1
MMV. Копияverified_logits илимитcaptured_rows — явныеAPI fixes, а не новаяGPU
арифметика. Ранее отмеченные проблемы getterhead иfeature_max_rows исправлены:
logits() теперь отдаётcached verifiedrawrows, capture уважаетmaxrows.
Нельзя заменять runtimeparity этимsource выводом.

Performance граница: p_min, меняющийK междуrounds, заставляет boundedone-entry
cache пересоздаватьвсеgraphs; для первогоизмеренияfixedK/p_min0 отделяетэтоот
GPU/CPU compute. Расширение доmulti-Kcache должно суммироватьвсеowners явно.

## Последующие owner fixes в live исходниках

`DraftFusionGuard` теперь временно выбирает unfused0 вокруг draft compute и
восстанавливает прежний target mode. При current0 или inherited-1 с
DISABLE_FUSION1 setter не вызывается, поэтому baseline не теряет captures.
На combo с targetfusion1 переключение0/1 всё равно инвалидирует captures;
для первых M3 замеров используется полностью unfused профиль.

Оба planner path теперь дополнительно учитывают retained verifier bytes
суммой с dense allocator; ordinary forward всё равно освобождает verifier
до своих allocations. Это консервативная оценка для смешанного API.

В новой live ветке batch_dense layout зависит от runtime mode. Найденный
недостающий mode в persistent key owner добавил как первый элемент:
`[batch_dense_mode,nk_per_query...]`; длина минус1 кодируетK. Пример с
возможным отсутствующим batched_input был лишь предположением review:
такого field в этой реализации нет. Риск заключался в reuse неверных graphops.
Frozen V3 до этих batch_dense изменений не затронут. После этого review
выбранные V4–V10 combined profiles прошли собственные selected128 numerical
и raw greedy proofs; P7-v8 также selected16/32K anchor+tail32. Это не
финальная gate20/GSM/tools приёмка нового canonical profile.
