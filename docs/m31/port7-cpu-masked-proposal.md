# CPU masked-RHS reserve для DSpark cached experts

Резервный вариант, **не применён** к production/liveV4.
Base CPU header304c128b…; patch `tools/m31/port7_cpu_masked.patch`;
изолированный candidate13b9b0e9… в build-port7-cpu-masked.

API forward_masked принимает те же weight/scales/x/full IDs/full route weights,
затем uint8 cpu_mask[N,used] (1CPU miss,0GPU hit). Возвращает уже взвешенные
F32 terms[N,used,embd] в исходных rank slots, GPU slots равны+0. GPU helper
cached владельца возвращает hit-major weighted terms по {token,rank,slot}.
Сначала восстановить slots, затем складывать строго rank0→rankused−1.
Предварительно суммировать CPU/GPU части нельзя: меняется ассоциативность.

Masked callback quantizes только tokens с CPU misses; gate/up/OAI/down и
квантизация down только выбранным назначениям. Группировка/weighted partition
считает активные назначения, а не полный N*used. All-hit возвращает нули до
allocation/compute CPU graph. Обычный forward сохраняет исходное compute тело
и сбрасывает masked state. Никакие Engine/GPU/live CPU files не менялись.

Проверка `port7-cpu-masked-proof-v1/receipt.json`: PASS,2880masked batches,
10800token rows,25200weighted-term rows и2880последующих обычных forward
побитно. K1/2/4/8,used1/2/4,3routepatterns,8masks,CPU pair modes0/2;
реальные IQ3/IQ4/NVFP4 weights5слоёв. GPU slots+0, CPU slots сверены с
независимыми CpuFusedDecode contributions, merged outputs побитны.
All-hit bypass доказан armed CPU backend abort callback: all-hit проходит,
контрольный обычный graph abort-ится; после callback reset обычный путь
снова совпадает. Есть signed-zero и non-associative rank fixtures.

Это CPU компонент: GPU terms в этой проверке взяты из CPU reference.
Actual GPU worker/двухjoins overlap, model parity/quality и speed не заявляются.
GPU batched proposal разрабатывается cached агентом отдельно. Интеграция
только по решению root после результатов model batched-MMV speed.

## Интеграция после frozen V5

Root через владельца DSpark одобрил перенос проверенного header в live. Применён SHA256 `13b9b0e941444c3a4eca9a1a4d636f1058b5d2d3a666950e7c8926f7377dae63`; API стабилен, исходный unmasked путь и pair modes сохранены. Frozen V5 и прежние receipts не менялись. Совместный компонент CPU/GPU прошёл 39 случаев и 3 391 488 F32 terms побитно; это не измерение скорости модели.
