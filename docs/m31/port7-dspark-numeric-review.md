# Независимый review DSpark target verification

Владелец проверки — routed_cpu,27октября2026. Это анализ кода,
не утверждение о результате ещё не выполненного GPU parity прогона.

## Почему обычный batch2..8 может отличаться от serial n1

1. `attention()` выбирает joined QKV и index QK только для n1. При n>1
   строятся отдельные проекции и меняется CUDA dispatch MMV/MMQ/cuBLAS.
2. MSA n1 собирает sparse KV в score-sorted порядке выбранных блоков.
   n>1 оставляет KV в физическом порядке и применяет top-k mask. Даже при
   одинаковом множестве блоков порядок flash-attention reduction иной.
3. Score GEMM имеет форму [nk,index_heads*n], а не [nk,index_heads].
   Последние биты могут изменить дискретный выбор блока около границы top-k.
4. CPU routed n1 использует принятый CpuFusedDecode с vec_dot(nrc=1).
   CPU mul_mat_id для n2..8 группирует назначения и при>=8 может выбрать
   другой tiled kernel. Порог GPU экспертов>8 устраняет только переключение
   routed FFN на GPU, но не перечисленные отличия.

Проверка meanKL/top1>=99% не выполняет отдельное требование «greedy DSpark
совпадает с обычным greedy». Для DSpark нужен100% top1 на выбранных
продолжениях, затем exact generated IDs serial vs spec. Parity mode driver
сохраняет top1_different даже если process rc0; проверяющая обвязка должна
читать поле, а не считать rc0 доказательством.

## Вариант exact layer-major verification

Для каждого target слоя K входных residuals уже известны после предыдущего
слоя. Dense/attention проходит K запросов по одному, с теми же n1 графами,
KV causal mask, sorted gather и arithmetic dispatch. Собранные K норм,
маршруты и веса поступают в `CpuBatchedVerify`: группировка по expert,
внешний цикл по строке весов, внутренний — по token assignments. Исходные
CPU quant/from_float/vec_dot(nrc1)/scales/SwiGLU/rank sum сохраняются.
После CPU batch outputs переходят к следующему слою. Это отдельный
scheduler; обычный Engine::forward(batch) ему не эквивалентен.

Новый компонент `cpu_batched_verify.hpp` принимает тот же набор трёх весов
и scales, затем token-major x/ids/weights, возвращает token-major output.
Borrowed persistent CPU pool; maxK8; v8 n1 путь не меняет. Компонентный
битproof должен покрыть реальные IQ3_XXS/IQ4_XS/NVFP4, K1/2/4/8,
1/2/4 активных эксперта, одинаковые/частично общие/разные маршруты.
Его CPU-only тест не закрывает полную GPU scheduler equivalence.

Экономия bandwidth определяется union экспертов между K токенами.
При независимых4 routes из128 дляK8 union может быть близок к32 —
амортизации чтения почти нет. Нужны реальные route trace и распределение
acceptance, а не оценка ускорения какK автоматически.

## Default isolation и lifetime

При пустом draft_feature_layers captured_rows=0, capture method не вызывается;
добавляются только пустые владельцы графов и проверки. Найден и исправлен
DSpark агентом teardown: draft_capture_graphs должны освобождаться явно в
~Impl перед backend free. Rewind сбрасывает все graph-backed outputs и
действует только на ранее завершённый prefix. Stale future KV/index rows
маскируются перед block-max и attention; для гейта нужны ветвления на128/256
границах, длины>3000 и16K/32K, где реально выбираются MSA блоки.
