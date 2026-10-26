# Аудит итоговых артефактов и контрактов

26.10.2026, контрольная точка около20:40UTC, cache_ranking. Только чтение
JSON/TSV/manifest и небольших файлов: logits, модельные payload и ELF заново
не читались. Новых inference/замеров/сборок нет. Числа ниже — проверка уже
существующих результатов, не повторный benchmark.

## Найденные расхождения документации

Неверных измеренных чисел в проверенных таблицах не найдено. Есть недообновлённые
или неоднозначные current claims, которые нужно синхронизировать перед final:

1. `STATUS.md:301` всё ещё «full candidate128+seed128 ещё выполняются»;
   оба `gpu-candidate-*-code128-compare-v1.json` уже pass, worker/result rc0.
   Последнего append с GPU candidate speed21.57/21.18, PP46.2 и raw bytecmp
   в проверенном хвосте STATUS ещё нет. Исторические записи можно сохранить,
   но нужен актуальный завершающий checkpoint.
2. `docs/m31/M2.md:45` всё ещё «Seed API ещё проходит интеграцию», хотя
   direct F32 probe, real seed20/code128 и measured GPU candidate уже прошли.
   Полный20 для позднего host32/worklist/reads/seed кандидата не запускался:
   его приёмка остаётся коротким128-case, не старым frozen gate20.
3. `HANDOFF.md:33` называет последним speed окном CPU `port4-low-ack-speed-v1`,
   хотя таблица43–44 уже включает более поздний GPU candidate. Нужна явная
   ссылка и на `port4-gpu-candidate-speed-v1/`. В `M2.md:20` «Медианы Xolotl»
   21.325/124.55 относятся именно к CPU routed; GPU candidate имеет отдельные
   21.375/46.2. Общие «native ниже8.59%/70.69%» (`HANDOFF.md:49`) тоже относятся
   к CPU routed, не GPU candidate (у него8.38%/89.13% ниже ref).

Ранее в начале аудита M2 ещё называл full GPU gate выполняющимся, а HANDOFF —
candidate128/seed128 непроверенными. Координатор обновил эти строки во время
аудита; в повторном чтении около20:39 они уже отражают принятые результаты.

## Сверенные результаты

| Объект | Проверенное доказательство | Вывод |
|---|---|---|
|Исходный CPU-strict M1|`suite20-compare-v1.json`, `job-suite20-v1.result.json`|20 cases/28723rows, top1=1, mean/maxKL0, per-case maxabs0, все process rc0|
|Frozen exact-GPU M1 против CPU oracle|`gpu-integrated-suite20-c2048-d32-compare-v1.json`, соответствующий job result/unit|20/28723, top1=1, KL/maxabs0, scope `M1-complete-logits`, rc0|
|Побайтовый полный gate|`gpu-fullgate-bytecmp-v1.json`|20×cmp rc0,22,985,753,088байт = **5,746,438,272 F32 значений**, не5.746Bбайт|
|Поздний candidate seed0|`gpu-candidate-code128-combined-compare-v1.json`|128rows, top1=1, KL/maxabs0, scope `diagnostic`, job rc0|
|Поздний candidate seed20|`gpu-candidate-seed20-code128-compare-v1.json`|128rows, top1=1, KL/maxabs0, scope `diagnostic`, job rc0|
|Tools23|`server-tools23-v3/m31-port4-tools23-v3.summary.json` и23 строк results|20/23,20 tool+3follow,453.891с,0length/0leaks, один stochastic run|

Все40 manifests полного frozen GPU gate просмотрены без logits:20+20 cases,
28723 positions на сторону, все rows перечислены полностью, vocab200064.
Обе стороны chunk2048/context4096/decode_tail32/threads64/F16/nofusion.
У oracle `oracle_op_offload=false`, у native `gpu_cpu_quants_requested=true`,
arena170/cache6/min32. Поэтому это strict CPU-quant oracle contract; min32
оставляет однотокенный decode на CPU. Более позднему candidate автоматически
этот gate не приписывается — HANDOFF17–20 это корректно оговаривает.

Tools rows действительно20 correct, failures11/18 иfollow1, finish не length,
leak=false. Summary явно содержит `gsm8k_run=false`, `tool_execution=false`,
temperature1/top_p.95/max_tokens6000/default thinking.20/23 против ref19/23
не является доказательством статистически значимого улучшения. В M2/HANDOFF
один стохастический прогон оговорён корректно.

## Скорость и условия сравнения

| Артефакт TSV | TG4K, два прогона | PP4K, два прогона | Медианы TG/PP |
|---|---|---|---|
|`port4-low-ack-speed-v1/speed.tsv`|21.34/21.31|124.5/124.6|21.325/124.55|
|`port4-gpu-candidate-speed-v1/speed.tsv`|21.57/21.18|46.2/46.2|21.375/46.2|
|Лучший1GPU ref REPORT|23.33|425 (32K433)|—|

Оба новых session spec реально запускают неизменённый `om_bench.py` с
`OM_OFF={"reasoning_effort":"low"}`, warmup,4K,4K. SHA клиентского скрипта
совпадает с записанным manifest. Все четыре строки —256 generated,
finish=length; prompt counts4348/4349 и4350/4352 включают random prefix/template.
HTTP/card job result rc0; у speed launcher `measurement=true`, то есть
exclusive build lock. Unit CPU-low и GPU-candidate завершены rc0.

Это matched **request body**, не одинаковая memory/placement конфигурация:
native F16 KV/ctx8192/chunk2048/t64; CPU arm без routed GPU cache, GPU arm
arena170/cache6/host32/seed20/min32. Ref Q8_0KV/ctx65536/ubatch2048,
3 целых слоя экспертов GPU. Эти различия в актуальных M2/HANDOFF указаны.
Ранние enable_thinking=false окна явно переобозначены диагностическими
(`STATUS.md:263`, `M2.md:22–23`, `HANDOFF.md:50–51`); ошибки считать их strict ref
comparison в финальном текущем описании нет.

GSM8K100 и полные native speed16K/32K **не измерены** и не заявлены как
готовые. В сохранённом plan есть будущая quality команда100, но реальные
HTTP session clients содержат smoke+speed, не запуск GSM. Не путать plan с
job result. Точно так же preparation/review-only фразы в старых manifests
описывают подготовку плана; факт выполнения подтверждают отдельные result/TSV.

## Stock GGML CUDA: отдельный диагностический контракт

Проверены `job-gpu-ggml-gpuoracle-suite20-c2048-d32-v1.json`, code128 job и
`gpu-ggml-gpuoracle-policy-v1.json`. GPU-агент подтвердил: это frozen candidate
ELF, `M31_GPU_CPU_QUANTS=0`, host/sort unset, async+reads+seed20; reference
oracle запускается с `--oracle-op-offload1`. CPU golden и strict full20
артефакты сохранены отдельно и не подменяются.

Sidecar содержит explicit contract, placement/env, seed/min32 и SHA6 малых
job/runner/launcher файлов; все6 SHA совпали при этом аудите. Native/oracle ELF
SHA согласуются с прежними frozen manifests; сами ELF повторно не хешировались.
У готового stock code128 result: top1=1/KL0, scope `GgmlGPUoracle-code128`,
contract «not M1 CPU-exact acceptance». Full20 ещё идёт; partial oracle
manifests имеют `oracle_op_offload=true`, n2048/tail32/F16/nofusion. Full result
ещё отсутствует, поэтому pass полного stock gate здесь не заявляется.

Job wrapper после сравнения явно меняет scope на `GgmlGPUoracle-complete-logits`
и добавляет contract. Это необходимо: generic `compare_suite(require_suite=True)`
сам по себе присваивает `M1-complete-logits` и не проверяет placement policy.
Перед будущим fastmode benchmark нужен собственный завершённый stock gate
и отдельное окно; быстрый mode не получает strict CPU M1 по названию oracle.
В policy отмечена известная stockGPU-vsCPU разница code128: top1.78125,
meanKL.77857. Fastmode speed на момент аудита ещё не запускался.
