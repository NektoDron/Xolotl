# Readonly review strict greedy comparator (prefix agent)

Проверены `port7_greedy_compare.py`, новый `port7_sweep_compare.py` и описанные
V2/V3 positive artifacts. Код comparator не менялся этим агентом.
Это source review, а не повторный запуск полного сравнения.

Первые три gaps переданы owner и закрыты в прочитанном обновлённом коде:

1. Ordinary reference ранее не отличался от spec/self-compare. Теперь role
   проверяется по явному mode и serial record из собственного summary;
   reference/candidate output directories должны различаться.
2. Backend ранее отсутствовал в target settings. Теперь требуется явный CUDA
   backend и сравниваются остальные target CLI параметры.
3. Ранее можно было передать лог другого command/arm. Теперь argv и rc0
   подтверждаются atomic job result, selected log связан сего записью.

ELF SHA сверяется сrun manifest и build receipt. Frozen Engine иdriver source
hash сверяются сreceipt. Prompt IDs имеют hash/размер; выбранные chunks должны
покрывать prompt непрерывно ииметьnk. Generated IDs привязаны кcommand out и
case label, длинастрого соответствует --generate. Передscoring сравниваются
source/model path/precision/prompt/chunks/target settings/explicitENV/unset.
F16 KV подтверждается source regex для нынешней архитектуры без runtime
precision override; будущая alternate precision branch потребуетболееявного
runtime descriptor, одного наличияF16 allocation вsource тогдабудетмало.

Per-arm adapter проходитstructured prefill_receipts втомжеordinal order,
передаётoutputcase ипроверяетcontext/maxchunk/chunks против выбранногоruntime
traversal. Это существенно сильнее произвольного выборафайла generatedIDs.

Оставшиеся границы, переданные owner:

- `--summary` adapter должен быть связан сcommand --out/summary.jsonl. Иначе
  одноимённый foreign summary ссовпадающимиchunks может датьчужие diagnostic
  tps/arm параметры при верныхgeneratedIDs. Нужна path/SHA equality. Dict specs
  также не должен молча скрывать дубликаты одногоcase.
- Profile сейчас требует48key, но не проверяетихvalues/requiredunset против
  passed profile. Ref/candidate explicitENV должны совпасть, однакоэто ещёне
  attestation всего effective runtime environment/wrapper effects. Если именно
  это и естьscope, егонадо явноназывать, анеобещатьполныйruntimeENV.
- `source_receipt_sha256` вmanifest опционален сfallback ктекущемуdigest. У
  историческогоV2matched manifest поля нет, уV3pilot есть. Frozen artifacts
  переписыватьнельзя; новымfinaljobs лучше требоватьanchoredhash явно, legacy
  evidence обозначать отдельно. ActualELF/source matching всёравнопроверяется.

Old wronggeometry reference остаётсяinvalid доscoring; selected128outputs
даже припрошедшемpreflight не заменяютfullgate20, longcontexts, GSM/tools
или om. Hash equality/current source audit сами посебе не доказываютгейты
другойсборкиилидругогопрофиля.
