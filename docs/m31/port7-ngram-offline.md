# Prompt/ngram lookup: offline оценка

Без Engine/GPU реализации оценены8 frozen ordinary greedy traces по128 токенов
из `port7-actual-decode-routes-v1`: manifest задаёт6 train и2 held. Ни om_bench,
ни held для выбора параметров не использовались. Unit
`codex-m31-port7-ngram-offline-v1` rc0, runtime0.21с, peak10.8MiB, shared build
lock. Результат `/mnt/ai/logs/m31-port/port7-ngram-offline-v1.json`; source snapshot
и SHA bindings — `.py` и `.manifest.json` с тем же basename.

Предложение использует только prompt и уже видимые generated tokens включая
sampled anchor. Для n3/4/5 выбирается последнее прошлое совпадение suffix;
continuation не выходит за известную историю. Проверены fixedN и longest3..N,
длины1/3/7, всего18 заранее заданных конфигураций. После proposal будущий обычный
trace применяется только как oracle для accept/reject. Target bonus учитывается:
matched drafts+1, ровно127 actually-fed rows на128 sampled tokens/case.

| longest3..N | K | train acceptance | train tokens/round | held acceptance | held tokens/round |
|---|---:|---:|---:|---:|---:|
|3|1|92.2%|1.639|98.3%|1.854|
|3|3|80.5%|2.281|96.5%|2.886|
|3|7|67.8%|2.498|96.2%|3.387|
|4|1|91.3%|1.632|98.3%|1.854|
|4|3|79.8%|2.275|96.5%|2.886|
|4|7|68.2%|2.507|96.2%|3.387|
|5|1|90.7%|1.628|98.3%|1.854|
|5|3|79.8%|2.275|96.5%|2.886|
|5|7|67.5%|2.507|96.2%|3.387|

Train-only выбор — longest3..4,K7: train304 rounds/762 fed rows, proposed672,
accepted458; held75 rounds/254 fed, proposed186, accepted179. Verification
expansion1.281 train/1.028 held. Tokens/round — идеальный upper bound сокращения
числа round; это не measured throughput и не обещание ускорения.

Существенное ограничение: frozen trace сырого продолжения сильно повторяется.
Held `code_06_512` имеет только2 уникальных ID на128 токенов, `math_07_768` —9;
train `ru_00_64` —3 и `en_01_127` —3. Проверенные EOG IDs200004/200005/200020
не объясняют эти циклы; это не просто tail после EOS. Высокий lookup acceptance
опирается на повторение и не доказывает перенос на нормальные chat/GSM/tools.

Для контекста, DSpark V5 на другом corpus3324,p_min0 даёт K1 acceptance70.7%,
1.707tokens/round; K3 —54.5%,2.612; K7 —37.4%,3.556. Сопоставление не является
matched-prompt benchmark: у восьми short cases нет frozen DSpark предложений.
На имеющихся данных lookup нельзя отвергнуть по acceptance, но его инженерная
приёмка и заявка35+t/s не обоснованы. Реализация Engine не начата; направление
остаётся неподтверждённым до качественных autoregressive chat traces с теми же
prompt IDs и paired target verification.

На matched corpus3324 фиксированные train-chosen longest3..4/K3,K7 проверены
без подбора: promptSHA11cc63f6…, ordinary128 SHA41464429… совпадают с DSparkV5
sweep. `port7-ngram-matched-v1.json`, unitrc0, CPU84ms после exclusive PP окна.
При нормировке обеих схем к127 fed rows:

| K | lookup rounds | lookup accepted/proposed | lookup tokens/round | DSpark rounds | DSpark tokens/round |
|---|---:|---:|---:|---:|---:|
|3|64|63/87|1.984|49|2.592|
|7|56|71/141|2.268|36|3.528|

Lookup verification expansion1.189/1.551 против DSpark1.520/2.213. Conditional
acceptance lookup выше, но покрытие хуже и полезных accepted tokens меньше:
выигрыша по round reduction на matched prompt нет. Это не measured timing;
реализацию по этим данным не начинаем. Штатные PPv3 HTTP choices хранят parsed
text, а observer — полные prompt IDs. Полные generated IDs не записаны, поэтому
для ngram оценки этих ответов ретокенизация текста не используется.
