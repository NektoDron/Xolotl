# M31: рейтинг экспертов и бюджет VRAM

26.10.2026, port4, субагент cache_ranking. Движок и GPU-кеш не менялись.
Новый `cache_calibrate.cpp` читает `Config.route_observer` после исходного
router и пишет только выбранные IDs/веса; повторная маршрутизация, pruning и
переквантование отсутствуют. Калибровка использует принятый CPU-routed путь,
CUDA dense, nofusion, chunk128/threads64/F16 KV.

## Реальные данные

`codex-m31-port4-cache-calibration-v1` rc0, штатный card_session/aictl,
down и точечный rollback выполнены. Логи в `/mnt/ai/logs/m31-port/`:

- `cache-routing-v1.jsonl`: 4,938,344 байт, 960 токенов, все57 routed слоёв.
- train: RU64, EN127, code128, math129, всего448 токенов/102144 маршрута.
- held: другие RU/EN/code/math промпты, первые128 токенов каждого, всего512.
- `cache-inputs-v1.json`: исходные и усечённые IDs, SHA256, train/held split.
- `cache-ranking-v3.{json,bin}`:7296 экспертов, частота, сумма route weights,
  model/tensor/input identity и held-out replay. Порядок: частота, weight sum,
  затем layer/expert для разрешения равенства.
- Бинарный заголовок: little endian `<8sIIII>`, magic `M31RANK\0`, version1,
  layers60, experts128, count7296; далее32 байта identity SHA256 и7296 uint32
  глобальных IDs `layer*128+expert`. Плотные0..2 не входят.

SHA256 содержимого пяти GGUF взяты из уже проверенного ref
`integrity-ud-original.json` (revision02c81e4...), заново150+ГиБ не хешировались.
Сохранены текущие paths/sizes/mtime_ns и SHA256 tensor directory. Перед будущей
загрузкой rank надо проверять identity, dims, count, полный уникальный набор
routed IDs и происхождение GGUF. Иные веса требуют новой калибровки.

## Held-out replay, это не замер ускорения

Decode replay переставляет захваченные batch-маршруты в token-major порядок,
а prefill replay обрабатывает уникальные эксперты слоя на chunk128. Маршруты
реального autoregressive decode и длинные контексты ещё не записаны.

| Slots | LRU decode hit | Hot+stream decode hit | Hot seeds | LRU prefill hit | Hot+stream prefill hit |
|---:|---:|---:|---:|---:|---:|
|128|0%|0%|0|0%|0%|
|248 (6GiB)|22.085%|23.166%|20|0%|2.298%|
|384|23.562%|28.684%|156|0%|4.871%|
|512|32.048%|32.865%|284|0%|7.269%|

Для decode оставлено228 stream slots =57 layers×4 selected experts, остальное
можно закреплять под hot seeds. Слишком агрессивный pinning ухудшает результат:
244hot+4stream при248slots даёт16.680% против22.085% LRU. Такой seed режим
не следует включать по умолчанию. Полный sweep сохранён в JSON, включая
цену начального seed upload. Для prefill оставляется128 stream slots.
Hit rate считается по экспертным запросам; prefill и decode denominators разные.

Самые частые train пары: (layer23,expert20)262 раза, (18,116)228,
(22,31)197. Это результат короткой выборки, не универсальные hot experts.
Один code128 тем более не даёт репрезентативного production rank.

## Бюджет карты0

`cache_rank.py budget` использует настоящие1191 tensor descriptors без чтения
payload. Воспроизведён точный текущий fixed slot stride **25,909,632 байт**,
включая LCM блоков98/136/36 и128-byte alignment. При6GiB ровно248 slots,
как в реальном логе GPU-кеша. Некрупнейшие эксперты занимают такой же slot:
это цена единого quant-stride, уменьшить её можно несколькими пулами по layout.

`cache-vram-budget-v1.json`: карта32110MiB, CUDA dense weights9.947GiB,
chunk2048. После весов, KV, явного MSA envelope и2GiB резерва остальных
scratch/context/transfer получаются **оценочные**, а не измеренные бюджеты:

| ctx | Native F16 KV | MSA envelope | Native slots | Q8 main + F16 index slots |
|---:|---:|---:|---:|---:|
|4K|0.527GiB|0.346GiB|768|777|
|32K|4.219GiB|2.768GiB|514|587|
|64K|8.438GiB|5.536GiB|225|371|

Native q8 KV **не реализован**; последняя колонка — сценарий для планирования.
MSA envelope складывает2 FP32 scores, F16 attention mask, converted index K
и общие masks/positions. Реальный gallocr использует aliases, а другие живые
тензоры могут увеличить peak; перед реальным auto-budget нужен измеренный
reserve/peak и safety margin. Нельзя автоматически выделять всё число slots
по этой таблице без CUDA allocator проверки.

Ref64Kq8/ubatch2048 держит3 полных слоя10..12, CUDA compute6801.21MiB,
main KV4080MiB, index KV1020MiB. Ref indexer выделяет и dummyV; native хранит
лишь F16 indexK. У ref три слоя10..12 занимают меньше максимальных fixed slots;
поэтому `slots//128` нашего универсального пула не равен числу ref GPU layers.
Необходимо учитывать layout padding, а также измеренный scratch.

## Продолжение и воспроизведение

1. Настоящий cache seed loader: identity validation, rank permutation guard,
   upload top budget без изменения router, protected seed entries отдельно от
   текущих batch protected IDs; зарезервировать stream working set.
2. Сначала source q8_K-compatible GPU correctness gate, затем replay exact
   routing/weights и сравнение logits seed0/seedN. Текущий GPU default min32
   вообще оставляет decode на CPU, поэтому этот rank ещё не ускоряет сервер.
3. Снять production-like decode и long-context калибровку. Выбор seed count
   обучать отдельно, held-out не использовать для заявления выигрыша.
4. Измерить VRAM allocator peak на4/32/64K, затем окно om_bench с тёплым и
   холодным кешем. Hit rate сам по себе не доказывает прирост tok/s.

Сборка `cache_build.sh` через systemd-run использует отдельный
`build-cache-ranking`, shared build lock и готовые vendor/engine archives.
Engine archive должен соответствовать текущему `Config` ABI/route_observer;
после изменения заголовка сначала пересобирается CPU-decode engine.
Сборка rc0; `python3 tools/m31/cache_test.py`4/4, включая запрет eviction
защищённых batch entries и отказ испорченному/неполному trace. Коммитов нет.
