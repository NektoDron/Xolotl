# Профили M31

`q2-stock.json` и `q2-native.json` предназначены только для отдельной волны
UD-Q2_K_XL. Оба наследуют `port7-baseline.json`: stock выключает custom CPU
decode и использует RoutedGraph, native включает `M31_CPU_STOCK_EXPERTS=1`
для штатных IQ2_XS dot-функций в существующем fused CPU decode. Флаг
выключен без явного opt-in; арифметика прежних IQ3_XXS/IQ4_XS/NVFP4
не меняется. DSpark/batch-dense и GPU cached decode остаются выключенными.
Эти профили не наследуют качество или приёмку IQ3_S; результаты нового
кванта фиксируются отдельно в `REPORT-q2kxl.final.md`.

`native-gpu.json` содержит48 ENV значений и CLI параметры финального native
CUDA кандидата port6: ctx40000, max chunk32768, CPU80, host32, pack8,
expert cache7GiB, RAM arena170GiB, seed0. Включены reuse, persistent pool,
CPU residual interface, CPU/dense decode caches и GPU prefill chain.
Fusion выключен. BANKS/GU/MASK остаются opt-in и здесь выключены.
Профиль не содержит путей модели, executable или aictl preset.

Получить готовые argv и ENV для просмотра:

```sh
python3 tools/m31/profile_plan.py \
  --profile tools/m31/profiles/native-gpu.json \
  --engine /path/to/m31_serve \
  --model /path/to/model.gguf
```

Скрипт только печатает JSON, не запускает engine и не меняет aictl.
`--output NEW.json` сохраняет новый plan без перезаписи существующего файла.
Параметры `engine_arguments` предназначены непосредственно для `m31_serve`
(булевы CLI значения имеют вид `--cpu-residual 1`). HTTP adapter может
использовать те же значения через свой интерфейс опций.

Потребитель сначала снимает `unset_environment`, затем применяет
`environment`. Так seed0 не зависит от унаследованного seed receipt,
а границы регистрации/default threads не переопределяются чужим ENV.
`profile_plan.load_profile(path)` разрешает относительное `inherits`,
объединяет environment/requires и unset lists. Модель и executable передаёт
потребитель. GPU сессия по-прежнему резервируется штатным aictl launcher.
Профиль ничего не развёртывает и не меняет defaults программы.

`exact-debug.json` наследует native профиль и явно включает EXACT1,
CHAIN0, FAST0, CPU_QUANTS1. Нужна сборка
`STRATA_M31_CPU_QUANTS_CUDA=ON`; поле requires описывает это требование,
planner не проверяет compile flags executable автоматически. Выбор режима
делает EXACT; старый CPU_QUANTS1 сохранён как явное обозначение debug режима.
Промежуточные F32 возвращаются в RAM. Oracle здесь CPU-strict PR31192;
нативная производительность и полная quality приёмка к этому режиму не относятся.

Источник native параметров — acceptance-v1/speed manifest и canonical build
receipt. JSON хранит их SHA256 и SHA256 обоих проверяемых ELF; это provenance,
а не жёсткий путь к файлу. Итоговые gate20, long16/32K, GSM100, tools23 и
повторные speed результаты находятся в отчёте port6. Сам профиль не объявляет
приёмку пройденной. Native math сравнивается с GPU-offload oracle PR31192
при top1>=99%, meanKL<=1e-3, maxKL<=0.05.

Проверка профиля без GPU прошла12 checks: точное совпадение48 ENV и CLI,
seed0/unsets, source и canonical ELF digests, отсутствие локальных путей,
exact inheritance, literal argv, отказ при cycle/malformed input и CLI plan.
Receipt: `port6-profile-native-v1.json` в каталоге логов port6.
