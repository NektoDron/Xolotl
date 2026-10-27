# Port7: каноническая сборка CPU → CUDA

Подготовлены `tools/m31/port7_canonical.py` и оболочка `port7_canonical_build.sh`. Сборка **не запускалась**. `plan` только печатает команду systemd; `build --execute` требует systemd `User=nekto` и новый каталог `build-port7-canonical/ID`. До запуска координатор выбирает winner-профиль, замораживает исходники и даёт окно сборки. Оболочка берёт shared build-lock; размер `--jobs` ограничен 24. Координатор учитывает этот процесс как один из трёх слотов и исключает его пересечение с замерами скорости.

Пример подготовки команды (пути профиля и tokenizer выбирает координатор):

```
python3 tools/m31/port7_canonical.py plan \
  --out /mnt/ai/Xolotl-m31/build-port7-canonical/WINNER \
  --profile /ABS/WINNER-profile-resolved.json --tokenizer /ABS/m31_tokenizer \
  --model /ABS/MiniMax-M3.1-Flash.gguf --jobs 24
```

Скрипт сохраняет SHA256 исходников и поддержки (CMake/helpers, C/C++/CUDA/INC, Python/server/session/profiles), копии текстовых исходников, внешние input identities, версии/хеши toolchain, CMakeCache, compile_commands, команды линковки, все статические библиотеки включая ggml, readelf/ldd конечных ELF. До/после каждой стадии проверяет неподвижность source/support и защищённых v8 ELF/receipt. Модель не копируется; фиксируются stat-поля. Исходники собираются из root при заморозке, поэтому это не механизм параллельной разработки во время сборки.

Vendor — отдельное sparse working tree из локальной object database pinned `6050b467c07a95c7b87debd4ddac2f8ba8d9b807`, без сети. Common patch и три API patches (decode-fusion, hardened exact-MMV, RMS-only fusion) применяются явно; сохраняются их SHA, исходный/изменённый manifests и vendor diff. Исходное managed vendor tree и v8 не меняются. CPU и CUDA — отдельные configure/build dirs; семь конечных инструментов (включая port7_verify_dump) копируются в `bin-cpu`/`bin-cuda` и хешируются. Логи находятся в `/mnt/ai/logs/m31-port/port7-canonical-INVOCATION/`, ссылка на них есть в output.

Проверки после сборки используют конечные ELF: tiny NumPy math/chunk/arena, реальный worker prefix cache, реальный Python RPC с CPU tiny model (DSpark fallback при отсутствующем sidecar, grammar ACK, cancel, recovery), server/vocab и launcher fixtures. Отдельный C++ controller fixture проверяет DSpark queue/rollback на независимой модели; это не качество реального sidecar. Необязательный `--draft-fixture JSON` добавляет реальный CPU DSpark probe на сохранённых features. Весь builder запускается с пустым CUDA_VISIBLE_DEVICES и сам GPU не занимает.

`compile_complete` и CPU fixtures не означают приёмку модели. Отдельный `gpu-validation-required.json` оставляет PENDING: gate20 через Engine.verify с rollback/features/head1, выборочные 16K/32K, om_bench в чистом окне, GSM8K100/tools23 и HTTP/SSE. Их выполняет координатор на финальных ELF через aictl/card-lock. HIP/SYCL и полные внешние SDK здесь не проверяются.

Статически проверены Python AST и bash syntax. Configure, компиляция и новые fixtures до ACK не выполнялись.

Финальный профиль должен быть заранее RESOLVED: JSON m31-runtime-profile-v1 без ключа inherits; иначе plan/build отклоняются. Если используется profile_plan.load_profile(), перед сохранением удалить оставшийся inherits key. Это позволяет runtime-helpers/runtime-profile.json жить отдельно от исходного native-gpu.json. Финальное решение root в14:13UTC: один canonical build с --jobs24, CPU→CUDA последовательно, после clean X. Это заменяет прежний подготовительный j3 план; другие сборки и замеры координатор не совмещает с этим окном. Отдельная controller fixture имеет include dirs frozen source-tree/tools/m31, source-tree/include и managed vendor/ggml/include, как уже проверенный capture-view-session builder; vendor/src/nlohmann этой fixture не нужны. Ни API patch chains, ни CMake не должны дополнительно менять vendor после сохранения vendor-patched.json.
