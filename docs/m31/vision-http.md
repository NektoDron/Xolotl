# HTTP vision в M31

Оба opt-in адаптера передают `mixed_manifest` в нативный `m31_serve`.
Вся последовательность, включая текст, границы изображений, кадры и timestamps,
попадает в один слоевой префилл. Частота видео по умолчанию — 1 fps;
декодирование кадров и timestamps выполняет тот же mtmd PR, что в reference.

У standalone `tools/m31/serve_m31.py` включение явное: `--vision-config FILE`.
У common bridge `tools/m31/serve/serve_m31.py` тот же объект помещается в `vision`
основного JSON-конфига. Пример объекта (пути зависят от выбранной сборки):

```json
{
  "adapter": "/mnt/ai/Xolotl-m31-vision/build-vision/m31_vision_adapter",
  "model": "/mnt/ai/models/MiniMax-M3.1-Flash-GGUF/UD-IQ3_S/MiniMax-M3.1-Flash-UD-IQ3_S-00001-of-00005.gguf",
  "mmproj": "/mnt/ai/models/MiniMax-M3.1-Flash-GGUF/mmproj-BF16.gguf",
  "cache_dir": "/mnt/ai/logs/m31-vision/http-cache",
  "backend": "cpu",
  "threads": 16,
  "cpu_workers": 6,
  "limits": {
    "video_fps": 1,
    "max_media_bytes": 33554432,
    "max_total_bytes": 134217728,
    "max_media": 32,
    "max_image_pixels": 16000000,
    "max_video_seconds": 120,
    "max_video_frames": 120,
    "max_video_fps": 30,
    "allow_http": false,
    "allow_local": true
  }
}
```

При standalone `model` в vision-конфиге можно опустить: берётся `--model`.
Для GPU-энкодера требуется `backend=cuda` либо opt-in `auto`, собственная aictl-сессия GPU0 и
свободная VRAM одновременно с LLM. Контекст и expert cache подбирает владелец
карты; HTTP-слой не выгружает модель и не переключает карту самостоятельно.
CPU-энкодер использует параллельные независимые mtmd contexts.
`auto` проверяет свободную VRAM при первом encode, после загрузки LLM; порог
задаётся `gpu_min_free_mib` (по умолчанию 4096 MiB), иначе используется CPU.

OpenAI content parts:

```json
{
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "Опиши изображение и порядок событий."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
    {"type": "video_url", "video_url": {"url": "file:///path/clip.mp4", "fps": 1}}
  ]}],
  "video_fps": 1,
  "max_tokens": 256
}
```

Поддержаны `image_url` со строкой или объектом `url`, data URLs, локальные
пути/file URLs, `video_url` и `video_file` со строкой либо `url`/`path`.
HTTP(S) включается `allow_http=true`; чтение ограничено размером и таймаутом.
Частота в части видео перекрывает `video_fps` запроса, затем действует default.
Несколько видео одного запроса используют одинаковую частоту. Число кадров
проверяется до энкодера по длительности ffprobe и выбранной частоте.

Порядок media сохраняется внутри сообщения и между ходами диалога. Helper
получает текстовые уникальные маркеры; после шаблона они заменяются mtmd marker,
который разворачивает native mtmd в границы, timestamps и visual rows. Сырые
`<__media__>` в мультимодальном промпте запрещены, чтобы пользовательский текст
не мог поглотить чужую картинку. Исходный словарь сообщений не меняется.

Источники сохраняются по SHA256 байтов. Encoder отдельно кеширует embeddings
медиа и полный подготовленный промпт. Локальный файл копируется в собственный
кеш перед обработкой, поэтому замена файла отправителем не подменяет уже
подготовленную картинку. Автоматического удаления кеша пока нет.

`tools/m31/vision_http_session.py` запускается только внутри собственного
card worker и окна exclusive `.build.lock`: native profile → standalone HTTP →
два cold-media low TTFT запроса → шесть vision-проверок → три tool calls →
два warm low TTFT запроса. Он пишет полный
receipt, хеши, ENV, логи, ответы и параметры измерения. Quality-запросы используют canonical `reasoning_effort=low`, `max_tokens=1200`;
low-запросы повторяют prompts/порядок/effort/temp/max_tokens эталонного
клиента. У warm low encoder кеш уже прогрет quality-пробами; cold-media image —
первый encoder запрос, cold-media video идёт следом с уже загруженным worker. LLM prefix
cache и context shifting не используются. GPU запуск этим документом не заявлен.

CPU-проверки: `python3 tools/m31/serve/test_vision_http.py` (10 проверок, включая
настоящий HTTP POST с fake engine), `python3 tools/m31/test_vision_http_session.py`
(3 проверки profile/tools/ref request). `test_vision_standalone_cli.py` запускает
настоящий standalone CLI из временного cwd без репозитория в PYTHONPATH, с fake
JSONL peers; это проверяет import-пути и HTTP media→encoder→native handoff.
Проверка native numeric parity и
скоростные результаты находятся в итоговом отчёте координатора.
