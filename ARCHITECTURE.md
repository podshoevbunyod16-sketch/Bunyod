# Архитектура ModelForge AI

## 1. Принципы построения

1. **Собственное модульное ядро (`ModelForgeCore`)**: ни один внешний ML- или агентный фреймворк не является жёстким монолитным центром. Все подсистемы связаны через чёткие контракты и ленивую загрузку опциональных пакетов (`torch`, `litellm`, `llama-cli`, `safetensors`).
2. **Единая бизнес-логика**: CLI (`modelforge/cli`), REST/WebSocket сервер (`modelforge/backend`) и графические клиенты (`apps/flutter_app` + встроенный Web Studio) вызывают одни и те же сервисы `ModelForgeCore`.
3. **Честность вычислений и безопасности**:
   - Обучение фиксируется как выполненное только при реальном изменении тензоров весов (`weights_changed == True`, `weight_l2_delta > 0`) и сохранении проверяемого по SHA-256 чекпоинта.
   - Исполнение кода тестов помечено как `restricted_subprocess` (изолированная рабочая директория, очищенное окружение, таймауты, `RLIMIT_CPU`/`RLIMIT_AS` на POSIX) — без ложных заявлений о полноценной виртуальной машине ядра ОС.
   - Данные из сети, документов и MCP-инструментов проходят через `sanitize_untrusted_content` (`executable=False`) и никогда не исполняются как системные команды.

---

## 2. Карта модулей

| Модуль / Директория | Назначение |
|---|---|
| `modelforge/core/` | Конфигурация, режимы (`local`, `remote`, `hybrid`, `offline`), проекты, `EventBus`, `SystemStateManager`, пул задач `TaskManager`, центральный `ModelForgeCore` |
| `modelforge/orchestration/` | Детерминированный DAG состояний `StateGraph` (параллельные и последовательные волны, повторы, отмена) и `WorkerRegistry` / `RemoteExecutorClient` |
| `modelforge/platforms/` | Детектор ОС/ABI/Termux/Android, телеметрия CPU/RAM/диска, автоматический `Low Memory Mode` (`<= 3.5 ГБ RAM`) |
| `modelforge/security/` | Изоляция путей `resolve_within_workspace`, политики разрешений `PermissionPolicy`, одноразовые токены подтверждения `ConfirmationManager`, хранилище секретов `SecretManager`, хеш-цепочка `AuditLogger` |
| `modelforge/storage/` | SQLite WAL база (`StorageDatabase` с авто-восстановлением прерванных задач и `backup`/`restore`), `ArtifactStore`, раздельная память (`project`, `agent`, `dialogue`) и локальный BM25/триграммный RAG с точными цитатами строк (`MemoryAndRagStore`) |
| `modelforge/tokenizers/` | `CharTokenizer` и `UnicodeBPETokenizer` с верификацией русского, таджикского (`ғ,ӣ,қ,ӯ,ҳ,ҷ`) и английского текста |
| `modelforge/datasets/` | Импорт `TXT`, `MD`, `JSON`, `JSONL`, `CSV`, очистка пустых строк, дедупликация, проверка лицензий и детектор утечек между `train/val/test` |
| `modelforge/architectures/` | Движок обратной автодифференциации `tensor_engine.py` + 10 архитектур (`decoder_transformer`, `tiny_transformer`, `encoder_transformer`, `encoder_decoder_transformer`, `rnn`, `lstm`, `gru`, `ssm`, `hybrid_attention_ssm`, `custom_block`) + опциональный `pytorch_adapter.py` |
| `modelforge/model_builder/` | Парсер текстовых промптов в `ModelSpec`, предварительная эвристическая оценка ресурсов (`is_approximate=True`), генерация кода модели в проекте и управление версиями |
| `modelforge/training/` | Цикл обучения `scratch` / `finetune` / `transfer`, `AdamW`/`SGD`, cosine warmup, градиентное накопление, динамическое уменьшение `batch_size`, чекпоинты `.mfckpt` и экспорт `.safetensors` |
| `modelforge/inference/` | Потоковая генерация с отменой, `temperature`, `top-k`, `top-p`, `repetition_penalty`, профилирование скорости и бинарный сериализатор/инспектор **GGUF v3** + адаптер `llama.cpp` |
| `modelforge/evaluation/` | Реальные метрики: cross-entropy loss, perplexity, точность следующего токена, `chrF3`, скорость `tokens/sec`, потребление памяти и сравнение моделей |
| `modelforge/agents/` | 10 типов агентов, `TaskRouter`, `MessageBus`, изолированная память, безопасные инструменты, защита от бесконечных циклов |
| `modelforge/prompt_edit/` | Prompt-to-Project: снимок состояния, применение правок в пределах проекта, unified diff, автотесты с ограниченным числом попыток ремонта и точный `rollback` |
| `modelforge/research/` | Цикл автономного исследования, разделение фактов и гипотез, сохранение отрицательных результатов и полный `Experiment Registry` |
| `modelforge/integrations/` | Провайдеры LLM (без скрытого переключения), `MCPManager` (`STDIO` + `Streamable HTTP`) со встроенным локальным MCP-сервером, `PluginManager`, `GitHubIntegration`, `ComposioAdapter` |
| `modelforge/registry/` | Единый поиск по проектам, моделям, агентам, датасетам, архитектурам и экспериментам + экспорт артефактов |
| `modelforge/backend/` | FastAPI `/api/v1/*`, WebSocket `/ws/events` и встроенный адаптивный Material-3 Web Studio |
| `modelforge/cli/` | Единый CLI `modelforge` |
| `apps/flutter_app/` | Нативное кроссплатформенное приложение Flutter / Dart / Material 3 (12 экранов, RU/TG/EN, Light/Dark, Desktop + Android ARM64) |
| `platform/` | Инсталляторы и сборочные скрипты для `android/`, `termux/`, `windows/`, `linux/`, `macos/` (без `__init__.py`, чтобы не затенять стандартный модуль Python `platform`) |

---

## 3. Доставка Python Core по платформам

- **Windows x64 (`Setup.exe` / `Portable.zip`)**: Python Core компилируется PyInstaller в автономную директорию `backend/modelforge-backend.exe` внутри бандла Flutter. При старте GUI автоматически запускает локальный процесс при работе в `Local Mode`.
- **Linux x64 (`AppImage` / `DEB`)**: аналогично комплектуется автономным `backend/modelforge-backend` внутри `/opt/modelforge-ai/backend` и `AppDir/usr/bin/backend`.
- **macOS (`DMG`)**: автономный `modelforge-backend` упакован в `ModelForge AI.app/Contents/Resources/backend/`.
- **Android ARM64 (`APK` / `AAB`)**: подключается к локальному сервису Termux ARM64 (`127.0.0.1:8765`, без root) или к удалённому серверу (`Remote` / `Hybrid Mode`), а без сервера работает с локальным офлайн-кэшем устройства.
