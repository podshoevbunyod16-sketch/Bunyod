# ModelForge AI — Phase 0: Аудит окружения, рисков и поэтапный план реализации

Дата аудита: **2026-11-27 (UTC)**  
Репозиторий: `https://github.com/podshoevbunyod16-sketch/Bunyod`  
Рабочая ветка: `arena/a4eb2945-bunyod`

---

## 1. Результаты аудита окружения

### 1.1 Локальное сборочно-тестовое окружение (Sandbox)
- **ОС / Ядро**: Linux `6.1.158+` x86_64 GNU/Linux
- **CPU**: Intel Xeon @ 2.60GHz, 2 vCPU (AVX2, AVX-512)
- **RAM**: 3.8 GiB физическая память, 0 B swap — идеально подходит для реальной проверки **Low Memory Mode (<= 4 ГБ RAM)**!
- **Диск**: 20 GiB свободно
- **Инструменты**: Python 3.11.2, `pip`, `venv`, `gcc`/`g++`/`make`, Node.js / `npm`, `git`, `gh` CLI (авторизован к `podshoevbunyod16-sketch/Bunyod`).
- **Сетевая политика локального sandbox**: доступ открыт к `github.com`, `codeload.github.com`, `api.github.com`, `pypi.org`, `files.pythonhosted.org`, `registry.npmjs.org`. Прямой доступ из локального контейнера к `storage.googleapis.com` / `pub.dev` / `dl.google.com` закрыт политикой контейнера.
- **Следствие для кроссплатформенных сборок**:
  - Python Core, Autograd/Model Runtime, Training Engine, Inference, Agents, Prompt-to-Project, Research Engine, MCP Manager, CLI и FastAPI Backend + Web Studio UI тестируются **локально** И в **GitHub Actions** (`ubuntu-latest`, `windows-latest`, `macos-latest`).
  - Flutter GUI (`apps/flutter_app/`), `flutter analyze`, `flutter test`, а также нативные сборки **Android ARM64 (`.apk` + `.aab`)**, **Windows x64 (`.exe` + Portable `.zip`)**, **Linux x64 (`.AppImage` + `.deb`)** и **macOS (`.dmg`)** реально собираются и проверяются через **GitHub Actions runners** при пуше в рабочую ветку `arena/a4eb2945-bunyod`.

### 1.2 Аудит лицензий и ABI зависимостей
- **Базовые зависимости ядра**: `numpy` (BSD-3), `pydantic` (MIT), `pyyaml` (MIT) — кроссплатформенные колёса доступны для Linux x64, Windows x64, macOS (x64/ARM64) и собираются/устанавливаются в Termux ARM64 (`pkg install python-numpy`).
- **Серверные и форматные модули**: `fastapi` (MIT), `uvicorn` (BSD-3), `httpx` (BSD-3), `websockets` (BSD-3), `psutil` (BSD-3), `safetensors` (Apache-2.0) — подключаются модульно с проверкой доступности и graceful degradation.
- **Тензорный движок и обучение**:
  - Собственный **Native Autograd Tensor Engine (`modelforge.architectures.tensor_engine`)** поверх `numpy` (точная обратная автодифференциация, слои Transformer/GPT, Encoder, Encoder-Decoder, RNN, LSTM, GRU, S4D-lite SSM, Hybrid Attention+SSM, AdamW, Cosine Warmup, Gradient Accumulation, Safetensors + GGUF v3 экспорт). Работает везде, включая устройства с 3 ГБ RAM и Termux ARM64 без 2 ГБ CUDA-зависимостей.
  - Опциональный **PyTorch Adapter (`modelforge.architectures.pytorch_adapter`)** — лениво загружается при наличии `torch`, обеспечивая совместимость с экосистемой PyTorch (`.pt` checkpoints, `nn.Module`).

### 1.3 Обоснование структуры пакетов
В соответствии с разделом 19 спецификации («Структуру разрешено улучшать, если это обосновано»):
- Весь Python-код сгруппирован внутри единого пакета `modelforge/` (`modelforge/core`, `modelforge/backend`, `modelforge/orchestration`, `modelforge/agents`, `modelforge/model_builder`, `modelforge/architectures`, `modelforge/tokenizers`, `modelforge/datasets`, `modelforge/training`, `modelforge/inference`, `modelforge/evaluation`, `modelforge/research`, `modelforge/experiments`, `modelforge/registry`, `modelforge/integrations`, `modelforge/storage`, `modelforge/security`, `modelforge/prompt_edit`, `modelforge/platforms`, `modelforge/cli`). Это исключает затенение стандартной библиотеки Python (`platform`) и сторонних пакетов HuggingFace (`tokenizers`, `datasets`) при установке wheel-пакета.
- Корневая директория `platform/` (без `__init__.py`) содержит платформенные адаптеры, инсталляторы и сборочные скрипты для `android/`, `termux/`, `windows/`, `linux/`, `macos/`.

---

## 2. Поэтапный план и критерии готовности (Phases 1–12)

1. **Phase 1 — Core MVP (`modelforge/core`, `storage`, `security`, `orchestration`, `platforms`)**
   - Режимы `local`, `remote`, `hybrid`, `offline` (жёсткая блокировка сети в `offline`).
   - SQLite WAL метаданные + файловое хранилище проектов и артефактов + разделённая память (проект / агент / диалог) + локальный RAG с цитированием фрагментов.
   - Менеджер задач, шина событий, граф состояний (DAG), повторы, отмена, восстановление.
   - Изоляция рабочих пространств, лимиты CPU/RAM/времени/диска, подтверждение опасных действий, редактирование секретов, хеш-цепочка аудита.
2. **Phase 2 — Backend API & CLI & Flutter Shell (`modelforge/backend`, `modelforge/cli`, `apps/flutter_app`)**
   - Версионированный FastAPI `/api/v1/*` + WebSocket `/ws/events` + встроенный адаптивный Web Studio.
   - Полный CLI `modelforge` со всеми командами спецификации (раздел 13).
   - Кроссплатформенное приложение Flutter / Dart / Material 3 со всеми 12 экранами (раздел 12), RU/EN/TJ локализацией, тёмной/светлой темой и переключением Local/Remote/Hybrid/Offline.
3. **Phase 3 — Model Builder & Tokenizers & Architectures**
   - Токенизаторы (Char + Byte/Unicode BPE) с полной поддержкой русского и таджикского алфавитов (`ғ, ӣ, қ, ӯ, ҳ, ҷ`).
   - Архитектуры: Decoder-only GPT/Tiny Transformer, Encoder, Encoder-Decoder, RNN, LSTM, GRU, SSM (diagonal state-space), Hybrid (Attention + SSM), Custom Block.
   - Построение проверяемой спецификации модели и оценка ресурсов по текстовому промпту.
4. **Phase 4 — Training Engine & Inference & Evaluation**
   - Валидация датасетов (кодировка, пустые строки, дедупликация, лицензия, утечки train/val/test).
   - Реальный цикл обучения с нуля, fine-tuning и transfer learning (заморозка слоёв), градиентное накопление, динамическое снижение batch size, контроль RAM/времени, чекпоинты `.mfckpt` / `.safetensors`, экспорт `.gguf` v3, resume и early stopping.
   - Потоковая генерация с отменой, temperature, top-k, top-p, repetition penalty и профилированием памяти/задержки.
5. **Phase 5 — Agent Builder & Multi-Agent Runtime**
   - Все 10 типов агентов, Agent Registry, Task Router, Message Bus, изолированная память, безопасные инструменты, защита от циклов и превышения бюджета шагов/токенов.
6. **Phase 6 — Prompt-to-Project Engine**
   - Анализ проекта, создание снимка/ветки состояния, безопасные правки в пределах рабочей директории, unified diff, прогон тестов, ограниченные попытки автоисправления, история промптов, undo/rollback и подтверждение разрушительных операций.
7. **Phase 7 — Research Engine & Experiment Registry**
   - Полный цикл: вопрос → источники (с URL, датой и разделением фактов/гипотез) → гипотеза → план → запуск → измерение → сравнение → сохранение (включая отрицательные результаты) в Experiment Registry.
8. **Phase 8 — Integrations (LLM Providers, MCP STDIO/HTTP, Plugins, GitHub, Composio)**
   - Единый адаптер провайдеров без скрытого переключения.
   - Реальный MCP Manager (JSON-RPC 2.0 по STDIO и Streamable HTTP), санитизация недоверенных ответов, журнал вызовов и права агентов.
   - Система плагинов с `manifest.json`, JSON Schema входа/выхода и проверкой разрешений.
9. **Phase 9–11 — Platform Packaging & GitHub Actions CI/CD**
   - Termux ARM64 без root (`platform/termux/`), Android ARM64 APK/AAB (`android-build.yml`), Windows x64 Setup EXE + Portable ZIP (`windows-build.yml`), Linux x64 AppImage + DEB (`linux-build.yml`), macOS DMG (`macos-build.yml`), CI (`ci.yml`) и Release pipeline (`release.yml`).
10. **Phase 12 — Hardening, End-to-End Verification & Honest Reporting**
    - Полный прогон unit/integration/API/CLI/E2E тестов, реальное обучение русско-таджикской компактной модели с проверкой изменения весов и генерацией текста из чекпоинта, проверка workflow runs в GitHub Actions и финальная документация.
