# Changelog

Все заметные изменения проекта **ModelForge AI** фиксируются в этом файле.

## [1.0.0] — 2026-11-27

### Added
- Полная инициализация платформы **ModelForge AI** с нуля (удалён устаревший исследовательский прототип).
- **ModelForge Core**: конфигурация, режимы `local`/`remote`/`hybrid`/`offline`, детектор платформ и `Low Memory Mode`, SQLite WAL хранилище, раздельная память (`project`, `agent`, `dialogue`) и локальный RAG с цитированием строк.
- **Security**: изоляция рабочих пространств `resolve_within_workspace`, одноразовые подтверждения опасных операций, хранилище секретов с редактированием логов и хеш-цепочка аудита SHA-256.
- **Model Builder, Tokenizers & Architectures**: `UnicodeBPETokenizer` и `CharTokenizer` (русский, таджикский `ғ,ӣ,қ,ӯ,ҳ,ҷ`, английский), движок обратной автодифференциации и 10 архитектур нейросетей.
- **Training, Inference & Export**: реальное обучение (`scratch`, `finetune`, `transfer`), resume из чекпоинтов, потоковая генерация с отменой, экспорт в `GGUF v3` и `Safetensors`.
- **Agents, Prompt-to-Project & Research**: 10 типов агентов, мультиагентный `StateGraph`, правка проектов текстовыми промптами с diff и `rollback`, `Research Engine` и `Experiment Registry`.
- **Integrations**: провайдеры LLM без скрытого переключения, `MCPManager` (`STDIO` + `Streamable HTTP`) со встроенным локальным MCP-сервером, плагины `tajik_text_inspector` и `architecture_advisor`.
- **UI & CLI**: приложение Flutter Material 3 (`apps/flutter_app/`, 12 экранов, RU/TG/EN), встроенный Web Studio, CLI `modelforge`.
- **CI/CD**: GitHub Actions workflows `ci.yml`, `android-build.yml`, `windows-build.yml`, `linux-build.yml`, `macos-build.yml`, `release.yml`.
