# ModelForge AI

**Кроссплатформенная лаборатория создания ИИ-моделей, автономных агентов, научных исследований и prompt-based разработки.**  
**Android ARM64 • Termux ARM64 • Windows x64 • Linux x64 • macOS • Web Studio + CLI**

---

## Главные возможности (v1.0.0)

- **Model Builder & 10 архитектур нейросетей**: создание по текстовому промпту компактных языковых моделей (`decoder_transformer`, `tiny_transformer`, `encoder_transformer`, `encoder_decoder_transformer`, `rnn`, `lstm`, `gru`, `ssm`, `hybrid_attention_ssm`, `custom_block`).
- **Многоязычные токенизаторы (RU / TG / EN)**: реальное обучение `UnicodeBPETokenizer` и `CharTokenizer` с полной проверкой кириллицы и таджикских символов (`ғ, ӣ, қ, ӯ, ҳ, ҷ`).
- **Training Engine**: реальное обучение с нуля (`scratch`), `fine-tuning` и `transfer learning` (заморозка эмбеддингов и нижних блоков) на собственном движке обратной автодифференциации (`modelforge.architectures.tensor_engine`) с проверкой изменения весов (`weight_l2_delta > 0`), градиентным накоплением, динамическим снижением `batch_size`, контролем RAM/времени, сохранением проверяемых чекпоинтов (`.mfckpt` с SHA-256), возобновлением (`resume`) и `early stopping`.
- **Inference & Экспорт**: потоковая генерация с отменой, `temperature`, `top-k`, `top-p`, `repetition_penalty`, профилированием скорости и RAM, а также реальный экспорт в **GGUF v3** (`.gguf`), **Safetensors** (`.safetensors`), **PyTorch** (`.pt` при установленном `torch`), `JSON` и `YAML`.
- **Agent Builder & Multi-Agent Runtime**: 10 типов агентов (`Research`, `Architecture`, `Data`, `Coding`, `Training`, `Evaluation`, `Debugging`, `Documentation`, `Supervisor`, `Custom`), `TaskRouter`, `MessageBus`, изолированная память, граф состояний (`StateGraph` DAG для последовательного и параллельного выполнения) и защита от зацикливания.
- **Prompt-to-Project**: редактирование проекта текстовыми инструкциями с предварительным снимком, unified diff, запуском тестов в ограниченном подпроцессе (`restricted_subprocess`), историей промптов и точным откатом (`rollback`).
- **Research Engine & Experiment Registry**: контролируемые сравнительные эксперименты архитектур с жёстким разделением фактов, гипотез и допущений и обязательным сохранением отрицательных результатов.
- **MCP STDIO / Streamable HTTP & Плагины**: встроенный менеджер JSON-RPC 2.0 MCP-серверов, санитизация недоверенных внешних ответов, журнал вызовов и плагины с `manifest.json` и проверкой схем.
- **4 режима работы**: `Local Mode`, `Remote Mode`, `Hybrid Mode` и `Offline Mode` (с блокировкой внешних сетевых вызовов) + автоматический **Low Memory Mode** для устройств с `<= 3.5 ГБ RAM`.
- **Единое ядро для 3 интерфейсов**:
  1. Нативное кроссплатформенное приложение **Flutter / Dart / Material 3** (`apps/flutter_app/`, 12 экранов, RU/TG/EN локализация, тёмная/светлая тема, масштабирование текста).
  2. Встроенный **Web Studio** (`http://0.0.0.0:8765/`) + версионированный REST API (`/api/v1/*`) и WebSocket (`/ws/events`).
  3. Полнофункциональный **CLI (`modelforge`)**.

---

## Быстрый старт (CLI и Сервер)

```bash
# 1. Установка Python-ядра и серверных модулей
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e ".[full,dev]"

# 2. Полная диагностика окружения и платформы
modelforge doctor

# 3. Создание проекта, русско-таджикской гибридной модели и реальное обучение
modelforge create project --name "RU-TJ Lab"
modelforge create model --prompt "Создай компактную языковую модель для русского и таджикского текста с экспериментальной архитектурой"
modelforge train --model <MODEL_ID> --epochs 6
modelforge model generate --model <MODEL_ID> --prompt "Салом ҷаҳон! Привет"
modelforge export --model <MODEL_ID> --format gguf

# 4. Запуск сервера ядра и интерактивного Web Studio
modelforge server --host 0.0.0.0 --port 8765
```

Подробные инструкции по установке для **Android ARM64, Termux ARM64, Windows x64, Linux x64 и macOS** приведены в [`INSTALL.md`](INSTALL.md). Архитектура описана в [`ARCHITECTURE.md`](ARCHITECTURE.md), честный перечень ограничений — в [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md).
