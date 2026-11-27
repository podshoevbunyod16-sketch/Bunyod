# REPRODUCE v0.17.0

Всё воспроизводится офлайн, без сети и без установки зависимостей, кроме
`pytest` (только для тестов). Python ≥ 3.10 (проверено на 3.11.2).

## 1. Проверить, что дерево зелёное

```bash
cd Bunyod
python -m pytest tests_v16 -q            # 348 passed
python -m pytest tests_v17 -q            # 332 passed
python -m pytest tests_v9 tests_v10 -q   # 54 passed (40 + 14)
python -m pytest rasm_cim_v8/tests -q    # 52 passed
# итого 786 passed, 0 failed, 0 skipped
```

Корневые `test_rasm*.py` и `rasm_cim_v1_snapshot` удалены из рабочего дерева
2026-11-27 вместе с Rust-прототипом и отчётами v0.2–v0.10; восстановить их можно
из git-истории (`git log --diff-filter=D -- test_rasm.py`). Остальные
исторические деревья связаны зависимостями и оставлены (см. `README.md`).

## 2. CLI

```bash
python -m rasm_cim version
# rasm-cim v0.17.0 (ядро v0.16.0, схема kb 2)
python -m rasm_cim --help    # 24 подкоманды
```

## 3. Все измерения — одной серией

```bash
python -m rasm_cim metrics      # семантика + источники (План B, D)
python -m rasm_cim coref        # кореференция (План C)
python -m rasm_cim generation   # связность (План E)
python -m rasm_cim learn        # обучение без забывания (План F)
python -m rasm_cim stress --n 100   # фаззинг, офлайн, время, память (План G)
python -m rasm_cim verify       # инварианты базы
```

Каждую можно добавить `--json` и сравнить два прогона: они детерминированы
(`metrics`, `coref`, `generation`, `learn` — побайтово; в `stress` меняются
только тайминги).

## 4. Ожидаемые числа (демо-база, эта машина)

| команда | ключевые значения |
|---|---|
| `metrics` | акт 37/38 = 97,4 %; F1 0.975; ответы 27/28 = 96,4 %; без источника 0,0 %; источники 1.0, посылки 13/13 |
| `coref` | верно 4/5 = 80 %, отказов 1, ошибочных связок 0, ответы 4/4 |
| `generation` | утверждения с источником 13/13, доказательство 42,9 % планов, 1,86 предложения, детерминизм да |
| `learn` | уроки 12/12, удержание 100 %, новых противоречий 0, дрейф 0 %, откат точный |
| `stress --n 100` | падений 0, нарушений сети 0 (38 файлов), p50 ≈ 0,54 мс, пик ≈ 26 КБ, рост ×0,98–1,16 |

## 5. Диалог вручную

```bash
export RASM_CIM_HOME=$(mktemp -d)
python -m rasm_cim teach "Пушкин написал Евгений Онегин"
python -m rasm_cim ask "Кто написал Евгений Онегин?"
python -m rasm_cim explain "Кто написал Евгений Онегин?"      # доказательство
python -m rasm_cim plan    "Кто написал Евгений Онегин?"      # план ответа
python -m rasm_cim teach "Париж находится в Франция"
python -m rasm_cim teach "Франция находится в Европа"
python -m rasm_cim ask "Париж находится в Европе?"           # INFERRED
python -m rasm_cim say "А кто написал Евгений Онегин?"       # контекст/местоимение
```

## 6. Если что-то не совпало

1. Сравнить `python --version` (ожидается 3.11.x).
2. Прогнать `python -m rasm_cim stress --json` и посмотреть `offline.violations`
   и `fuzz.crashes`.
3. Числа метрик зависят от корпусов: их sha256 напечатаны в отчётах
   (`BENCHMARK_REPORT.md`, `TEST_REPORT.md`). Другой корпус — другие числа.
