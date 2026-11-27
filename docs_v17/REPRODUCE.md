# REPRODUCE v0.17

Всё воспроизводится офлайн, без сети и без установки зависимостей, кроме
`pytest` (для запуска тестов). Python ≥ 3.10 (проверено на 3.11.2).

## 1. Проверить, что дерево зелёное

```bash
cd Bunyod
python -m pytest tests_v16 -q                    # ожидается 348 passed
python -m pytest tests_v9 tests_v10 -q           # ожидается 54 passed
python -m pytest tests_v17 -q                    # ожидается 91 passed
python -m pytest test_rasm.py test_rasm_v2.py test_rasm_v3.py \
      test_rasm_v4.py test_rasm_v6.py test_rasm_v6_v2.py rasm_cim_v8/tests -q
                                                 # ожидается 102 passed
```

`test_rasm_v5.py` требует sklearn и в прогон не входит (см. LIMITATIONS).

## 2. Единый CLI

```bash
python -m rasm_cim version          # или: python -m rasm_cim_v17 version
python -m rasm_cim --help
```

Ожидается `rasm-cim v0.17.0 (ядро v0.16.0, схема kb 2)`.

## 3. Импорт реального корпуса v0.11

```bash
export RASM_CIM_HOME=/tmp/rasm-demo
python -m rasm_cim import tests_v17/fixtures/v11_demo_facts.jsonl --dry-run
python -m rasm_cim import tests_v17/fixtures/v11_demo_facts.jsonl
python -m rasm_cim ask "Где столица Франция?"
python -m rasm_cim verify
```

Ожидается: dry-run печатает «34 к импорту, 0 пропусков» и список новых
отношений (`capital_of`, `boils_at`, `has_satellite`, `is_capital_of`,
`satellite_of`); импорт — «+34 утверждения»; `verify` — «ПРОВЕРКА: OK».

Повторный импорт должен напечатать «ИМПОРТ не выполнялся: источник уже
импортирован».

## 4. Откат импорта

```bash
python -m rasm_cim imports --json | python -c "import json,sys; print(json.load(sys.stdin)['entries'][0]['id'])"
python -m rasm_cim restore <ledger_id>
python -m rasm_cim stats
```

## 5. Импорт каталога v0.12 (`rasm-cim-kb/1`)

```bash
# каталог должен содержать meta.json и events.jsonl
python -m rasm_cim import /path/to/kb --dry-run --verbose
python -m rasm_cim import /path/to/kb
```

## 6. Сводка обучающего артефакта v0.15

```bash
python -m rasm_cim learned /path/to/rasm_cim_v15/learned
python -m rasm_cim import  /path/to/rasm_cim_v15/learned   # подсказки + алиасы
```

Непроверенный по sha256 артефакт печатает `MISMATCH` и ничего не применяет.

## 7. Независимая проверка формата

`tests_v17/conftest.py` содержит писатель `rasm-cim-kb/1` (`write_legacy_kb`) —
минимальный, но совместимый с v0.12. Сгенерировать такой каталог и прочитать
его можно, не имея v0.12 под рукой:

```python
from tests_v17.conftest import write_legacy_kb, legacy_entity, legacy_statement
```

## Ожидаемое время

Полный набор v0.17 — меньше 2 секунд; все наборы вместе — около 15 секунд на
машине разработчика (x86_64). Замеров на ARM64/Termux нет.
