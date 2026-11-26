# REPRODUCE.md — точные команды (v0.10, 2026-11-26)

## 0. Чистая среда (host или Termux)
```bash
python3 -V                      # ≥3.10 (проверено на 3.13.18)
pip install -r requirements-dev.txt     # pytest hypothesis atheris (numpy — опция)
pip install -e .                        # + console-script rasm-cim-demo
```
Внимание: в окружениях без persistence pip-пакетов (песочницы/CI) ставьте заново —
без hypothesis `tests_v9/test_m5_props.py` даёт collection error (это воспроизведённый факт,
не баг кода).

## 1. Тесты (все 163)
```bash
python -m pytest test_rasm*.py -q                    # 57  (v0.1–v0.7 корневые)
python -m pytest rasm_cim_v8/tests -q                # 52  (v0.8)
python -m pytest tests_v9 -q                         # 40  (v0.9: m1_m2, props, fuzz-регрессии, граф)
python -m pytest tests_v10 -q                        # 14  (kill-compact, ctx-manager)
```

## 2. Целостность истории
```bash
md5sum -c results_v9/v08_freeze_manifest.md5 | grep -c ': OK'   # 51
find rasm_cim_rust -name '*.rs' -exec md5sum {} + | sort | md5sum
# 451bbfa52e67a72b04c0046d9df4e9d5
```

## 3. Замеры
```bash
python bench_v10/profile_core.py            # полная матрица ~10 мин → results_v10/profile_matrix.json
python bench_v10/profile_core.py --quick    # 1k+20k, 2 повтора (Termux-щадящий)
python bench_v9/m1_m2_bench.py              # v0.9-матрица (сравнение v2/v3, residue, update)
python bench_v9/m3_tuning.py                # LRU-нагрузки/warm/mmap/декомпозиция/scan
python -m rasm_cim_v9.cli bench --n 50000   # 15-сек витрина p50/p95/p99
python -m rasm_cim_v9.cli crash             # SIGKILL-демо восстановления
```
Методологические правила (встроены в скрипты): изолированный subprocess на ячейку,
медиана из 3 повторов + min/max, одинаковые seed/данные/пробы для v0.8 и v0.10,
tracemalloc включается ТОЛЬКО в memory-ячейках (он кратно искажает латентность —
зафиксировано и исправлено в этом же этапе).

## 4. Фаззинг
```bash
python bench_v9/fuzz_m5_atheris.py header journal records corruptmid compact 120
# корпусы: rasm_cim_v9/fuzz/corpus_* ; крэш-артефакты: rasm_cim_v9/fuzz/crashes/ (2, исторические)
# регрессии найдены/закрыты: python -m pytest tests_v9/test_m5_regressions.py -q
```

## 5. Устройство (Termux, ARM64)
```bash
# телефон: распакуйте архив в ~/rasm_cim_v9_pkg
bash rasm_cim_v9_pkg/rasm_cim_v9/mobile/termux_run_v09.sh          # r3-харнесс, изоляция arms
python3 - <<'EOF'   # извлечение НЕ напечатанных arms из СЫРЫХ r2-JSON (закрывает ≤5ms-гейт)
import json, glob, os
for f in sorted(glob.glob(os.path.expanduser('~/rasm_v9_device_results/v09_ab_*.json'))):
    d = json.load(open(f))
    for k, v in d.items():
        if isinstance(v, dict) and "picks" not in v:
            print(f.split('/')[-1], "|", k, ":", {a: b for a, b in v.items() if a != "picks"})
EOF
```
Присланные JSON кладите в `results_v10/device/` (отдельно от host-чисел — правило версий).

## 6. Архив/поставка
```bash
zip -qr RASM-CIM_v0.1-v0.10_PYTHON_FULL_$(date +%F).zip test_rasm*.py rasm_cim* \
  docs_v* results_v* plots* tests_v9 tests_v10 bench_v9 bench_v10 \
  README.md pyproject.toml requirements-dev.txt -x '*__pycache__*' -x '*.pyc' \
  -x '*.zip'
unzip -t <архив> && python -m pytest ... (после распаковки — см. §1)
```
