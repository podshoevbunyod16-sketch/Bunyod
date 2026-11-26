# FINAL_REPORT.md — v0.11 → v0.15 (2026-11-26)

Честный итог работы по ТЗ «Managed Improvement / Compound Information Machine» поверх
замороженного ядра v0.10. Репозиторий: `podshoevbunyod16-sketch/Bunyod`, ветка
`arena/aa17ff3d-bunyod`. Все числа — измеренные на Linux-host; Android/Termux **не измерялся**.

## Что сделано (и проверено)

| Этап | Содержание | Тесты | Проверка |
|---|---|---|---|
| v0.11 (ada4dff) | Русский языковой слой: нормализация, псевдо-стемминг, интент (7 классов), CLI language-check/ask-text; корпус 161 пример, split по семействам | +56 | 219 passed |
| v0.12 (a72721f) | KB rasm-cim-kb/1: сущности/отношения/алиасы с контролем неоднозначности (без автомержа), provenance-цепочки source/doc/fragment/method/parser_version, append-only журнал, идемпотентный JSONL import/export; seed-demo repair | +27 | 246 passed |
| v0.13 (25c764a) | Диалог: контекст 20 реплик / TTL 30 мин / атомарный corrupt-safe файл; местоимения (gender/number); мульти-вопросы ≤8 с порядком; CLARIFY при неоднозначности; hypothetical/quote не сохраняются | +40 | 286 passed |
| v0.14 (e572322) | Reasoning pipeline: EvidenceSet (direct/derived/missing), категориальная confidence, R01–R03 + KB-BFS depth≤3, композиция ответа ТОЛЬКО из EvidenceSet, финальный evidence-check gate; verdict отдельно от confidence; 2 багфикса, найденные ТОЛЬКО held-out test | +21 | 307 passed |
| v0.15 (5b7f651) | Управляемое улучшение: корпус 265/84/88 (family-split, seed 0, sha256 manifest), learned-артефакты rasm-cim-learned/1 (parent-ссылки, sha256, откат), hash-chained corrections (verify/undo/конфликты показываются, контроль неоднозначности), offline eval (intent_acc dev 0.833→0.857 / test 0.761→0.784 — маргинально, не значимо; verdict_acc=1.0 на KB-answerable подмножестве) | +14 | 321 passed |
| Packaging (ee81645) | `python -m rasm_cim` (chat/ask/teach/memory/evaluate/system-info), stdout=результат/stderr=диагностика, exit 0/2/3/4/5, `--demo` строго opt-in; README переписан под факт; pyproject 0.15.0 (rasm-cim entry); freeze-дельта задокументирована | +10 CLI +1 регресс | **332 passed** |

Freeze-дисциплина: исторические версии v0.1–v0.10 и Rust v0.7 (sha256-set `451bbfa5…`) не изменялись.
v0.8 freeze-манифест: **0 mismatch**. Baseline-272 (v0.10-эпоха, `results_v11/freeze_manifest_baseline_v0.10.md5`,
никогда не переписывался): ровно **3 задокументированных исключения** — `README.md`, `pyproject.toml`,
`requirements-dev.txt` (все три — spec-mandated; см. `results_v15/freeze_delta_report_v0.10_to_v0.15.md`).

## Инциденты (зафиксированы, не скрыты)

1. Frozen v0.1 `rasm_cim/__init__.py` был случайно перезаписан каркасом unified-CLI → freeze-проверка поймала,
   восстановлен byte-identical (`git checkout HEAD -- rasm_cim/__init__.py`), md5 re-verified (9c19f403…).
   Unified-CLI живёт только в добавленных `rasm_cim/cli.py` / `rasm_cim/__main__.py`.
2. Свежая песочница потеряла pip-пакеты → переустановлены; discoveries: frozen v0.5 тесты импортируют
   **scikit-learn** (записан в dev-extras), frozen v0.1 пакет импортирует **numpy** (declared dependency).
3. `rasm_cim_v15/artifacts.py` игнорировал `out_dir` при записи `current.json` → dangling-указатель
   утекал в tracked-файл `rasm_cim_v15/learned/current.json` (замечен дважды: и в прошлом прогоне, и сейчас).
   Исправлено; регресс-тест добавлен; tracked-файл оставлен на committed-артефакте `lv_000`.
4. `python -m rasm_cim --demo` вызывал `rasm_cim_v12/cli.py` как скрипт → ImportError; исправлено на
   `python -m rasm_cim_v12.cli` с PYTHONPATH=_ROOT.

## Что НЕ сделано / пределы

- **Android/Termux НЕ измерялся.** Host-числа (диалог p50 ≈ 1.4 ms; reasoning p50 ≈ 236–343 µs,
  RSS ≤ 23 MB; v0.15 eval latency p50 ≈ 0.6–0.7 ms) получены на Linux-host и НЕ подставляются в
  device-гейты. Flash-fsync остаётся INCONCLUSIVE с v0.10. Device-прогон требует пользователя с устройством.
- **Никаких заявлений о понимании языка.** Это "не фабрикует неподтверждённое" + детерминизм + откат.
- intent/naive-bayes улучшение на синтетике (+2.3–2.4pp, 2–4 срабатывания fallback) — маргинально и статистически не значимо; НЕ заявлено как «модель стала лучше».
- Корпус — синтетика 54 семейства на 34 фактах demo-домена; morphology = псевдо-стемминг;
  интент — 7 классов; no таджикского языка; никаких LLM в ответном пути.
- Correction events могут содержать персональные данные; `corrections.jsonl` исключён из git.
- Hash-chain коррекций — аудит, не защита от перезаписи файла с пересчётом хэшей (нет внешнего witness).
- `docs_v15/TEST_REPORT.md` синхронизирован с финальным прогоном; исторические отчёты v0.1–v0.10 не переписывались.

## Как воспроизвести

См. `docs_v15/REPRODUCE.md` + `results_v15/final_full_gate.log`. Кратко:

```bash
pip install -r requirements-dev.txt           # pytest hypothesis atheris numpy scikit-learn
python -m pytest test_rasm*.py rasm_cim_v8/tests tests_v9 tests_v10 tests_v11 tests_v12 \
    tests_v13 tests_v14 tests_v15 tests_phase_c -q     # 332 passed
md5sum -c results_v9/v08_freeze_manifest.md5           # 0 mismatch
python -m rasm_cim --demo ask "Какая столица у Франции?"   # Франция: Париж [TRUE] [st_…]
python -m rasm_cim system-info
```

## Итог

Unified-CLI и упаковка готовы; 332 теста зелёные; freeze честно задокументирован.
PR в `main` несёт разницу «что сделано / что нет» (см. описание PR). Рекомендация перед merge:
принять 3 documented freeze-exceptions как spec-mandated; device-гейты оставить открытыми до
реального Termux-прогона.
