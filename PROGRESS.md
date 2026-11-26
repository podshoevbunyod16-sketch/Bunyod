# PROGRESS.md — RASM-CIM v0.10 → v0.15 (ТЗ от 26.11.2026)

Рабочая ветка: `arena/aa17ff3d-bunyod`. Базовый commit: f784e76 (main, v0.10).
Фактическая среда этой сессии: Linux x86-64, 2 CPU, Python 3.11.2 (не 3.13.18 из docs_v10!),
numpy 2.4.6, scikit-learn 1.9.1, hypothesis/pytest переустановлены (pip в песочнице не
персистит).

## Базовая линия (до изменений)
- 163 исторических теста зелёные (57+52+40+14) — воспроизведено.
- v0.8 freeze 51/51 OK; Rust v0.7 hash 451bbfa5… OK.
- Создан baseline freeze: results_v11/freeze_manifest_baseline_v0.10.md5 (272 файла).

## Этапы
- [x] **v0.11** — языковой слой: language/ (normalize/tokenize/morphology/intent/models),
  адаптер к TruthGraph9, CLI language-check/ask-text, корпус 161 пример (split по
  семействам), 56 новых тестов, итого 219 passed. Отчёты: docs_v11/, results_v11/.
  Честные пробелы: morphology = псевдо-стемминг; intent на 7 классов; корпус
  тем-генерирован; dev-ошибка «Добрый вечерок» сохранена как измеренная.
- [x] v0.12 — concepts/entities/relations + provenance + Teach/Remember API
- [x] v0.13 — диалоговый контекст, местоимения, мульти-вопросы
- [x] **v0.14** — EvidenceSet pipeline (direct/derived/missing, категориальная confidence,
  правила R01-R03 + KB-BFS depth≤3, composition ТОЛЬКО из EvidenceSet + evidence-check gate);
  verdict (TRUE/FALSE/UNKNOWN/CONFLICT) отделён от confidence (supported/partial/unknown/conflict);
  307 passed (163 ист. + 56 + 27 + 40 + 21); freeze 0 mismatch; Rust unchanged;
  bench: p50≈236–343µs, RSS 22.7 MB, 11-вопросный accuracy 1.0 / unsupported 0.0 / abstention 1.0;
  seed-demo type repair (персонажи — единый тип person).
- [x] v0.15 — корпус 200–500/50–100/50–100, обучение, correction loop, offline eval
- [x] **Финал (v0.15 packaging)** — unified CLI `python -m rasm_cim` (chat/ask/teach/memory/
  evaluate/system-info; stdout=результаты, stderr=диагностика; exit 0/2/3/4/5; кодучи 0-5;
  `--demo` строго opt-in; `python -m rasm_cim_v9.cli` сохранён), README переписан под фактическое
  состояние, pyproject → 0.15.0 (packages v8…v15 + rasm_cim, rasm-cim entry), requirements-dev
  и pyproject[dev] дополнены scikit-learn (свежей средой обнаружено: frozen v0.5 tests import
  sklearn), numpy (frozen v0.1 rasm_cim/ импортирует numpy). Freeze: v0.8 манифест 0 mismatch;
  baseline-272 → ровно 3 задокументированных исключения (README.md, pyproject.toml,
  requirements-dev.txt — см. results_v15/freeze_delta_report_v0.10_to_v0.15.md); rasm_cim/__init__.py
  (frozen v0.1) восстановлен byte-identical после случайного касания и повторно верифицирован
  по md5. v0.15-layer bug fix: LearnedArtifacts pointer (current.json) игнорировал out_dir и утёк в tracks-файл; исправлено + регрессионный тест добавлен. Полный гейт: 332 passed. Архив/PR — в работе.

## Правила, принятые в этой работе
- Исторические версии и Rust v0.7 не изменяются; новые пакеты — аддитивны.
- test не используется для настройки правил (corpus: train/dev/test по family-split).
- Честные отчёты: не скрывать FAIL/NO IMPROVEMENT; числа только измеренные.
- После каждого этапа: полный регрессионный прогон + freeze-проверка; commit/push.
