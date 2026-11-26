# ARCHITECTURE.md — RASM-CIM v0.15 (managed improvement)

Контур управляемого улучшения — строго реверситс, весь обучаемый state в файлах,
frozen-код и семантика TruthGraph9 не изменяются никогда. Мoдули:

## corpus_build.py — шаблонные смысловые семейства
Данные: demo corpus (34 факта) × РУКОПИСНЫЕ ЧАСТИ-DAME_TEMPLATES (54 семейства),
каждый split зафиксирован ДО генерации: train 265 / dev 84 / test 88 примеров,
семейство целиком в одном split (анти-leakage). manifest.json: sha256 каждого
файла + corpus_sha256. Split policy детерминированная; seed=0.

## artifacts.py — rasm-cim-learned/1
Артефакт = полный снапшот: intent_fallback (токен×интент counts), predicate_hints
(normalized question → predicate, из relation-коррекций), alias_delta, config
(alpha=0.5, tau=0.6, use_fallback_on_unknown), corrections[] (id свёрнутых событий),
seed, corpus_sha256, parent, metrics. Версии lv_000… со ссылкой parent; rollback =
переустановка current.json на сохранённый файл (истории никогда не переписываются).

## intent_fallback — Naive Bayes text→intent
Обучен на train (+свёрнутые intent-коррекции); применяется ТОЛЬКО когда v0.11
analyze() возвращает unknown и fallback-margin ≥ tau (0.6). Инертно к любым другим
ответам: вопросы с известным intent вообще не проходят через fallback.

## pipeline15.py — единственные два пути
1. нет hint-подсказки → v0.14 ask() без изменений (reasoning EvidenceSet давит);
2. есть predicate_hint для normalize(question) → узкий путь _answer_with_hint:
   тот же EvidenceSet напрямую из KB (evidence ids на каждом предложении).
   Коррекция «relationship» = переутверждение маршрутизации, не факта.
Ответы всё ещё только из подтверждённых данных; fallback НИКОГДА не утверждает
содержание — только выбирает маршрут.

## corrections.py — события коррекции (hash-chain, tamper-evident)
correction event: correction_id, ts, kind (intent|alias|entity|answer|relation),
wrong/correct, target_text, session_id, artifact_version, prev_hash/event_hash.
record → validate_event (ambiguity control алиасов: запрещено связывать занятую
поверхность с другим объектом) → apply_corrections → НОВАЯ версия артефакта +
метрики до/после на dev + проверка regression-пробы (6 сценариев через реальный
pipeline) + список конфликтов (одинаковая цель, разные значения — оба сохраняются,
конфликт показывается, применяется последний по времени).
Undo: `rollback <version>` — указатель current.json перевыставляется; ничего не
стирается.

## Инварианты
1. Ничто обучаемое не пишет в KB; коррекция effect проверяем.
2. Источник каждого числа — файл или замер; metrics лежат в самом артефакте.
3. Test-split не участвовал в выборе hyperparameters (tau=0.6 выбран раз и не настраивался).
