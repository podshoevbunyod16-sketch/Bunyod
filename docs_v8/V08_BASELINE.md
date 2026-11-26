# V08_BASELINE.md — зафиксированные базовые показатели ДО изменений v0.8

Все замеры — эта же машина (Linux x86_64, 2 CPU, 1982 MiB RAM), 2026-11-25.
v0.8 их НЕ пересчитывает и НЕ правит; они — точка отсчёта для M2/M5/M6.

## 1. Тесты/совместимость (до начала v0.8)

| Набор | Результат |
|---|---|
| pytest `test_rasm*.py` (v0.1–v0.6, 57 тестов) | 57 passed, 3 warnings |
| Rust v0.7 (заморожен) `cargo test --release` | 32 unit + 6 property — не запускается от имени v0.8; статус взят из v0.7 отчёта |
| MD5 всех .rs v0.7 (src+bindings) | `recorded-at-audit; superseded see below` |

## 2. Python-side время (свежий прогон того же harness, что и v0.7)

| Метрика | Значение |
|---|---|
| MemorySystem build, 5 000 фактов | 0.0802 s |
| MemorySystem exact 5 000 | 0.0021 s |
| PackedStore 50k put / get | 0.0224 / 0.0158 s |
| Noisy 200 queries D=256 (numpy float HV) | 0.1725 s |
| **MemorySystem build 100k (D=256 float HV)** | **1.813 s, peak RSS 296.6 MiB** |
| Exact-query loop 100k×10k выборок | 0.0096 s, hits 10000/10000 |
| results_v7/bench_v7.json (v0.7, та же машина) | py medians: rns 9.98 ms, mem-build 81.6 ms (5k), noisy 181 ms, packed 22.4 ms |

Ключевой вывод базлайна: **float-HV D=256 = 2 KB/запись** ⇒ 100k уже 296 MiB
RSS. Цель M2 (<40 MiB при 100k) из этого базлайна недостижима без смены
представления (bit-pack 32 B/запись) или без постраничного чтения — это и есть
экспериментальный вопрос v0.8, а не гарантированный результат.

## 3. Исторические константы, которые v0.8 обязан воспроизвести (regression)

- BFS synthesis на v0.6-задаче: программа `add 2 → mul 3`, candidates **238**.
- Sparse CA (rule 90) на паттерне v0.6 D1 (активные 10/20/30, N=64): ops **9** vs dense 64.
- Память: dedup hit, второй уникальный value ⇒ CONFLICT; неизвестный ключ ⇒ UNKNOWN.
- РНS (3,5,7,11,13): M=15015; encode(15015) ⇒ OverflowError; signed −7507..7507.
- Формат v1: `[RASM][ver=1][len][16B×n records][FNV64]` (ARCHITECTURE §5 v0.7).

## 4. Ошибки/уточнения, найденные НА АУДИТЕ v0.8 (до кода)

1. **Уточнение к v0.7 ERRORS #7 (не переписывание, а коррекция факта):**
   расходящихся паттернов у legacy-таблицы rule 110 — **5** (011,100,101,110,111),
   не 3: в v0.7 перечислены только 101/110/111; таблица ca4.py
   `[0,1,1,0,1,0,0,1]` vs канон `(110>>p)&1` = `[0,1,1,1,0,1,1,0]`.
   На исторические результаты это не влияет (всегда rule 90), но влияет на
   объём M3-таблицы переходов. Исправление живёт в v0.8 (`rule110_canonical`),
   замороженный Rust не правится.
2. На хосте нет `/sys/class/thermal` и `cpufreq` — M1-замеры температуры на
   Android возможны, здесь — нет; harness делает это различие явным.

## 5. Среда v0.8 (зафиксировано)

python 3.13.18 · numpy 2.3.5 · **hypothesis 6.170.17 · atheris (установлен —
coverage-guided возможен)** · **numba 0.66.0** · psutil 7.2.2 · sklearn 1.6.1 ·
mmap/os.pread/resource — stdlib, доступны.


**FREEZE PROOF (reproducible):** `find rasm_cim_rust -name '*.rs' -exec md5sum {} + | sort | md5sum` = `451bbfa52e67a72b04c0046d9df4e9d5` (23 files; identical at v0.8 start AND at v0.8 close — the raw hash alone; the v0.1–v0.7 ZIP archive diff is the authoritative check). No .rs file has an mtime newer than the v0.8 start marker.

### 4.2 v0.8 test-writing findings (additions to §4, same audit trail)
- `write_episodic` returns **True on DUPLICATE-suppressed** (not on stored) — v0.8 tests now pin the source contract verbatim.
- TruthGraph `retract` discards from `true_edges` only (false-assertions survive retraction) — recorded as observed behavior; frozen v0.4 untouched; PLAN_v0.9 item 6.
- rule110 module uses **circular boundaries**; legacy/canonical tables are tuples indexed by pattern (v0.8 test pins both).
