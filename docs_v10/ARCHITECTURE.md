# ARCHITECTURE.md — RASM-CIM v0.10 (фактическая, по коду этого дерева)

## Слои и потоки данных

```
 естественный язык ─────────────✗ (v0.11+ roadmap; сейчас нет)

 CLI (rasm_cim_v9/cli.py, python -m rasm_cim_v9.cli)
   build | ask | partial | update | bench | crash(демо) 
        │
 MobileMemory9 (rasm_cim_v9/memory/store9.py)   ярусы: stream / residue / hv
   query_exact ──► V3Store.get ──► bisect(resident index) ─► page ─► crc32 ─► record
   query_partial ► ResidueIndex (dict[(pos,res)] -> array('q'), моды 97/89/83)
   query_noisy   ► StreamHV (hv: gen/block/block_numpy/combo|auto; кэш 0-inic)
   update/txn    ► append_txn (журнал)
        │
 V3Store (rasm_cim_v9/storage/journal.py)
   resident: base keys array('q') + offs array('Q') (16 B/rec) + journal arrays
   read: pread|mmap; PageCache (LRAMixin v0.8); on_bad raise|skip; ctx-manager
   journal: кадры <QII(seq,count,crc32(body))> + body; commit=кадр целиком
   replay:  header-bounds(fstat) → short=тorn tail(игнор) → crc → seq-дедуп(R3.3)
            mid-file bad crc(+преемники) → JournalCorruptError
   compact: scan(base+journal) → build_v3_base(.new) → os.replace (атом; kill-safe)
        │
 Файл v3 (rasm_cim_v9/storage/index.py)
   [40B header: magic|ver=3|hdr_size|page_recs|n_base|index_off|base_end + crc32]
   [data pages = payload(≤2048×16B)+crc32(4B)+pad(4B) — математика v2]
   [index pages <qQ> sorted (key,off)]  [journal frames...]
   commit точки: header_crc (база), кадр-crc (журнал). build: tmp + os.replace.
        │
 read-only опора на v0.8 (не меняется): storage/format (v1/v2), storage/streaming
 (LazyStore/PageCache), memory/stream_memory, hv/bitpack + memory/hv_stream.
 Семантика get (TRUE/UNKNOWN/CONFLICT=sorted(set)) сохранена 1:1 (16 тестов).

 Память фактов/граф: TruthGraph9 (rasm_cim_v9/graph.py) — формализованный
 assert_true/assert_false/retract(symmetric|legacy), история append-only.
 Совместимость с замороженным v0.4 доказана исчерпывающими A/B-переборами.
```

## Что оптимизировано против v0.8
- сборка: генератор + 2 плоских массива, без списка кортежей (peak @1M 88 vs 142 MiB);
- сортировка: numpy-lexsort при n≥200k (import по требованию), иначе two-pass order-sort;
- residue: array-корзины (−62% RAM @1M), строятся из уже отсортированных ключей;
- update: журнал (O(кадр)), а не пересборка файла (1.0 s → 0.03 ms медиана host/tmpps);
- hot-key LRU (opt-in): ×5.7–6.6 на skew-нагрузках, ×0.75–0.95 на uniform/scan (замерено).

## Файлы v0.10 (новые): cli.py, tests_v10/, bench_v10/profile_core.py,
## pyproject.toml, requirements-dev.txt; в journal.py — __enter__/__exit__.
