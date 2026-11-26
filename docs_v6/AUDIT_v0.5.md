# Независимый аудит v0.5 (2026-11-25)

## Что реально работает (доказано)
- HypergraphKB query/compose/arith на малых примерах (N1)
- HierarchicalMemory exact/partial/noisy до 100k (N3)
- Replay снижает forgetting (N4: 0.33->0 на Task1)
- NEXTGEN math/parity/fact/memory/program на малых задачах (N5)
- 45 тестов passed

## Ограничения / ошибки (найдены аудитом)
1. NEXTGEN confidence=1.0 захардкожен без калибровки (system5.py ask) — нарушение §8.
2. LUTComputer: precompute 101x101 не включён в mul time; max_val=100 жёстко.
3. HypergraphKB.arithmetic_hyperedge: a%M, b%M — silent wrap для больших значений.
4. HierarchicalMemory L2 только для модулей 97/89/83; L3 HV случайные (не семантические); tau=0.5 без калибровки.
5. Reservoir фиксированный случайный; firstbit T2=0.56 (забывает ранние входы).
6. Synthesis BFS — перебор по фиксированным опкодам (не открытое изобретение); ARG_VALUES ограничен.
7. Тесты синтетические малые; независимых датасетов нет; LLM-базлайн отсутствует (честно, но сравнение неполно).
8. Утечек train/test не найдено (разные seed), но test малые и из того же распределения — обобщение не доказано.

## Вывод аудита
v0.5 — рабочий прототип точной памяти/арифметики/композиции, не универсальный интеллект.
v0.6 исправляет confidence, LUT-учёт, wrap, калибровку tau, добавляет Arch A/B/C и честные ablation/benchmarks.
