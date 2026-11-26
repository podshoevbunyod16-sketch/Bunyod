# Обязательный предварительный аудит v0.1–v0.6 (для v0.7 Rust)

**Дата:** 2026-11-25. Составлен ДО программирования Rust, по требованию §1.

## 1. Зависимости между версиями

```
v0.1 (rasm_cim): rns, fpe, cam, state(SSM), resonator, logic, system
  └─> v0.2 (rasm_cim_v2): rns2(PackedRNS), cam2, ssm2(selective), resonator2, logic2, program2, tools, system2
        └─> v0.3 (rasm_cim_v3): core, memory3, ssm3(BPTT), resonator3, program3(BFS), arch A/B/C
              └─> v0.4 (rasm_cim_v4): rns4(range-aware), graph4(TruthGraph), program4(split), ca4(CA+reservoir), archD, benchmark
                    └─> v0.5 (rasm_cim_v5): knowledge5(HypergraphKB), compute5(LUT/CA/BFS), memory5(Hierarchical), learn5(replay), system5(NEXTGEN)
                          └─> v0.6 (rasm_cim_v6_v2): memory_system, archA/B/C, selective_ssm, llm_compare, compute_core(PackedStore), capabilities, profiles
```

## 2. Карта компонентов к переносу

| Компонент | Python-источник | Нужен ли Rust | Причина |
|---|---|---|---|
| RNS арифметика + CRT | rns.py, rns2.py, rns4.py | **ДА** | Целочисленные ops, горячий путь |
| Exact/partial/noisy memory | cam.py, cam2.py, memory5.py, memory_system.py | **ДА** | Поиск, индексы, масштабирование |
| TruthGraph / HypergraphKB | graph4.py, knowledge5.py | **ДА** | Графы, обходы |
| CA dense/sparse | ca4.py | **ДА** | Параллельные обновления |
| Reservoir (ESN) | ca4.py EchoStateReservoir | **ДА** | Матричные ops, динамика |
| Program synthesis BFS | program3.py, program4.py | **ДА** | Поиск, перебор |
| PackedStore | compute_core.py | **ДА** | Хранение, RAM |
| Selective SSM BPTT | ssm3.py, selective_ssm.py | **ЧАСТИЧНО** | numpy BPTT; Rust без autograd — только inference |
| Resonator networks | resonator.py, resonator2/3 | **НЕТ** | Экспериментальный, numpy-зависимый |
| FPE (Fractional Power Encoding) | fpe.py | **НЕТ** | numpy-зависимый, не в ядре v0.6 |
| LLM compare / sklearn | llm_compare.py | **НЕТ** | Внешние инструменты, не ядро |
| Profiles/benchmarks harness | profiles*.py | **НЕТ** | Python-оркестрация |

## 3. Таблица соответствия Python → Rust

| Python | Rust | Примечание |
|---|---|---|
| `dict[int, list]` (exact memory) | `HashMap<i64, Vec<i64>>` | std HashMap; для скорости — `FxHashMap` (не добавляем зависимость без нужды) |
| `defaultdict(set)` (inverted index) | `HashMap<(usize, i64), HashSet<i64>>` | |
| `numpy.ndarray` (HV, reservoir) | `Vec<f64>` / `Vec<f32>` | Без ndarray в ядре; f64 для точности |
| `numpy.random` | `rand` crate (ChaCha) | Детерминированный seed |
| `struct.pack("<qq")` (PackedStore) | `to_le_bytes()` / `from_le_bytes()` | Zero-copy через `bytemuck`-подобный подход вручную |
| `OverflowError` / `ValueError` | `thiserror` enum `RasmError` | Типизированные ошибки |
| `tracemalloc` / `time.perf_counter` | `std::time::Instant` + `criterion` | Benchmarks |
| `json` (results) | `serde` + `serde_json` | Сериализация результатов |
| `multiprocessing` / numpy threads | `rayon` | Параллельные итераторы |
| PyO3 bindings | `pyo3` (abi3) | Python → Rust вызовы |

## 4. Известные ошибки и ограничения (сохранены)

1. **v0.1:** LogicMemory.query всегда возвращает лучший факт (нет UNKNOWN); ProgramMemory.decode теряет порядок; `eval()` в logic.py (заменён AST whitelist в v0.2); DiagonalSSM не влияет на CAM; RNS tuple без преимущества памяти.
2. **v0.2:** context_id argmax коллизии (zeros→0, ones→0); selective SSM вредит cumsum (R² 0.235 vs 1.0); packed RNS 8 B/item vs binary 2 B/item — нет преимущества; HV float32 1024 B/item >> RNS.
3. **v0.3:** BPTT parity train 0.79/test 0.49 (не решает parity); BFS length-4 требует max_nodes=50000; Python getsizeof overhead >> packed.
4. **v0.4:** H8 коллизии 2000→1155 (small-M residues как ключи); GraphMemory перезапись; parity sum cheat; synthesis без split.
5. **v0.5:** confidence=1.0 без калибровки; LUT precompute не в тайминге; RNS silent wrap (a%M); L2 только 97/89/83; reservoir firstbit 0.56; BFS фиксированные опкоды; нет LLM-базлайна.
6. **v0.6:** selective SSM 0.46 < RNN 0.475 на parity (L1 REFUTED); reservoir забывает ранние входы; BFS экспоненциален; Python overhead >> packed (5.8×); SNN/differentiable/Bayesian не реализованы.

## 5. Что НЕ переносится (с объяснением)

- **Resonator networks** — экспериментальные, numpy-зависимые, не в ядре v0.6; перенос без numpy даст другую числовую точность.
- **FPE** — numpy-зависимый кодировщик, не используется в Arch A/B/C v0.6.
- **Selective SSM BPTT training** — требует autograd; в Rust только inference (forward). Обучение остаётся в Python (numpy BPTT).
- **LLM compare / sklearn** — внешние инструменты; Rust не нужен.
- **Profiles harness** — Python-оркестрация; Rust даёт только измерения через criterion.

## 6. Вывод аудита

Переносу подлежат: RNS, Memory (exact/partial/noisy/semantic/episodic/procedural), TruthGraph/Hypergraph, CA, Reservoir, BFS synthesis, PackedStore. Именно они дают измеримый выигрыш (целые ops, поиск, параллелизм). Остальное — Python-оркестрация или numpy-зависимые эксперименты.
