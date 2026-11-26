# RASM-CIM v0.2 — план исправлений (от аудита v0.1 от 25.11.2026)

## Аудит v0.1: что найдено в коде

| # | Проблема v0.1 | Где в коде | Исправление v0.2 |
|---|---|---|---|
| 1 | RNS без преимущества памяти (Python tuple overhead, биты не посчитаны) | `rns.py` encode->tuple | PackedRNS на numpy uint16, подсчёт бит/байт vs binary vs HV; честный вывод: RNS ≥ binary по битам, выигрыш только параллелизм |
| 2 | CAM — только симуляция, выдаваемая за «CAM-like» | `cam.py` search_cam_vectorized | CAM2 с inverted index (частичный поиск), порогом отказа, Hopfield, счётчиком операций; явная маркировка SIMULATION, теоретическая модель площади/энергии отдельно |
| 3 | SSM не интегрирована (только cumsum/parity отдельно) | `state.py`, `system.py` | SelectiveSSM + контекст как часть ключа CAM; абляция доказывает влияние |
| 4 | Resonator не в процессе (только отдельный тест) | `resonator.py` | Resonator2 с confidence+fallback, используется в Logic и Program путях |
| 5 | End-to-end ограничен (только key->value) | `system.py` | RASMCIMv2: факты+программы+последовательности+противоречия+инструменты в одном пайплайне |
| 6 | Программы — brute force по малым комбинациям, порядок теряется | `logic.py` ProgramMemory.decode | Композиционные программы 1–3 оп. как последовательность HV, пошаговое декодирование резонатором |
| 7 | LogicMemory не отказывает неизвестным (всегда возвращает лучший) | `logic.py` query | Порог tau, калибровка на unknown, метрики FP/FN, UNKNOWN |
| 8 | `eval()` в reference_apply и demo | `logic.py`, `demo.py` | Безопасный AST-интерпретатор whitelist, без eval/exec; тест на инъекцию |

## Архитектура v0.2 (интегрированная)

```
Вход (int / факт / программа / последовательность / запрос инструмента)
  │
  ├─ RNS encoder (packed residues) ──────────────┐
  │                                               ▼
  ├─ Selective SSM (контекст) ──► context_id ──► CAM2 write
  │   (lam_t = sigmoid(Wx), h=tanh(lam*h+Bx))    (residues+context -> value,
  │                                               inverted index + HV store)
  ▼
Запрос ──► CAM2 search (exact / partial / noisy+tau)
  │            │ UNKNOWN если sim<tau
  │            ▼
  ├─ Resonator2 factorize (confidence, fallback brute)
  │            ▼
  ├─ Logic2: query / compose (2-step) / contradiction check
  ├─ Program2: decode 1–3 ops ──► SafeInterpreter (AST, whitelist)
  └─ Tools: CAM-select tool ──► validated call
  │
  ▼
Выход + confidence + счётчики операций + ablation-флаги
```

Каждый компонент: входы/выходы/метрики задокументированы в docstring. Абляция: отключение любого компонента должно ухудшить хотя бы одну задачу (проверяется в F7).

## Этапы реализации
1. `rns2.py` PackedRNS + binary baseline + op counter
2. `cam2.py` CAM2 inverted index + tau + Hopfield
3. `ssm2.py` SelectiveSSM + trainable readout
4. `resonator2.py` confidence + fallback
5. `logic2.py` refusal + contradiction + compose via resonator
6. `program2.py` compositional + safe interpreter (no eval)
7. `tools.py` tool registry via CAM
8. `system2.py` RASMCIMv2 integrated + ablation
9. `experiments2.py` F1–F10 + `demo2.py` + `test_rasm_v2.py`
10. Отчёт, схема SVG, ZIP v0.2

## Критерий успеха
Доказанное улучшение ≥1 характеристики без необоснованного ухудшения остальных.
Кандидаты: отказ от неизвестных (FP↓), композиционные программы unseen (acc↑), контекст-зависимый recall (CAM alone fail → +SSM success), частичный поиск (ops↓ vs full scan).
