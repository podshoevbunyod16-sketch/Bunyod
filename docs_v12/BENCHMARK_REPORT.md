# BENCHMARK_REPORT.md — v0.12 (host, Linux x86-64, Python 3.11.2, tmpfs)

Метод: изолированный процесс-в-процесс вызов из bench-скрипта (tempfile в /tmp),
500 фактов is_a-класса (17 объектов), один прогон + JSON-артефакт. Числа orientation,
не SLA; устройство не измерялось (унаследованное ограничение).

| метрика | значение |
|---|---|
| build 500 фактов (teach, с provenance и fsync на событие) | 0.41 s (~1.2 ms/факт) |
| экспорт в JSONL | 457 104 B (~892 B/факт) |
| импорт (replay + append, включая dedup-проверки) | 0.44 s; 1017 событий |
| replay rate | ~1130 events/s |
| query predicate=is_a (материализация 500 Statement) | ~238 µs |
| open+replay пустого KB | < 5 ms |

Выводы/ограничения: teach() платит fsync на КАЖДОЕ событие — честная цена append-only.
Для батч-заливки нужен bulk-import (import_jsonl уже пишет строкой); снапшоты/индексы
за пределами v0.12. Latency-иерархия: v0.11 analyze() ~12–40 µs ≪ query_statements ~238 µs.
См. LIMITATIONS.
