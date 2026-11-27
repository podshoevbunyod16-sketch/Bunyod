"""Расширенная память диалога для v0.17: кто и когда упоминался.

Зачем отдельная структура.  Грунтование местоимения требует ответа на вопрос
«звучало ли это в этом разговоре», а `ContextEngine` v0.16 хранит только
текущую активность с затуханием: после двух-трёх нейтральных реплик от
антецедента остаётся salience 0.1, а после `reset()` не остаётся ничего.
Память диалога — счётчик упоминаний с номером хода, а не вес; она не затухает
внутри сессии и переживает ходы, которые парсер не разобрал.

Ограничение `MAX_ITEMS` нужно, чтобы длинный диалог не рос бесконечно: при
переполнении выбрасываются записи с самым ранним последним упоминанием.
"""
from __future__ import annotations

MAX_ITEMS = 400


class DialogueMemory:
    """Упоминания сущностей в разговоре: id → (счётчик, первый, последний ход)."""

    def __init__(self, max_items: int = MAX_ITEMS):
        self.max_items = max_items
        self.entries: dict[int, dict] = {}
        self.dropped = 0

    # -- запись ------------------------------------------------------------
    def note(self, entity_id: int | None, surface: str = "", turn: int = 0) -> None:
        if not entity_id:
            return
        entity_id = int(entity_id)
        entry = self.entries.get(entity_id)
        if entry is None:
            self.entries[entity_id] = {"count": 1, "first": turn, "last": turn,
                                       "surfaces": [surface] if surface else []}
            self._trim()
            return
        entry["count"] += 1
        entry["last"] = turn
        if surface and surface not in entry["surfaces"][-4:]:
            entry["surfaces"].append(surface)

    def _trim(self) -> None:
        while len(self.entries) > self.max_items:
            oldest = min(self.entries, key=lambda k: self.entries[k]["last"])
            del self.entries[oldest]
            self.dropped += 1

    # -- чтение ------------------------------------------------------------
    def ids(self) -> set[int]:
        return set(self.entries)

    def count(self, entity_id: int) -> int:
        return int(self.entries.get(int(entity_id), {}).get("count", 0))

    def last_turn(self, entity_id: int) -> int:
        return int(self.entries.get(int(entity_id), {}).get("last", -1))

    def recent(self, limit: int = 5) -> list[int]:
        ordered = sorted(self.entries, key=lambda k: self.entries[k]["last"],
                         reverse=True)
        return ordered[:limit]

    def summary(self) -> dict:
        return {"unique": len(self.entries), "dropped": self.dropped,
                "mentions": sum(e["count"] for e in self.entries.values())}

    # -- сериализация ------------------------------------------------------
    def to_dict(self) -> dict:
        return {"max_items": self.max_items, "dropped": self.dropped,
                "entries": {str(k): v for k, v in self.entries.items()}}

    @classmethod
    def from_dict(cls, data: dict) -> "DialogueMemory":
        memory = cls(max_items=int(data.get("max_items", MAX_ITEMS)))
        memory.dropped = int(data.get("dropped", 0))
        memory.entries = {int(k): dict(v)
                          for k, v in (data.get("entries") or {}).items()}
        return memory
