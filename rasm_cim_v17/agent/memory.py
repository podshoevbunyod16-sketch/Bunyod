"""Память агента: словарный запас и очередь фактов на подтверждение.

Два разных хранилища, и это принципиально:

* **словарный запас** (`vocab`) — слова, которые агент встретил, с
  определением, источником и уверенностью. Слово попадает туда либо из
  обучения пользователем, либо из проверенного источника;
* **предложения фактов** (`pending`) — то, что агент извлёк из источника, но
  ещё **не** записал в граф. В граф факт попадает только после подтверждения
  человеком или после подтверждения из второго независимого источника. Так
  «запомнил» не превращается в «сохранил случайную фразу из интернета».
"""
from __future__ import annotations

import json
import os
import re
import time

STATE = "agent_memory.json"


class AgentMemory:
    def __init__(self, home: str, clock=None):
        self.home = home
        self.path = os.path.join(home, STATE)
        self._clock = clock or (lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
        self.vocab: dict[str, dict] = {}
        self.pending: list[dict] = []
        self.seen: dict[str, int] = {}
        self.load()

    # ------------------------------------------------------------------ io
    def load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return
        self.vocab = payload.get("vocab", {}) or {}
        self.pending = payload.get("pending", []) or []
        self.seen = payload.get("seen", {}) or {}

    def save(self) -> None:
        payload = {"version": 1, "vocab": self.vocab, "pending": self.pending,
                   "seen": self.seen}
        try:
            os.makedirs(self.home, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=1,
                          sort_keys=True)
        except OSError:
            pass

    # -------------------------------------------------------------- словарь
    @staticmethod
    def key(word: str) -> str:
        return (word or "").strip().casefold().replace("ё", "е")

    def note_word(self, word: str) -> bool:
        """Отметить встреченное слово. True — если оно новое для агента."""
        key = self.key(word)
        if not key or len(key) < 3:
            return False
        self.seen[key] = self.seen.get(key, 0) + 1
        return key not in self.vocab and self.seen[key] == 1

    def defines(self, word: str) -> bool:
        return self.key(word) in self.vocab

    def define(self, word: str, definition: str, *, source: str,
               confidence: float = 0.6, kind: str = "word") -> dict:
        key = self.key(word)
        old = self.vocab.get(key)
        if old and old.get("definition", "").strip() == definition.strip():
            old["confirmations"] = old.get("confirmations", 1) + 1
            if source and source not in (old.get("source") or ""):
                sources = [s for s in (old.get("source") or "").split(" | ") if s]
                if source not in sources:
                    sources.append(source)
                old["source"] = " | ".join(sources)
            self.save()
            return old
        entry = {"word": word.strip(), "definition": definition.strip(),
                 "source": source, "confidence": round(confidence, 2),
                 "kind": kind, "learned_at": self._clock(), "confirmations": 1}
        self.vocab[key] = entry
        self.save()
        return entry

    def describe(self, word: str) -> dict | None:
        return self.vocab.get(self.key(word))

    def words(self) -> list[dict]:
        return sorted(self.vocab.values(), key=lambda item: item["word"].casefold())

    def unknown_words(self, limit: int = 20) -> list[str]:
        """Слова, которые встречались, но определения у агента нет."""
        return [word for word, count in
                sorted(self.seen.items(), key=lambda kv: (-kv[1], kv[0]))
                if count >= 1 and word not in self.vocab][:limit]

    # ---------------------------------------------------------------- факты
    def propose(self, subject: str, obj: str, *, source: str = "",
                sentence: str = "") -> dict | None:
        """Поставить факт в очередь. Дубликаты не добавляются."""
        subject, obj = (subject or "").strip(), (obj or "").strip()
        if not subject or not obj:
            return None
        for item in self.pending:
            if self.key(item["subject"]) == self.key(subject) and \
                    self.key(item["object"]) == self.key(obj):
                if source and source not in (item.get("sources") or []):
                    item.setdefault("sources", []).append(source)
                    item["confirmations"] = len(item["sources"])
                    self.save()
                return item
        item = {"subject": subject, "object": obj, "predicate": "instance_of",
                "sources": [source] if source else [], "sentence": sentence.strip(),
                "confirmations": 1 if source else 0, "proposed_at": self._clock()}
        self.pending.append(item)
        self.save()
        return item

    def pending_facts(self) -> list[dict]:
        return list(self.pending)

    def drop_pending(self) -> None:
        self.pending = []
        self.save()

    def confirm(self, teach) -> tuple[list[dict], list[dict]]:
        """Записать подтверждённые факты через `teach(text) -> bool`.

        Возвращает (записанные, отвергнутые). Словарь пополняется только для
        записанных фактов: у класса появляется определение «родовой термин
        для <субъект>» с источником.
        """
        written, rejected = [], []
        for item in self.pending:
            subject, obj = item["subject"], item["object"]
            source = ", ".join(item.get("sources") or []) or "пользователь"
            text = f"{subject} — это {obj}"
            if teach(text):
                written.append(item)
                self.define(obj, f"родовой термин для «{subject}»",
                            source=source,
                            confidence=0.75 if item.get("sources") else 0.6,
                            kind="class")
                continue
            # ядро принимает только односложный класс: «Москва — это столица
            # России» оно разобрать не может. Тогда записываем головное слово,
            # а полную формулировку сохраняем в словаре — и говорим об этом.
            head = re.split(r"\s+", obj.strip())[0] if obj.strip() else ""
            user_supplied = not item.get("sources")
            if head and head != obj and user_supplied and \
                    teach(f"{subject} — это {head}"):
                item = dict(item)
                item["object"] = head
                item["full_phrase"] = obj
                written.append(item)
                self.define(obj, f"то, чем является «{subject}»", source=source,
                            confidence=0.6, kind="phrase")
                self.define(head, f"родовой термин для «{subject}»",
                            source=source, confidence=0.6, kind="class")
                continue
            rejected.append(item)
        self.pending = [item for item in self.pending if item in rejected]
        self.save()
        return written, rejected
