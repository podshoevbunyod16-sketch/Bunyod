"""ContextStore (v0.13) — bounded, TTL'd, serializable dialogue state.

Design contract:
  * Context is NOT the knowledge base. It carries conversation state only:
    turns, active topic, recent entities (with gender guess), open questions,
    provenance links (statement ids) — never new facts of its own.
  * Offline; single JSON file; writes are atomic (tmp file + os.replace + fsync).
  * Bounded: max_turns (default 20, FIFO eviction), ttl_seconds (default 1800).
    Expired context is treated as empty and persisted as empty on next save.
  * Sensitive data: turns may contain user text; `export_session` ships them
    only on explicit request; diagnostics never print turn contents.
"""
from __future__ import annotations

import json
import os
import time
from typing import List, Optional

SCHEMA = "rasm-cim-context/1"


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def calendar_timegm_safe(s: str) -> float:
    import calendar
    try:
        return calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ"))
    except Exception:
        return 0.0


class ContextStore:
    def __init__(self, path: str, max_turns: int = 20, ttl_seconds: int = 1800):
        self.path = path
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds
        self._suspend = False
        self.state: dict = self._empty()
        self.load()

    # ------------------------------------------------------------- (de)ser
    def _empty(self) -> dict:
        return {"schema": SCHEMA, "created_at": utcnow(), "updated_at": utcnow(),
                "active_topic": None, "topic_stack": [],
                "recent_entities": [],          # [{name, gender, role, ts, kb_id?}]
                "open_questions": [],           # [{id, question, slot, missing, ts, candidates?}]
                "turns": []}                    # [{id, ts, role, text, intent, entities,
                                                #   statement_ids, mode}]

    def load(self) -> None:
        if not os.path.exists(self.path):
            self.state = self._empty()
            return
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("schema") != SCHEMA:
                raise ValueError(f"unsupported context schema {data.get('schema')!r}")
            self.state = data
        except Exception:
            # a corrupt context must never crash the dialogue — start empty,
            # keep the bad file for forensics under a new name
            try:
                os.replace(self.path, self.path + ".corrupt")
            except OSError:
                pass
            self.state = self._empty()
        if self._expired():
            self.state = self._empty()

    def save(self) -> None:
        if self._suspend:
            return
        self.state["updated_at"] = utcnow()
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.state, fh, ensure_ascii=False, indent=2)
            fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, self.path)

    def _expired(self) -> bool:
        ts = self.state.get("updated_at")
        epoch = calendar_timegm_safe(ts) if ts else 0.0
        return bool(epoch) and (time.time() - epoch) > self.ttl_seconds

    # ------------------------------------------------------------- turns
    def add_turn(self, role: str, text: str, *, intent: Optional[str] = None,
                 entities: Optional[List[str]] = None,
                 entity_ids: Optional[List[str]] = None,
                 statement_ids: Optional[List[str]] = None,
                 mode: str = "normal") -> dict:
        turn = {"id": len(self.state["turns"]) + 1, "ts": utcnow(), "role": role,
                "text": text, "intent": intent,
                "entities": list(entities or []),
                "entity_ids": list(entity_ids or []),
                "statement_ids": list(statement_ids or []),
                "mode": mode}
        self.state["turns"].append(turn)
        if len(self.state["turns"]) > self.max_turns:
            evicted = len(self.state["turns"]) - self.max_turns
            self.state["turns"] = self.state["turns"][evicted:]
        if role == "user":
            pass
        self.save()
        return turn

    def record_entities(self, entities: List[dict]) -> None:
        """entities: [{name, gender?, kb_id?, role?}]; MRU order, cap 12."""
        rec = self.state["recent_entities"]
        for ent in reversed(entities):
            item = {"name": ent["name"], "gender": ent.get("gender", "unknown"),
                    "kb_id": ent.get("kb_id"), "role": ent.get("role", "mentioned"),
                    "ts": utcnow()}
            key = lambda r: r.get("kb_id") or r["name"]
            ent_key = ent.get("kb_id") or ent["name"]
            rec = [r for r in rec if key(r) != ent_key]
            rec.insert(0, item)
        self.state["recent_entities"] = rec[:12]
        self.save()

    def set_topic(self, topic: Optional[str]) -> None:
        if topic and topic != self.state.get("active_topic"):
            stack = self.state["topic_stack"]
            if self.state.get("active_topic"):
                stack.append(self.state["active_topic"])
                self.state["topic_stack"] = stack[-5:]
            self.state["active_topic"] = topic
            self.save()

    def add_open_question(self, question: str, missing: str,
                          slot: str = "entity", candidates: Optional[List[str]] = None) -> dict:
        oq = {"id": f"oq_{int(time.time()*1000)}_{len(self.state['open_questions'])}",
              "question": question, "missing": missing, "slot": slot,
              "candidates": list(candidates or []), "ts": utcnow()}
        self.state["open_questions"].append(oq)
        self.state["open_questions"] = self.state["open_questions"][-10:]
        self.save()
        return oq

    def clear_open_questions(self) -> int:
        n = len(self.state["open_questions"])
        self.state["open_questions"] = []
        self.save()
        return n

    def suspend(self) -> None:
        """Batch mutations between suspend()/flush() — one fsync instead of four."""
        self._suspend = True

    def flush(self) -> None:
        self._suspend = False
        self.save()

    # ------------------------------------------------------------ exports
    def reset(self) -> None:
        self.state = self._empty()
        self.save()

    def export_session(self) -> dict:
        data = json.loads(json.dumps(self.state, ensure_ascii=False))
        data["schema"] = SCHEMA
        return data

    def summary(self) -> dict:
        return {"turns": len(self.state["turns"]),
                "active_topic": self.state["active_topic"],
                "topic_stack": list(self.state["topic_stack"]),
                "recent_entities": [e["name"] for e in self.state["recent_entities"]],
                "open_questions": len(self.state["open_questions"]),
                "max_turns": self.max_turns, "ttl_seconds": self.ttl_seconds,
                "updated_at": self.state.get("updated_at")}
