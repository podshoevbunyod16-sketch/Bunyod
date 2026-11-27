"""v0.16 — dialogue context: topics, active entities, pronoun resolution.

Spec §7 is explicit that the referent of a pronoun must NOT be "the last
noun".  This engine therefore keeps:

  * a *salience-ordered* set of active entities (recency + grammatical role),
  * per-entity gender (from the KB, explicit or heuristic) so that "он" and
    "она" cannot resolve to the wrong person,
  * a topic stack so that switching to another subject and coming back does
    not mix the two sets of entities (TEST F),
  * the last event referenced, so an elliptical "А когда?" can be bound to a
    concrete statement — and an honest clarification when it cannot,
  * the pending clarification state, so the system can finish a question it
    asked instead of silently dropping it.

Pronoun resolution outcomes:
    RESOLVED       exactly one plausible referent
    AMBIGUOUS      >=2 referents agree in gender and salience -> must ask
    UNRESOLVED     no active entity can take the pronoun -> must ask
Never a random pick.

Everything here is serialisable (``to_dict``/``from_dict``), versioned and
migratable — TEST G requires the context to survive a full process restart.
"""
from __future__ import annotations

import itertools
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .entity_resolver import (FEMININE, MASCULINE, NEUTER, PLURAL, Entity,
                              EntityResolver, infer_gender)
from .language_parser import PRONOUN_GENDER

CONTEXT_SCHEMA = 2

PR_RESOLVED = "RESOLVED"
PR_AMBIGUOUS = "AMBIGUOUS"
PR_UNRESOLVED = "UNRESOLVED"

#: salience decay per elapsed turn, and the role bonuses
DECAY = 0.12
ROLE_BONUS = {"subject": 0.30, "object": 0.15, "answer": 0.20, "mention": 0.05}
#: two candidates closer than this are considered tied -> AMBIGUOUS
TIE_MARGIN = 0.10

MAX_TURNS = 200          # episodic history cap (bounded memory)
MAX_ACTIVE = 40          # active-entity cap
MAX_TOPICS = 20


@dataclass
class Topic:
    topic_id: str
    label: str
    entity_ids: list[int] = field(default_factory=list)
    opened_at: str = ""
    last_active_at: str = ""
    turns: int = 0

    def as_dict(self) -> dict:
        return {"id": self.topic_id, "label": self.label,
                "entity_ids": list(self.entity_ids), "opened_at": self.opened_at,
                "last_active_at": self.last_active_at, "turns": self.turns}

    @classmethod
    def from_dict(cls, d: dict) -> "Topic":
        return cls(topic_id=d["id"], label=d.get("label", ""),
                   entity_ids=list(d.get("entity_ids", [])),
                   opened_at=d.get("opened_at", ""),
                   last_active_at=d.get("last_active_at", ""),
                   turns=int(d.get("turns", 0)))


@dataclass
class Turn:
    turn_id: str
    role: str                        # "user" | "system"
    text: str
    intent: str = ""
    entity_ids: list[int] = field(default_factory=list)
    statement_ids: list[str] = field(default_factory=list)
    answer_status: str = ""
    created_at: str = ""

    def as_dict(self) -> dict:
        return {"id": self.turn_id, "role": self.role, "text": self.text,
                "intent": self.intent, "entity_ids": list(self.entity_ids),
                "statement_ids": list(self.statement_ids),
                "answer_status": self.answer_status, "created_at": self.created_at}

    @classmethod
    def from_dict(cls, d: dict) -> "Turn":
        return cls(turn_id=d["id"], role=d.get("role", "user"),
                   text=d.get("text", ""), intent=d.get("intent", ""),
                   entity_ids=list(d.get("entity_ids", [])),
                   statement_ids=list(d.get("statement_ids", [])),
                   answer_status=d.get("answer_status", ""),
                   created_at=d.get("created_at", ""))


@dataclass
class PronounResolution:
    pronoun: str
    status: str
    entity_id: int | None = None
    candidates: list[int] = field(default_factory=list)
    confidence: float = 0.0
    question: str = ""
    reason: str = ""

    def as_dict(self) -> dict:
        return {"pronoun": self.pronoun, "status": self.status,
                "entity_id": self.entity_id, "candidates": list(self.candidates),
                "confidence": round(self.confidence, 3), "question": self.question,
                "reason": self.reason}


@dataclass
class ActiveEntity:
    entity_id: int
    salience: float
    last_turn: int
    role: str = ""
    gender: str = ""
    gender_source: str = ""

    def as_dict(self) -> dict:
        return {"id": self.entity_id, "salience": round(self.salience, 4),
                "last_turn": self.last_turn, "role": self.role,
                "gender": self.gender, "gender_source": self.gender_source}

    @classmethod
    def from_dict(cls, d: dict) -> "ActiveEntity":
        return cls(entity_id=int(d["id"]), salience=float(d.get("salience", 1.0)),
                   last_turn=int(d.get("last_turn", 0)), role=d.get("role", ""),
                   gender=d.get("gender", ""),
                   gender_source=d.get("gender_source", ""))


class ContextEngine:
    """Dialogue state.  Owns no knowledge — only what is being talked about."""

    def __init__(self, resolver: EntityResolver | None = None, clock=None,
                 session_id: str = "s1", max_turns: int = MAX_TURNS):
        self.resolver = resolver
        self._clock = clock or (lambda: "")
        self.session_id = session_id
        self.max_turns = max_turns
        self._tcount = itertools.count(1)
        self.turn_no = 0
        self.history: list[Turn] = []
        self.active: "OrderedDict[int, ActiveEntity]" = OrderedDict()
        self.topic = Topic(topic_id="t-1", label="", opened_at=self._clock())
        self.topic_stack: list[Topic] = []
        #: last referenced event, for elliptical follow-ups ("А когда?")
        self.last_event: dict = {}
        #: last (subject, predicate, object) that the dialogue was about
        self.focus: dict = {}
        self.pending_clarification: dict = {}
        self.pronoun_log: list[dict] = []

    # ------------------------------------------------------------------
    # turns
    # ------------------------------------------------------------------
    def new_turn_id(self) -> str:
        n = next(self._tcount)
        self.turn_no = max(self.turn_no, n)
        return f"u-{n:05d}"

    def record_user_turn(self, text: str, intent: str = "",
                         entity_ids: Sequence[int] = (), when: str = "") -> Turn:
        return self._record("user", text, intent, entity_ids, [], "", when)

    def record_system_turn(self, text: str, intent: str = "",
                           entity_ids: Sequence[int] = (),
                           statement_ids: Sequence[str] = (),
                           answer_status: str = "", when: str = "") -> Turn:
        return self._record("system", text, intent, entity_ids,
                            statement_ids, answer_status, when)

    def _record(self, role, text, intent, entity_ids, statement_ids,
                answer_status, when) -> Turn:
        when = when or self._clock()
        t = Turn(turn_id=self.new_turn_id(), role=role, text=text, intent=intent,
                 entity_ids=list(entity_ids), statement_ids=list(statement_ids),
                 answer_status=answer_status, created_at=when)
        self.history.append(t)
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]
        return t

    # ------------------------------------------------------------------
    # active entities / salience
    # ------------------------------------------------------------------
    def touch(self, entity_id: int, role: str = "mention", when: str = "") -> None:
        when = when or self._clock()
        self._decay_all()
        ent = self.resolver.get(entity_id) if self.resolver else None
        prev = self.active.get(entity_id)
        base = prev.salience if prev else 0.0
        sal = min(1.0, max(base, 0.55) + ROLE_BONUS.get(role, 0.05))
        ae = ActiveEntity(entity_id=entity_id, salience=sal,
                          last_turn=self.turn_no, role=role,
                          gender=(ent.gender if ent else "") or
                                 (infer_gender(ent.canonical) if ent else ""),
                          gender_source=(ent.gender_source if ent else ""))
        self.active[entity_id] = ae
        self.active.move_to_end(entity_id)
        while len(self.active) > MAX_ACTIVE:
            self.active.popitem(last=False)
        if entity_id not in self.topic.entity_ids:
            self.topic.entity_ids.append(entity_id)
            if len(self.topic.entity_ids) > 12:
                self.topic.entity_ids = self.topic.entity_ids[-12:]
        self.topic.last_active_at = when
        self.topic.turns += 1
        if not self.topic.label and ent is not None:
            self.topic.label = ent.canonical

    def _decay_all(self) -> None:
        for ae in self.active.values():
            gap = self.turn_no - ae.last_turn
            if gap > 0:
                ae.salience = max(0.0, ae.salience - DECAY * gap)
                ae.last_turn = self.turn_no

    def salience_ranked(self) -> list[ActiveEntity]:
        return sorted(self.active.values(),
                      key=lambda a: (-a.salience, a.entity_id))

    def active_entity_ids(self) -> list[int]:
        return [a.entity_id for a in self.salience_ranked() if a.salience > 0.05]

    def forget_inactive(self, threshold: float = 0.05) -> int:
        drop = [k for k, v in self.active.items() if v.salience <= threshold]
        for k in drop:
            del self.active[k]
        return len(drop)

    # ------------------------------------------------------------------
    # topics
    # ------------------------------------------------------------------
    def topic_covers(self, entity_ids: Sequence[int]) -> bool:
        cur = set(self.topic.entity_ids)
        stacked = {e for t in self.topic_stack for e in t.entity_ids}
        return bool(cur & set(entity_ids)) or bool(stacked & set(entity_ids))

    def classify_topic(self, entity_ids: Sequence[int]) -> str:
        """``same`` | ``resume`` | ``new`` — what a turn about these entities
        should do to the topic stack.  Checked BEFORE any entity is touched,
        otherwise ``touch`` would already have written the new entities into
        the old topic and the switch would never trigger (TEST F)."""
        ids = {e for e in entity_ids if e is not None}
        if not ids:
            return "same"
        if ids & set(self.topic.entity_ids):
            return "same"
        for t in self.topic_stack:
            if ids & set(t.entity_ids):
                return "resume"
        return "new"

    def switch_topic(self, entity_ids: Sequence[int], label: str = "",
                     when: str = "") -> str:
        """Push the current topic and start a new one.  Returns the action:
        ``new`` | ``resumed`` | ``same``."""
        when = when or self._clock()
        ids = list(entity_ids)
        if ids and set(ids) & set(self.topic.entity_ids):
            return "same"
        for i, t in enumerate(self.topic_stack):
            if ids and set(ids) & set(t.entity_ids):
                resumed = self.topic_stack.pop(i)
                self.topic_stack.append(self.topic)
                self.topic = resumed
                self.topic.last_active_at = when
                self.topic.turns += 1
                while len(self.topic_stack) > MAX_TOPICS:
                    self.topic_stack.pop(0)
                return "resumed"
        # an empty, nameless topic is not worth stacking — that is what made
        # the stack show a "(без имени)" entry on the first switch
        if self.topic.entity_ids or self.topic.label:
            self.topic_stack.append(self.topic)
        while len(self.topic_stack) > MAX_TOPICS:
            self.topic_stack.pop(0)
        self.topic = Topic(topic_id=f"t-{len(self.topic_stack)+1}",
                           label=label, entity_ids=list(ids),
                           opened_at=when, last_active_at=when)
        return "new"

    def return_to_previous(self) -> Topic | None:
        """Pop the most recent stacked topic and push the current one down."""
        if not self.topic_stack:
            return None
        prev = self.topic_stack.pop()
        self.topic_stack.append(self.topic)
        self.topic = prev
        self.topic.last_active_at = self._clock()
        self.topic.turns += 1
        return self.topic

    # ------------------------------------------------------------------
    # pronouns
    # ------------------------------------------------------------------
    def resolve_pronoun(self, pronoun: str, resolver: EntityResolver | None = None
                        ) -> PronounResolution:
        """Bind a pronoun to an active entity, or refuse and ask."""
        res = PronounResolution(pronoun=pronoun, status=PR_UNRESOLVED)
        gender = PRONOUN_GENDER.get(pronoun.lower(), "")
        if not gender:
            res.reason = f"{pronoun!r} is not a known pronoun form"
            return res

        ranked = [a for a in self.salience_ranked() if a.salience > 0.05]
        if not ranked:
            res.question = ("О ком или о чём идёт речь? Назови, пожалуйста, "
                            "сущность целиком.")
            res.reason = "no active entity in the context"
            return res

        agreeing, unknown_gender = [], []
        for a in ranked:
            g = a.gender or self._gender_of(a.entity_id, resolver)
            if not g:
                unknown_gender.append(a)
            elif g == gender or (gender == PLURAL) or (g == PLURAL):
                agreeing.append(a)

        pool = agreeing or (unknown_gender if len(ranked) == 1 else [])
        if not pool:
            names = ", ".join(self._name(a.entity_id, resolver) for a in ranked[:3])
            res.candidates = [a.entity_id for a in ranked[:5]]
            res.question = (f"Я не могу отнести «{pronoun}» ни к одной из "
                            f"сущностей в контексте ({names}). Уточни, о ком речь?")
            res.reason = (f"no active entity matches the required "
                          f"{'plural' if gender == PLURAL else gender} form")
            return res

        top = pool[0]
        tied = [a for a in pool[1:] if top.salience - a.salience <= TIE_MARGIN]
        if tied:
            names = " или ".join(
                self._name(a.entity_id, resolver) for a in [top] + tied[:3])
            res.status = PR_AMBIGUOUS
            res.candidates = [top.entity_id] + [a.entity_id for a in tied]
            res.question = f"Ты имеешь в виду {names}?"
            res.reason = (f"{len(tied)+1} candidates agree in gender and are "
                          f"within the salience tie margin {TIE_MARGIN}")
            self.pronoun_log.append(res.as_dict())
            return res

        res.status = PR_RESOLVED
        res.entity_id = top.entity_id
        res.candidates = [top.entity_id]
        res.confidence = 0.85 if agreeing else 0.6
        res.reason = ("single salience-dominant candidate agreeing in gender "
                      f"({gender})" if agreeing else
                      "only one active entity and its gender is unknown")
        self.pronoun_log.append(res.as_dict())
        return res

    def _gender_of(self, entity_id: int, resolver: EntityResolver | None) -> str:
        r = resolver or self.resolver
        ent = r.get(entity_id) if r else None
        if ent is None:
            return ""
        return ent.gender or infer_gender(ent.canonical)

    def _name(self, entity_id: int, resolver: EntityResolver | None) -> str:
        r = resolver or self.resolver
        ent = r.get(entity_id) if r else None
        return ent.canonical if ent else f"#{entity_id}"

    # ------------------------------------------------------------------
    # focus / ellipsis
    # ------------------------------------------------------------------
    def set_focus(self, subject_id: int | None = None, predicate: str = "",
                  object_id: int | None = None, statement_id: str = "",
                  when: str = "") -> None:
        f = dict(self.focus)
        if subject_id is not None:
            f["subject_id"] = subject_id
        if predicate:
            f["predicate"] = predicate
        if object_id is not None:
            f["object_id"] = object_id
        if statement_id:
            f["statement_id"] = statement_id
        f["turn"] = self.turn_no
        self.focus = f

    def set_last_event(self, statement_id: str, subject_id: int,
                       predicate: str, object_id: int, when: str = "") -> None:
        self.last_event = {"statement_id": statement_id,
                           "subject_id": subject_id, "predicate": predicate,
                           "object_id": object_id, "turn": self.turn_no,
                           "at": when or self._clock()}

    def has_last_event(self) -> bool:
        return bool(self.last_event.get("statement_id"))

    def set_pending(self, question: str, kind: str, payload: dict | None = None,
                    when: str = "") -> None:
        self.pending_clarification = {"question": question, "kind": kind,
                                      "payload": dict(payload or {}),
                                      "turn": self.turn_no,
                                      "at": when or self._clock()}

    def take_pending(self) -> dict:
        p = dict(self.pending_clarification)
        self.pending_clarification = {}
        return p

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def reset(self, keep_topics: bool = False) -> None:
        self.history = []
        self.active = OrderedDict()
        self.last_event = {}
        self.focus = {}
        self.pending_clarification = {}
        self.pronoun_log = []
        self.turn_no = 0
        self._tcount = itertools.count(1)
        if not keep_topics:
            self.topic = Topic(topic_id="t-1", label="", opened_at=self._clock())
            self.topic_stack = []

    def integrity_check(self) -> list[str]:
        """Return a list of problems (empty == healthy)."""
        problems = []
        seen = set()
        for t in self.history:
            if t.turn_id in seen:
                problems.append(f"duplicate turn id {t.turn_id}")
            seen.add(t.turn_id)
        ids = [a.entity_id for a in self.active.values()]
        if len(ids) != len(set(ids)):
            problems.append("duplicate entity in the active set")
        if self.resolver is not None:
            for a in list(self.active.values()):
                if self.resolver.get(a.entity_id) is None:
                    problems.append(f"active entity #{a.entity_id} is not in the KB")
            for t in self.topic_stack + [self.topic]:
                for e in t.entity_ids:
                    if self.resolver.get(e) is None:
                        problems.append(f"topic {t.topic_id} references unknown "
                                        f"entity #{e}")
        if self.turn_no < len(self.history):
            problems.append(f"turn_no {self.turn_no} < history length "
                            f"{len(self.history)}")
        return problems

    # -- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "schema": CONTEXT_SCHEMA,
            "session_id": self.session_id,
            "turn_no": self.turn_no,
            "history": [t.as_dict() for t in self.history],
            "active": [a.as_dict() for a in self.active.values()],
            "topic": self.topic.as_dict(),
            "topic_stack": [t.as_dict() for t in self.topic_stack],
            "last_event": dict(self.last_event),
            "focus": dict(self.focus),
            "pending_clarification": dict(self.pending_clarification),
            "pronoun_log": list(self.pronoun_log[-50:]),
        }

    def load_dict(self, d: dict) -> list[str]:
        """Load a serialised context.  Returns migration notes."""
        notes = migrate_context_dict(d)
        self.session_id = d.get("session_id", self.session_id)
        self.turn_no = int(d.get("turn_no", 0))
        self.history = [Turn.from_dict(t) for t in d.get("history", [])][-self.max_turns:]
        self.active = OrderedDict()
        for a in d.get("active", []):
            ae = ActiveEntity.from_dict(a)
            self.active[ae.entity_id] = ae
        self.topic = Topic.from_dict(d.get("topic", {"id": "t-1"}))
        self.topic_stack = [Topic.from_dict(t) for t in d.get("topic_stack", [])]
        self.last_event = dict(d.get("last_event", {}))
        self.focus = dict(d.get("focus", {}))
        self.pending_clarification = dict(d.get("pending_clarification", {}))
        self.pronoun_log = list(d.get("pronoun_log", []))
        self._tcount = itertools.count(self.turn_no + 1)
        return notes

    @classmethod
    def from_dict(cls, d: dict, resolver: EntityResolver | None = None,
                  clock=None) -> "ContextEngine":
        ce = cls(resolver=resolver, clock=clock)
        ce.load_dict(d)
        return ce


def migrate_context_dict(d: dict) -> list[str]:
    """In-place upgrade of an older context document.  Idempotent.

    v1 (pre-0.16) shape had no topics, no salience and a flat ``entities``
    list; it is upgraded losslessly — nothing is dropped, only re-labelled.
    """
    notes: list[str] = []
    ver = int(d.get("schema", 1))
    if ver >= CONTEXT_SCHEMA:
        return notes
    if "active" not in d and "entities" in d:
        act = []
        for i, e in enumerate(d.get("entities", [])):
            eid = e if isinstance(e, int) else int(e.get("id", 0))
            act.append({"id": eid, "salience": max(0.05, 1.0 - 0.1 * i),
                        "last_turn": int(d.get("turn_no", 0)), "role": "mention",
                        "gender": "", "gender_source": ""})
        d["active"] = act
        notes.append(f"v{ver}->v{CONTEXT_SCHEMA}: flat 'entities' list converted "
                     f"to salience-ordered 'active' ({len(act)} entries kept)")
    if "topic" not in d:
        ids = [a["id"] for a in d.get("active", [])][:12]
        d["topic"] = {"id": "t-1", "label": "", "entity_ids": ids,
                      "opened_at": "", "last_active_at": "",
                      "turns": len(d.get("history", []))}
        d.setdefault("topic_stack", [])
        notes.append(f"v{ver}->v{CONTEXT_SCHEMA}: topic tracking added, "
                     f"{len(ids)} entities carried over")
    d.setdefault("last_event", {})
    d.setdefault("focus", {})
    d.setdefault("pending_clarification", {})
    d.setdefault("pronoun_log", [])
    d["schema"] = CONTEXT_SCHEMA
    return notes
