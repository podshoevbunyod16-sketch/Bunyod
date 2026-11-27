"""ResponseIR — what an answer says, before it becomes a sentence.

Schema ``rasm-cim-response-ir/1``.  Deliberately small: a segment is a claim, an
evidence line, a qualifier, a refusal or a clarification, and it must carry the
ids it was built from (except refusals and clarifications, which stand on the
absence of facts).

``from_dict`` validates on the way in, because a plan that has been written to
disk and read back is a *format* — and formats need a reader with opinions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

RESPONSE_SCHEMA = "rasm-cim-response-ir/1"

SEGMENT_KINDS = ("claim", "evidence", "qualifier", "refusal", "clarification",
                 "correction", "reply")
#: segments that must be backed by ids
EVIDENCED_KINDS = ("claim", "evidence", "qualifier", "correction")
REGISTERS = ("short", "normal", "detailed")


class ResponseIRError(ValueError):
    """The plan is not a valid ResponseIR."""


@dataclass
class Segment:
    kind: str
    text: str
    fact_ids: list[str] = field(default_factory=list)
    statement_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.kind not in SEGMENT_KINDS:
            raise ResponseIRError(f"неизвестный вид сегмента: {self.kind!r}")
        if not (self.text or "").strip():
            raise ResponseIRError("пустой сегмент")
        self.fact_ids = [str(x) for x in self.fact_ids if x]
        self.statement_ids = [str(x) for x in self.statement_ids if x]

    @property
    def sourced(self) -> bool:
        return bool(self.fact_ids or self.statement_ids)

    def as_dict(self) -> dict:
        return {"kind": self.kind, "text": self.text,
                "fact_ids": list(self.fact_ids),
                "statement_ids": list(self.statement_ids),
                "sourced": self.sourced}


@dataclass
class ResponseIR:
    speech_act: str
    status: str
    segments: list[Segment]
    register: str = "normal"
    referents: list[str] = field(default_factory=list)
    plan_notes: list[str] = field(default_factory=list)
    schema: str = RESPONSE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RESPONSE_SCHEMA:
            raise ResponseIRError(f"неизвестная схема: {self.schema!r}")
        if self.register not in REGISTERS:
            raise ResponseIRError(f"неизвестный регистр: {self.register!r}")
        if not self.segments:
            raise ResponseIRError("план без сегментов")
        validate(self)

    # -- queries -----------------------------------------------------------
    def of_kind(self, kind: str) -> list[Segment]:
        return [s for s in self.segments if s.kind == kind]

    @property
    def claims(self) -> list[Segment]:
        return self.of_kind("claim")

    @property
    def unsourced(self) -> list[Segment]:
        return [s for s in self.segments
                if s.kind in EVIDENCED_KINDS and not s.sourced]

    @property
    def statement_ids(self) -> list[str]:
        seen: list[str] = []
        for s in self.segments:
            for sid in s.statement_ids:
                if sid not in seen:
                    seen.append(sid)
        return seen

    def as_dict(self) -> dict:
        return {"schema": self.schema, "speech_act": self.speech_act,
                "status": self.status, "register": self.register,
                "segments": [s.as_dict() for s in self.segments],
                "referents": list(self.referents),
                "plan_notes": list(self.plan_notes),
                "statement_ids": self.statement_ids}

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict) -> "ResponseIR":
        if not isinstance(data, dict):
            raise ResponseIRError("ожидался объект")
        try:
            # ``sourced`` is derived and written for readers; the dataclass
            # recomputes it, so an extra key must not be a format error
            fields = {"kind", "text", "fact_ids", "statement_ids"}
            segments = [Segment(**{k: v for k, v in seg.items() if k in fields})
                        for seg in data["segments"]]
            return cls(speech_act=data["speech_act"], status=data["status"],
                       segments=segments,
                       register=data.get("register", "normal"),
                       referents=list(data.get("referents", [])),
                       plan_notes=list(data.get("plan_notes", [])),
                       schema=data.get("schema", RESPONSE_SCHEMA))
        except KeyError as exc:
            raise ResponseIRError(f"в плане нет поля {exc.args[0]!r}") from exc
        except TypeError as exc:
            raise ResponseIRError(f"сегмент не соответствует схеме: {exc}") from exc

    @classmethod
    def from_json(cls, text: str) -> "ResponseIR":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ResponseIRError(f"не JSON: {exc}") from exc
        return cls.from_dict(data)


def validate(ir: ResponseIR) -> None:
    """Invariants every plan must hold, checked once at construction."""
    claims = ir.claims
    if ir.status in ("UNKNOWN", "AMBIGUOUS"):
        if claims:
            raise ResponseIRError("отказ не может содержать утверждение")
        if not (ir.of_kind("refusal") or ir.of_kind("clarification")):
            raise ResponseIRError("отказ без сегмента отказа")
        return
    if not claims and not ir.of_kind("reply"):
        raise ResponseIRError("ответ без утверждения")
    if ir.unsourced:
        kinds = ", ".join(s.kind for s in ir.unsourced)
        raise ResponseIRError(f"сегменты без источника: {kinds}")
