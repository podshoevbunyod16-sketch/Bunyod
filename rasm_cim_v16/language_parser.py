"""v0.16 — natural language understanding: intent + intermediate query form.

Pipeline (each stage separately testable, spec §5):

    text
      -> normalization.split_sentences / split_clauses     (segmentation)
      -> classify_intent                                   (what kind of act)
      -> segment                                           (subject | predicate | object)
      -> attach_predicate                                  (relation vocabulary)
      -> detect_negation / tense / modality
      -> QueryIR                                           (intermediate form)
      -> entity_resolver / knowledge_graph                 (execution, NOT here)

The parser NEVER writes to the KB.  A mis-parse can therefore produce a wrong
*answer*, but it cannot silently corrupt knowledge — ``relation_extractor``
decides separately whether an utterance is assertable, and applies its own
confidence rules before anything reaches the graph.

Case is preserved in ``subject_mention`` / ``object_mention`` on purpose: the
orthography ("Исаак Ньютон" vs "физик") is the cheapest reliable signal for
telling a proper name from a common noun, and the extractor needs it to
choose between ``alias_of`` and ``instance_of``.

``QueryIR`` is the contract between NLU and execution.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from .normalization import (DASH_TOKENS, normalize, split_clauses, squeeze,
                            tokens_keep_case, word_tokens)
from .relations import DEFAULT_SCHEMA, RelationSchema

# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------

#: wh-word -> (wants, semantic slot it asks for)
WH_WANTS = {
    "кто":   ("subject", "person"),
    "кем":   ("subject", "person"),
    "что":   ("object", "thing"),
    "чего":  ("object", "thing"),
    "чем":   ("object", "thing"),
    "какой": ("object", "class"),
    "какая": ("object", "class"),
    "какое": ("object", "class"),
    "какие": ("object", "class"),
    "какую": ("object", "class"),
    "какого": ("object", "class"),
    "когда": ("object", "time"),
    "где":   ("object", "place"),
    "куда":  ("object", "place"),
    "откуда": ("object", "place"),
    "почему": ("object", "cause"),
    "зачем": ("object", "cause"),
    "сколько": ("object", "quantity"),
}
WHO_WORDS = frozenset({"кто", "кем"})

PRONOUNS = {
    "он": "masculine", "его": "masculine", "ему": "masculine",
    "ним": "masculine", "нем": "masculine", "нём": "masculine",
    "него": "masculine", "сам": "masculine",
    "она": "feminine", "её": "feminine", "ее": "feminine", "ей": "feminine",
    "ею": "feminine", "ней": "feminine", "неё": "feminine", "нее": "feminine",
    "сама": "feminine",
    "оно": "neuter",
    "они": "plural", "их": "plural", "ими": "plural", "них": "plural",
    "им": "plural",
}
PRONOUN_SET = frozenset(PRONOUNS)
PRONOUN_GENDER = PRONOUNS

#: copula verbs / particles: "X это Y", "X есть Y", "X является Y", "X был Y"
COPULAS = frozenset({"это", "есть", "является", "являются", "был", "была",
                     "было", "были", "будет", "будут", "суть", "представляет"})

#: negation markers.  Multi-word ones are checked as substrings, single words
#: as tokens.
NEGATION_PHRASES = ("никогда не", "вовсе не", "отнюдь не", "не является",
                    "не являются", "не был", "не была", "не было", "не были",
                    "не стал", "не стала")
NEGATION_TOKENS = frozenset({"не", "нет", "неверно", "ложно"})

#: words that never belong to an entity mention
STOPWORDS = frozenset({
    "ли", "же", "ведь", "вот", "так", "тоже", "очень", "просто", "именно",
    "даже", "уже", "еще", "ещё", "только", "мне", "тебе", "нам", "вам",
    "ты", "вы", "мы", "я", "свой", "свои", "свою", "своей", "своим", "свое",
})

#: definition-request templates: (regex, kind)
DEFINITION_PATTERNS = (
    (r"^кто\s+тако[йе]\s+(?P<x>.+)$", "person"),
    (r"^кто\s+такая\s+(?P<x>.+)$", "person"),
    (r"^что\s+такое\s+(?P<x>.+)$", "thing"),
    (r"^расскажи(?:те)?\s+(?:мне\s+)?(?:о|про)\s+(?P<x>.+)$", "related"),
    (r"^что\s+ты\s+знаешь\s+(?:о|про)\s+(?P<x>.+)$", "related"),
    (r"^что\s+ты\s+(?:можешь\s+)?рассказать\s+(?:о|про)\s+(?P<x>.+)$", "related"),
    (r"^кто\s+это\s+(?P<x>.+)$", "person"),
    (r"^(?:подробнее|подробней)\s+(?:о|про)\s+(?P<x>.+)$", "related"),
    (r"^(?:а\s+)?что\s+ещё?\s+(?:о|про)\s+(?P<x>.+)$", "related"),
)

COMMAND_RE = re.compile(r"^:([a-zA-Z_]+)\s*(.*)$", re.DOTALL)
RETRACT_WORDS = frozenset({"забудь", "удали", "отмени", "retract", "forget"})
GREETING_WORDS = frozenset({"привет", "здравствуй", "здравствуйте", "hi", "hello"})

TENSE_PAST = frozenset({"был", "была", "было", "были", "стал", "стала",
                        "открыл", "написал", "создал", "сформулировал"})
TENSE_FUTURE = frozenset({"будет", "будут", "станет", "станут"})

#: negation words that must never leak into an entity mention
NEG_STRIP = frozenset({"не", "нет", "ни", "никогда"})

#: particles/prepositions stripped from the START of a mention
LEADING_PARTICLES = frozenset({
    "благодаря", "из-за", "для", "у", "про", "о", "обо", "со", "без",
    "после", "перед", "а", "и", "но", "же", "ли",
    "в", "во", "на", "из", "к", "от", "до", "при", "с", "по", "за", "над",
    "под", "между", "около", "возле", "среди", "ради", "кроме",
})

#: explicit teaching verbs (CLI-compatible, spec §12 TEST A)
TEACH_PREFIXES = ("teach", "научи", "запомни", "запомни-ка", "learn")

# --------------------------------------------------------------------------
# intents
# --------------------------------------------------------------------------

INT_TEACH = "TEACH"
INT_RETRACT = "RETRACT"
INT_ASK_DEFINITION = "ASK_DEFINITION"
INT_ASK_WH = "ASK_WH"
INT_ASK_YESNO = "ASK_YESNO"
INT_COMMAND = "COMMAND"
INT_GREETING = "GREETING"
INT_UNKNOWN = "UNKNOWN"

DEFINITION_PREDICATE = "instance_of"


@dataclass
class QueryIR:
    """Intermediate representation of one clause."""

    raw: str = ""
    norm: str = ""
    intent: str = INT_UNKNOWN
    wh: str = ""
    subject_mention: str = ""          # case preserved
    object_mention: str = ""           # case preserved
    subject_pronoun: str = ""
    object_pronoun: str = ""
    predicate: str = ""
    predicate_surface: str = ""
    wants: str = "none"                # subject|object|definition|related|none
    negation: bool = False
    tense: str = ""                    # ""|past|present|future
    modality: str = ""                 # ""|possible|obligation
    is_question: bool = False
    is_elliptical: bool = False
    command: str = ""
    command_arg: str = ""
    explicit_teach: bool = False
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)

    def has_subject(self) -> bool:
        return bool(self.subject_mention or self.subject_pronoun)

    def has_object(self) -> bool:
        return bool(self.object_mention or self.object_pronoun)

    def as_dict(self) -> dict:
        return {"raw": self.raw, "intent": self.intent, "wh": self.wh,
                "subject": self.subject_mention, "object": self.object_mention,
                "subject_pronoun": self.subject_pronoun,
                "object_pronoun": self.object_pronoun,
                "predicate": self.predicate, "wants": self.wants,
                "negation": self.negation, "tense": self.tense,
                "modality": self.modality, "is_question": self.is_question,
                "is_elliptical": self.is_elliptical, "command": self.command,
                "explicit_teach": self.explicit_teach,
                "confidence": round(self.confidence, 3), "notes": list(self.notes)}

    def describe(self) -> str:
        bits = [self.intent]
        if self.subject_mention:
            bits.append(f"subj={self.subject_mention!r}")
        if self.subject_pronoun:
            bits.append(f"subj-pronoun={self.subject_pronoun!r}")
        if self.predicate:
            bits.append(f"pred={self.predicate}")
        if self.object_mention:
            bits.append(f"obj={self.object_mention!r}")
        if self.object_pronoun:
            bits.append(f"obj-pronoun={self.object_pronoun!r}")
        if self.negation:
            bits.append("NEGATED")
        bits.append(f"conf={self.confidence:.2f}")
        return " ".join(bits)


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------

class LanguageParser:
    """Stateless text -> QueryIR.  No KB access, no side effects."""

    def __init__(self, schema: RelationSchema | None = None):
        self.schema = schema or DEFAULT_SCHEMA

    # -- public ------------------------------------------------------------
    def parse(self, text: str) -> list[QueryIR]:
        if not text or not text.strip():
            return []
        s = squeeze(text)
        cmd = COMMAND_RE.match(s.strip())
        if cmd:
            return [QueryIR(raw=s, norm=normalize(s), intent=INT_COMMAND,
                            command=cmd.group(1).lower(),
                            command_arg=cmd.group(2).strip(), confidence=1.0,
                            notes=["explicit :command"])]
        out = []
        for clause in split_clauses(s):
            ir = self.parse_clause(clause)
            out.append(ir)
        return out or [QueryIR(raw=s, norm=normalize(s), intent=INT_UNKNOWN,
                               confidence=0.0, notes=["no rule matched"])]

    def parse_clause(self, text: str) -> QueryIR:
        raw = squeeze(text)
        norm = normalize(raw)
        ir = QueryIR(raw=raw, norm=norm)
        if not norm:
            ir.notes.append("empty clause")
            return ir

        ir.is_question = raw.rstrip().endswith("?")
        toks = _tokenize(raw)                 # case preserved, keeps "-" copula
        low = [t.lower() for t in toks]       # lookups use lowercase

        # explicit teaching verb: "teach Исаак Ньютон - это физик".
        # `raw` becomes the *proposition* only, so evidence and provenance
        # record what was asserted, not the CLI verb used to assert it.
        if low and low[0] in TEACH_PREFIXES:
            ir.explicit_teach = True
            toks, low = toks[1:], low[1:]
            ir.raw = " ".join(toks)
            ir.notes.append(f"explicit teach verb {toks and '' or ''}"
                            f"stripped from the proposition")
        ir.tense = _detect_tense(low)
        ir.modality = _detect_modality(low)
        ir.negation = _detect_negation(norm, low)
        if ir.negation:
            ir.notes.append("negation detected")

        if any(t in RETRACT_WORDS for t in low):
            ir.intent = INT_RETRACT
            ir.confidence = 0.9
            ir.notes.append("explicit retract verb")
            keep = [t for t, l in zip(toks, low) if l not in RETRACT_WORDS]
            self._segment(ir, keep, [t.lower() for t in keep])
            return ir

        if any(t in GREETING_WORDS for t in low):
            ir.intent = INT_GREETING
            ir.confidence = 0.8
            ir.notes.append("greeting")
            return ir

        d = self._match_definition(norm)
        if d:
            x, kind = d
            ir.intent = INT_ASK_DEFINITION
            ir.wants = "related" if kind == "related" else "definition"
            ir.subject_mention = _clean_mention(x, raw)
            ir.predicate = DEFINITION_PREDICATE
            ir.confidence = 0.95
            ir.notes.append(f"definition template ({kind})")
            return ir

        rel_hits = self.schema.relations_for_surfaces(
            t for t in low if t not in STOPWORDS)
        wh = next((t for t in low if t in WH_WANTS), "")
        if wh:
            ir.wh = wh
            ir.notes.append(f"wh={wh}")

        if wh and ir.is_question:
            ir.intent = INT_ASK_WH
            self._segment_wh(ir, toks, low, wh, rel_hits)
            return ir

        self._segment(ir, toks, low)

        if ir.is_question:
            ir.intent = INT_ASK_YESNO
            ir.wants = "none"
            ir.confidence = 0.8 if (ir.predicate and ir.has_subject()) else 0.5
            ir.notes.append("yes/no question about "
                            + (f"the relation {ir.predicate}" if ir.predicate
                               else "an unresolved predicate"))
            return ir

        if ir.predicate and (ir.has_subject() or ir.has_object()):
            ir.intent = INT_TEACH
            ir.confidence = 0.75
            ir.notes.append("declarative sentence with a recognised predicate")
        elif ir.subject_mention and ir.object_mention:
            ir.intent = INT_TEACH
            ir.predicate = "related_to"
            ir.confidence = 0.4
            ir.notes.append("copula without a recognised predicate -> "
                            "related_to fallback (low confidence)")
        else:
            ir.intent = INT_UNKNOWN
            ir.confidence = 0.2
            ir.notes.append("no subject/predicate/object structure found")
        return ir

    # -- definition templates ---------------------------------------------
    def _match_definition(self, norm: str) -> tuple[str, str] | None:
        q = norm.rstrip("?").strip()
        for pat, kind in DEFINITION_PATTERNS:
            m = re.match(pat, q)
            if m:
                x = m.group("x").strip()
                if x and x not in PRONOUN_SET:
                    return x, kind
        return None

    # -- wh segmentation ---------------------------------------------------
    def _segment_wh(self, ir: QueryIR, toks: list[str], low: list[str],
                    wh: str, rel_hits: list[tuple[str, str]]) -> None:
        wants, _slot = WH_WANTS.get(wh, ("object", "thing"))
        ir.wants = wants
        pred_hit = next(((w, r) for w, r in rel_hits if r != "related_to"),
                        rel_hits[0] if rel_hits else None)
        if pred_hit:
            ir.predicate, ir.predicate_surface = pred_hit[1], pred_hit[0]
        elif wh == "когда":
            ir.predicate, ir.predicate_surface = "happened_at", "когда"
            ir.notes.append("predicate defaulted to happened_at for 'когда'")
        elif wh in ("где", "куда", "откуда"):
            ir.predicate, ir.predicate_surface = "located_in", "где"
            ir.notes.append("predicate defaulted to located_in for 'где'")
        elif wh in ("почему", "зачем"):
            ir.predicate, ir.predicate_surface = "causes", "почему"
            ir.notes.append("predicate defaulted to causes for 'почему'")

        wi = low.index(wh) if wh in low else -1
        vi = -1
        if ir.predicate_surface:
            if ir.predicate_surface in low:
                vi = low.index(ir.predicate_surface)
            else:
                vi = next((i for i, t in enumerate(low)
                           if self.schema.relation_for_surface(t) == ir.predicate), -1)
        if vi == wi:
            # the wh-word IS the predicate surface ("А когда?") -> no verb,
            # the clause is elliptical and needs context to bind a relation.
            vi = -1

        drop = {wi}
        if vi >= 0:
            drop.add(vi)
        # "какой закон сформулировал Ньютон" — the class noun between the wh
        # word and the verb belongs to the *question*, not to the known side.
        if 0 <= wi < vi and vi == wi + 2:
            mid = low[wi + 1]
            if (self.schema.relation_for_surface(mid) is None
                    and mid not in PRONOUN_SET and mid not in COPULAS):
                drop.add(wi + 1)
                ir.notes.append(f"class noun {toks[wi+1]!r} belongs to the "
                                f"wh phrase, dropped from the known side")

        remainder = [toks[i] for i in range(len(toks)) if i not in drop]
        remainder = [t for t in remainder
                     if t.lower() not in STOPWORDS and t.lower() not in COPULAS]
        mention = _clean_mention(" ".join(remainder), ir.raw)

        who = wh in WHO_WORDS
        wh_before_verb = (vi < 0) or (wi >= 0 and wi < vi)
        # A passive relation ("открыт", discovered_by) puts the AGENT in the
        # object slot, so "кем открыт X?" asks for the object and X is the
        # subject.  With the active relation it is the other way round.
        passive = ir.predicate.endswith("_by")
        if who:
            if passive:
                ir.subject_mention = mention
                ir.wants = "object"
                ir.notes.append(f"'{wh}' asks for the agent, which is the "
                                f"OBJECT of the passive relation "
                                f"{ir.predicate}; remainder = its subject")
            else:
                ir.object_mention = mention
                ir.notes.append("'кто' asks for the subject; remainder = object")
        elif wh_before_verb:
            ir.subject_mention = mention
            ir.notes.append("wh precedes the verb -> remainder = subject")
        else:
            ir.object_mention = mention
            ir.notes.append("wh follows the verb -> remainder = object")

        self._extract_pronouns(ir)
        ir.confidence = 0.85 if ir.predicate else 0.5
        ir.is_elliptical = not (ir.has_subject() or ir.has_object())
        if ir.is_elliptical:
            ir.notes.append("elliptical: only the wh-word was present")

    # -- declarative segmentation -----------------------------------------
    def _segment(self, ir: QueryIR, toks: list[str], low: list[str]) -> None:
        rel_hits = self.schema.relations_for_surfaces(
            t for t in low if t not in STOPWORDS)
        pred_hit = next(((w, r) for w, r in rel_hits if r != "related_to"),
                        rel_hits[0] if rel_hits else None)

        if pred_hit is None and ir.negation and any(t in COPULAS for t in low):
            ir.predicate = DEFINITION_PREDICATE
            ir.predicate_surface = next(t for t in low if t in COPULAS)
            ir.notes.append("negated copula -> instance_of with polarity false")
        elif pred_hit:
            ir.predicate, ir.predicate_surface = pred_hit[1], pred_hit[0]

        if ir.predicate_surface and ir.predicate_surface in low:
            vi = low.index(ir.predicate_surface)
            left = [t for t in toks[:vi]]
            right = [t for t in toks[vi + 1:]]
            ir.subject_mention = _clean_mention(" ".join(_strip_aux(left)), ir.raw)
            ir.object_mention = _clean_mention(" ".join(_strip_aux(right)), ir.raw)
            ir.notes.append(f"segmented around predicate "
                            f"{ir.predicate_surface!r}->{ir.predicate}")
            # NOTE: no passive swap.  Passive surfaces ("открыт", "сформулирован")
            # are registered under the *_by relations whose argument order
            # already matches the passive clause (theme - verb - agent).
            self._extract_pronouns(ir)
            return

        ci = next((i for i, t in enumerate(low)
                   if (t in COPULAS or t in DASH_TOKENS) and 0 < i < len(low) - 1),
                  None)
        if ci is not None:
            ir.predicate = ir.predicate or DEFINITION_PREDICATE
            ir.predicate_surface = low[ci]
            ir.subject_mention = _clean_mention(" ".join(_strip_aux(toks[:ci])), ir.raw)
            ir.object_mention = _clean_mention(" ".join(_strip_aux(toks[ci + 1:])), ir.raw)
            self._extract_pronouns(ir)
            ir.notes.append(f"segmented around the copula {low[ci]!r}")
            return


        ir.subject_mention = _clean_mention(" ".join(toks), ir.raw)
        self._extract_pronouns(ir)
        ir.notes.append("no structure: whole clause taken as the subject")

    def _extract_pronouns(self, ir: QueryIR) -> None:
        for side in ("subject", "object"):
            mention = getattr(ir, f"{side}_mention")
            if not mention:
                continue
            toks = tokens_keep_case(mention)
            low = [t.lower() for t in toks]
            pron = next((l for l in low if l in PRONOUN_SET), "")
            if pron and len(toks) <= 2:
                setattr(ir, f"{side}_pronoun", pron)
                setattr(ir, f"{side}_mention",
                        _clean_mention(" ".join(t for t, l in zip(toks, low)
                                                if l not in PRONOUN_SET), ir.raw))
                ir.notes.append(f"{side} pronoun {pron!r} -> needs context")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

# digits included: a year must survive tokenisation to be storable as a
# literal value ("родился в 1643 году")
_WORD_CHUNK = re.compile(r"[^\W_]+(?:-[^\W_]+)*", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Case-preserving tokens PLUS standalone dash tokens, in order.

    ``tokens_keep_case`` drops "-", which would hide the Russian copula from
    the segmenter; here the dash is kept as its own token so that
    "Ньютон - физик" segments the same way as "Ньютон это физик".
    """
    t = squeeze(text)
    out: list[str] = []
    for chunk in t.split():
        if chunk in DASH_TOKENS:
            out.append(chunk)
            continue
        out.extend(m.group(0).replace("ё", "е") for m in _WORD_CHUNK.finditer(chunk))
    return out


def _detect_tense(low: Sequence[str]) -> str:
    if any(t in TENSE_PAST for t in low):
        return "past"
    if any(t in TENSE_FUTURE for t in low):
        return "future"
    return "present" if low else ""


def _detect_modality(low: Sequence[str]) -> str:
    if any(t in ("может", "мог", "могла", "вероятно", "возможно", "кажется",
                 "похоже") for t in low):
        return "possible"
    if any(t in ("должен", "должна", "обязан") for t in low):
        return "obligation"
    return ""


def _detect_negation(norm: str, low: Sequence[str]) -> bool:
    if any(p in norm for p in NEGATION_PHRASES):
        return True
    return any(t in NEGATION_TOKENS for t in low)


def _strip_aux(toks: Sequence[str]) -> list[str]:
    return [t for t in toks
            if t.lower() not in STOPWORDS and t.lower() not in COPULAS
            and t.lower() not in NEG_STRIP and t not in DASH_TOKENS]


def _clean_mention(text: str, original_case_source: str = "") -> str:
    """Trim punctuation/stopwords from a mention while keeping the case the
    user actually typed."""
    t = re.sub(r"^[,.;:!?-]+|[,.;:!?-]+$", "", str(text or "").strip())
    toks = tokens_keep_case(t)
    out = [w for w in toks
           if w.lower() not in STOPWORDS and w.lower() not in NEG_STRIP]
    while out and out[0].lower() in LEADING_PARTICLES:
        out = out[1:]
    if not out:
        return ""
    # restore the exact casing from the source clause when possible
    if original_case_source:
        src = {w.lower(): w for w in tokens_keep_case(original_case_source)}
        out = [src.get(w.lower(), w) for w in out]
    return " ".join(out)
