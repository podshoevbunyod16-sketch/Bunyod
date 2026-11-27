"""DialogueManager (v0.13) — multi-turn, multi-question, pollution-guarded.

Pipeline per message:
  1. mode = normal | hypothetical | quote | teach   (pollution guard FIRST)
  2. sub_questions = split_questions(message)         (order preserved)
  3. per sub-question: analyze → entity match (KB, stem-based) → pronoun/ellipsis
     resolution against context → route by intent:
       greeting → canned;  ask_* / yes_no → KB statement lookup with
       statuses TRUE/FALSE/UNKNOWN/CONFLICT (+ hypothesis note);
       compare → partial (both is_a sides);  list → KB scan by category;
       unknown → clarification.
  4. ambiguous pronouns → clarifying question (candidates listed, never
     coin-flip); missing entity → "о ком/о чём идёт речь?".
  5. context updated ONCE per message (user turn + system turn, entities MRU,
     topic, open questions).

Nothing in hypothetical/quote mode is ever persisted to the KB. Only the
explicit `teach` path (e.g. "запомни: Франция — это страна") writes facts,
as user_assertion with full provenance.
"""
from __future__ import annotations

import re
import time
from typing import List, Optional

from rasm_cim_v11.language import analyze, lemma, normalize
from rasm_cim_v12.knowledge import (KnowledgeStore, Provenance, teach)
from rasm_cim_v12.knowledge.teach import canon_predicate

from .context import ContextStore, utcnow
from .pronouns import find_pronouns, guess_gender, resolve
from .splitter import split_questions

_KW_PRED = [
    ("столиц", "capital_of"),
    ("населени", "population_of"), ("жител", "population_of"),
    ("рост", "height_of"),
    ("написал", "написал"), ("автор", "написал"),
    ("лета", "может_летать"),
    ("спутник", "имеет_спутник"),
    ("кипит", "кипит_при"),
    ("существует", "существует"),
]
_DEF_TRIGGERS = ("что такое", "кто такой", "кто такая", "кто такие",
                 "что значит", "что означает")
_HYPOTHETICAL = re.compile(
    r"\b(допустим|например|представь(те)?|предположим|пусть|гипотетическ\w*|"
    r"условно( говоря)?|вообрази(те)?|если бы)\b", re.IGNORECASE)
_TEACH_LEAD = re.compile(
    r"^\s*(запомни|запишите|запиши|учти|учтите|сохрани факт)\s*[:,-]?\s*", re.IGNORECASE)
_GREETING_REPLY = "Привет! Чем помочь?"
_MAX_LIST = 10


def _stems(text: str) -> set:
    return {lemma(w) for w in re.findall(r"[А-Яа-яЁёA-Za-z0-9\-]+",
                                         normalize(text).text.lower())
            if w.strip()}


class DialogueManager:
    def __init__(self, kb: KnowledgeStore, context: ContextStore):
        self.kb = kb
        self.ctx = context

    # ------------------------------------------------------------ modes
    def _mode(self, message: str) -> str:
        if _TEACH_LEAD.match(message):
            return "teach"
        if _HYPOTHETICAL.search(message):
            return "hypothetical"
        s = message.strip()
        if (s.startswith("«") and s.endswith("»")) or \
           (s.startswith('"') and s.endswith('"')) or \
           (s.startswith("“") and s.endswith("”")):
            return "quote"
        return "normal"

    # ------------------------------------------------------- kb matching
    def _find_entities(self, text: str) -> List[dict]:
        """Stem-subset match of KB entity names/aliases against the question."""
        q = _stems(text)
        found = []
        for e in self.kb.entities.values():
            if e.status != "active":
                continue
            names = [e.name] + list(e.aliases)
            best = None
            for n in names:
                ns = _stems(n)
                if ns and ns <= q:
                    if best is None or len(n) > len(best):
                        best = n
            if best is not None:
                found.append({"id": e.id, "name": e.name, "type": e.type,
                              "gender": guess_gender(e.name), "matched_on": best})
        # longest names first, dedupe by id, cap 6
        found.sort(key=lambda d: -len(d["name"]))
        seen, out = set(), []
        for d in found:
            if d["id"] not in seen:
                out.append(d)
                seen.add(d["id"])
        return out[:6]

    def _predicate_for(self, normalized: str) -> Optional[str]:
        for kw, pred in _KW_PRED:
            if kw in normalized:
                return canon_predicate(pred)
        if any(t in normalized for t in _DEF_TRIGGERS):
            return "is_a"
        return None

    def _fmt_objs(self, stmts) -> str:
        names = []
        for s in stmts:
            if s.object_entity:
                e = self.kb.get_entity(s.object_entity)
                names.append(e.name if e else s.object_entity)
            else:
                names.append(str(s.object_literal))
        return ", ".join(sorted(set(names)))

    def _evidence(self, stmts) -> List[dict]:
        return [{"statement_id": s.id, "source": (s.provenance or {}).get("source"),
                 "doc_id": (s.provenance or {}).get("doc_id"),
                 "method": (s.provenance or {}).get("method"),
                 "reliability": (s.provenance or {}).get("reliability")}
                for s in stmts]

    # ------------------------------------------------------- teach flow
    _TEACH_PATTERNS = [
        (re.compile(r"^(?P<a>[А-Яа-яЁёA-Za-z][\w\s\-]{0,60}?)\s*[—-]\s*(?:это\s+)?(?P<b>[А-Яа-яЁёA-Za-z][\w\s\-]{0,60}?)[.!]?$"),
         "is_a"),
        (re.compile(r"^столица\s+(?P<a>[А-Яа-яЁёA-Za-z][\w\s\-]{0,60}?)\s*[—-]\s*(?P<b>[А-Яа-яЁёA-Za-z][\w\s\-]{0,60}?)[.!]?$", re.IGNORECASE),
         "capital_of"),
        (re.compile(r"^(?P<a>[А-Яа-яЁёA-Za-z][\w\s]{0,60}?)\s+написал[аи]?\s+(?P<b>[А-Яа-яЁёA-Za-z][\w\s\-]{0,80}?)[.!]?$", re.IGNORECASE),
         "написал"),
    ]

    def _type_of(self, name: str, default: str) -> str:
        status, e, _ = self.kb.find_by_name(name)
        return e.type if status == "found" else default

    def _teach(self, message: str) -> dict:
        body = _TEACH_LEAD.sub("", message).strip()
        for rx, pred in self._TEACH_PATTERNS:
            m = rx.match(body)
            if not m:
                continue
            a, b = m.group("a").strip(), m.group("b").strip()
            prov = Provenance(source="user_teach", doc_id=f"dialog:{utcnow()}",
                              fragment=message, method="manual",
                              parser_version="teach-0.13.0",
                              added_at=utcnow())
            if pred == "capital_of":
                res = teach(self.kb, a, "capital_of", b, subject_type="place",
                            object_type="place", provenance=prov)
            elif pred == "is_a":
                res = teach(self.kb, a, pred, b,
                            subject_type=self._type_of(a, "thing"),
                            object_type=self._type_of(b, "concept"),
                            provenance=prov)
            else:
                res = teach(self.kb, a, pred, b,
                            subject_type=self._type_of(a, "person"),
                            object_type=self._type_of(b, "thing"),
                            provenance=prov)
            return {"intent": "teach", "status": "TAUGHT",
                    "answer": f"Запомнил: {a} — {res['predicate']} — {b}.",
                    "evidence": [{"statement_id": res["statement_id"],
                                  "source": prov.source, "doc_id": prov.doc_id,
                                  "method": "manual", "reliability": "medium"}],
                    "mode": "teach"}
        return {"intent": "teach", "status": "TEACH_HELP",
                "answer": "Не понял факт. Формат: «запомни: А — это Б», «запомни: столица X — Y» или «запомни: X написал Y».",
                "evidence": [], "mode": "teach"}

    # ------------------------------------------------------- answers
    def _answer_fact(self, sub: str, analysis, entities, mode, intent: Optional[str] = None) -> dict:
        out_intent = intent or analysis.intent.intent
        note = " (гипотетика — не сохраняю как факт)" if mode == "hypothetical" else ""
        if mode == "quote":
            note = " (это цитата — отвечаю только учётом данных, ничего не записываю)"
        if not entities:
            return {"text": sub, "intent": out_intent,
                    "status": "CLARIFY",
                    "answer": "О ком или о чём идёт речь?" + note,
                    "evidence": [], "entities": []}
        predicate = self._predicate_for(analysis.normalized)
        if predicate is None:
            # unique-statement fallback: exactly one active statement among entities
            act = []
            for e in entities:
                act.extend(s for s in self.kb.query_statements(subject=e["id"])
                           if s.status == "active")
            if len(act) == 1:
                s = act[0]
                subj = self.kb.get_entity(s.subject)
                return {"text": sub, "intent": out_intent,
                        "status": "TRUE",
                        "answer": f"{subj.name}: {s.predicate} — {self._fmt_objs([s])}." + note,
                        "evidence": self._evidence([s]),
                        "entities": [e["name"] for e in entities]}
            return {"text": sub, "intent": out_intent,
                    "status": "CLARIFY",
                    "answer": f"Что именно узнать про {', '.join(e['name'] for e in entities)}?" + note,
                    "evidence": [], "entities": [e["name"] for e in entities]}

        pieces, ev_all, names = [], [], []
        statuses = []
        for subject in entities:
            subj_id = subject["id"]
            names.append(subject["name"])
            stmts = self.kb.query_statements(subject=subj_id, predicate=predicate)
            if not stmts:
                # nothing as subject — try reverse: X is object of predicate here
                rev = self.kb.query_statements(predicate=predicate, object=subj_id)
                rev = [s for s in rev if s.status == "active"]
                if rev:
                    ev_all.extend(self._evidence(rev))
                    names_rev = sorted({self.kb.get_entity(s.subject).name for s in rev})
                    pieces.append(f"{subject['name']}: {', '.join(names_rev)}")
                    statuses.append("TRUE")
                    continue
            confirmed = [s for s in stmts if s.status == "active"
                         and s.kind in ("user_assertion", "extracted_fact")]
            hyps = [s for s in stmts if s.status == "active" and s.kind == "hypothesis"]
            if not confirmed:
                ev_all.extend(self._evidence(hyps))
                if hyps:
                    pieces.append(f"{subject['name']}: только неподтверждённая гипотеза ({hyps[0].id})")
                    statuses.append("PARTIAL")
                else:
                    pieces.append(f"{subject['name']}: нет данных ({predicate})")
                    statuses.append("UNKNOWN")
                continue
            pols = {s.polarity for s in confirmed}
            if pols == {"assert", "deny"}:
                ev_all.extend(self._evidence(confirmed))
                pieces.append(f"{subject['name']}: данные противоречат друг другу ({', '.join(s.id for s in confirmed)})")
                statuses.append("CONFLICT")
                continue
            if predicate in ("capital_of", "population_of", "height_of"):
                objs = {s.object_repr for s in confirmed if s.polarity == "assert"}
                if len(objs) > 1:
                    ev_all.extend(self._evidence(confirmed))
                    pieces.append(f"{subject['name']}: разные значения — {self._fmt_objs(confirmed)}")
                    statuses.append("CONFLICT")
                    continue
            asserts = [s for s in confirmed if s.polarity == "assert"]
            ev_all.extend(self._evidence(asserts))
            if asserts:
                pieces.append(f"{subject['name']}: {self._fmt_objs(asserts)}")
                statuses.append("TRUE")
            else:
                pieces.append(f"{subject['name']}: {predicate} — нет")
                statuses.append("FALSE")

        uniq = set(statuses)
        if "CONFLICT" in uniq:
            status = "CONFLICT"
        elif len(uniq) == 1:
            status = next(iter(uniq))
        else:
            status = "PARTIAL"

        if out_intent == "ask_definition":
            parts = []
            for n, p in zip(names, pieces):
                prefix = f"{n}: "
                payload = p[len(prefix):] if p.startswith(prefix) else p
                if payload.startswith(("нет данных", "только", "данные", "разные")):
                    parts.append(f"{n}: {payload}")
                else:
                    parts.append(f"«{n}» — это {payload}")
            answer = "; ".join(parts) + "." + note
        else:
            answer = "; ".join(pieces) + "." + note
        return {"text": sub, "intent": out_intent,
                "status": status, "answer": answer,
                "evidence": ev_all, "entities": names}

    def _answer_compare(self, sub, analysis, entities, mode):
        if len(entities) < 2:
            return {"text": sub, "intent": "compare", "status": "UNSUPPORTED",
                    "answer": "Для сравнения нужны две сущности, которые я знаю. "
                              "Полный сравнительный контур — v0.14.",
                    "evidence": [], "entities": [e["name"] for e in entities]}
        a, b = entities[0], entities[1]
        ev, parts = [], []
        for e in (a, b):
            stmts = [s for s in self.kb.query_statements(subject=e["id"], predicate="is_a")
                     if s.status == "active"]
            if stmts:
                ev.extend(self._evidence(stmts))
                parts.append(f"{e['name']} — это {self._fmt_objs(stmts)}")
        if len(parts) < 2:
            return {"text": sub, "intent": "compare", "status": "UNKNOWN",
                    "answer": "Недостаточно данных для сравнения.",
                    "evidence": ev, "entities": [a["name"], b["name"]]}
        return {"text": sub, "intent": "compare", "status": "TRUE",
                "answer": "; ".join(parts) + ".",
                "evidence": ev, "entities": [a["name"], b["name"]]}

    def _answer_list(self, sub, analysis, entities, mode):
        norm = analysis.normalized
        cat_stems = {lemma(w) for w in re.findall(r"[А-Яа-яЁёA-Za-z]+", norm)}
        hits = []
        for e in self.kb.entities.values():
            if e.status != "active":
                continue
            es = _stems(e.name)
            if es and (es <= cat_stems):
                hits.append(e)
        if not hits:
            return {"text": sub, "intent": "list", "status": "UNKNOWN",
                    "answer": "Не знаю такой категории в памяти.",
                    "evidence": [], "entities": []}
        target = max(hits, key=lambda e: len(e.name))
        stmts = [s for s in self.kb.query_statements(predicate="is_a",
                                                     object=target.id)
                 if s.status == "active"]
        subj_names = sorted({self.kb.get_entity(s.subject).name for s in stmts
                             if self.kb.get_entity(s.subject)})
        if not subj_names:
            return {"text": sub, "intent": "list", "status": "UNKNOWN",
                    "answer": f"Не знаю примеров для «{target.name}».",
                    "evidence": [], "entities": [target.name]}
        pred = ", ".join(subj_names[:_MAX_LIST])
        extra = "" if len(subj_names) <= _MAX_LIST else f" и ещё {len(subj_names) - _MAX_LIST}"
        return {"text": sub, "intent": "list", "status": "TRUE",
                "answer": f"{pred}{extra}.",
                "evidence": self._evidence(stmts),
                "entities": subj_names[:_MAX_LIST]}

    def _answer_yes_no(self, sub, analysis, entities, mode):
        # yes/no reuses fact lookup; the machine verdict lives in `status`
        res = self._answer_fact(sub, analysis, entities, mode, intent="yes_no")
        res["intent"] = "yes_no"
        return res

    # ----------------------------------------------------------- main
    def process(self, message: str) -> dict:
        t0 = time.perf_counter()
        mode = self._mode(message)
        if mode == "teach":
            out = self._teach(message)
            result = {"message": message, "mode": mode,
                      "results": [{**out, "text": message}]}
            self._record_turn(message, result, entities_all=[], mode=mode)
            result["latency_us"] = int((time.perf_counter() - t0) * 1e6)
            return result

        subs = split_questions(message)
        if not subs:
            subs = [message]
        results = []
        all_entities: List[dict] = []
        for sub in subs:
            analysis = analyze(sub)
            intent = analysis.intent.intent
            entities = self._find_entities(sub)
            # pronoun + ellipsis resolution against context
            # plural pronouns: prefer entities of the LAST user turn, not global MRU
            last_user_entities = self._last_user_entities()
            if any(p["number"] == "p" for p in find_pronouns(sub)) and len(last_user_entities) >= 2:
                recent = last_user_entities
            else:
                recent = self.ctx.state["recent_entities"]
            prons = find_pronouns(sub)
            resolutions = resolve(prons, recent)
            # context-dependent follow-ups without question words are still questions
            if intent == "unknown" and any(r["status"] == "resolved" for r in resolutions):
                intent = "ask_fact"
            clarifications = []
            for r in resolutions:
                if r["status"] == "resolved":
                    for name in r["chosen"]:
                        if not any(e["name"] == name for e in entities):
                            e = self._find_entities(name)
                            entities = e + entities
                elif r["status"] == "ambiguous":
                    clarifications.append(
                        f"Разные «{r['pronoun']}»: {', '.join(r['candidates'])} — о ком речь?")
                elif r["status"] == "unknown" and prons:
                    clarifications.append(
                        f"Не понял, к кому/чему относится «{r['pronoun']}». О ком идёт речь?")
            if clarifications:
                q = " ".join(clarifications)
                self.ctx.add_open_question(q, missing="referent",
                                           slot="pronoun",
                                           candidates=resolutions[0]["candidates"] if resolutions else [])
                results.append({"text": sub, "intent": intent, "status": "CLARIFY",
                                "answer": q, "evidence": [], "entities": []})
                continue
            if intent == "greeting":
                results.append({"text": sub, "intent": intent, "status": "GREETING",
                                "answer": _GREETING_REPLY, "evidence": [], "entities": []})
            elif intent in ("ask_fact", "ask_definition"):
                results.append(self._answer_fact(sub, analysis, entities, mode, intent=intent))
            elif intent == "yes_no":
                results.append(self._answer_yes_no(sub, analysis, entities, mode))
            elif intent == "compare":
                results.append(self._answer_compare(sub, analysis, entities, mode))
            elif intent == "list":
                results.append(self._answer_list(sub, analysis, entities, mode))
            else:
                results.append({"text": sub, "intent": intent, "status": "UNKNOWN",
                                "answer": "Не поддерживается (честный unknown; см. v0.14 для приёма QA).",
                                "evidence": [], "entities": []})
            for e in entities:
                if all(x["id"] != e["id"] for x in all_entities):
                    all_entities.append(e)

        result = {"message": message, "mode": mode, "results": results}
        self._record_turn(message, result, entities_all=all_entities, mode=mode)
        result["latency_us"] = int((time.perf_counter() - t0) * 1e6)
        return result

    def _last_user_entities(self) -> List[dict]:
        for t in reversed(self.ctx.state["turns"]):
            if t["role"] != "user":
                continue
            out, seen = [], set()
            for eid in t.get("entity_ids", []):
                e = self.kb.get_entity(eid)
                if e is not None and e.id not in seen:
                    out.append({"id": e.id, "name": e.name, "type": e.type,
                                "gender": guess_gender(e.name), "matched_on": e.name})
                    seen.add(e.id)
            return out
        return []

    # ----------------------------------------------------------- context
    def _record_turn(self, message: str, result: dict,
                     entities_all: List[dict], mode: str) -> None:
        stmt_ids = [ev["statement_id"] for r in result["results"]
                    for ev in r.get("evidence", [])]
        ent_ids = [e["id"] for e in entities_all]
        self.ctx.suspend()
        self.ctx.add_turn("user", message,
                          intent=result["results"][0]["intent"] if result["results"] else None,
                          entities=[e["name"] for e in entities_all],
                          entity_ids=ent_ids,
                          statement_ids=stmt_ids, mode=mode)
        answer_text = " ".join(r["answer"] for r in result["results"])
        self.ctx.add_turn("system", answer_text,
                          entities=[e["name"] for e in entities_all],
                          statement_ids=stmt_ids, mode=mode)
        if entities_all:
            self.ctx.record_entities([{"name": e["name"], "gender": e["gender"],
                                       "kb_id": e["id"]} for e in entities_all])
            top = entities_all[0]
            self.ctx.set_topic(top["name"])
        self.ctx.flush()
