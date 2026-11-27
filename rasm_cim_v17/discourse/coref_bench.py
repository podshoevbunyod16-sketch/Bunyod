"""Coreference benchmark: cases, splits, sealing, and scoring with Wilson CIs.

Why a generator and not a hand-written list: 300+ dialogues must exist *before*
the system is measured, must be reproducible, and must not be edited phrase by
phrase when an answer disappoints.  Every case is therefore built from lexical
banks (names with gender and accusative form, classes, cities, books) by a
seeded generator that is committed together with its output.  The expected
referent follows from the *construction* of the dialogue — who was named last,
which gender the pronoun requires — never from what the system answered.

Rubric, stated once so every number can be recomputed:

    EXPECT_REFERENT  the discourse determines exactly one referent
                     correct = bound to it; wrong = bound to another entity;
                     refusal = counted separately (honest, but not correct)
    EXPECT_REFUSE    the discourse cannot determine a referent (no antecedent,
                     or no antecedent of the required gender)
                     correct = no binding; ungrounded = bound to anything
    EXPECT_RECENCY   two candidates of the same gender, one clearly more recent
                     correct = the recent one; wrong = the older one;
                     refusal = acceptable but not counted as correct

The hidden split is sealed: its sha256 is recorded in ``manifest.json`` before
any measurement, and the loader refuses a hidden file whose hash changed.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

BENCH_SCHEMA = "rasm-cim-coref-bench/1"
MANIFEST_SCHEMA = "rasm-cim-coref-manifest/1"

EXPECT_REFERENT = "REFERENT"
EXPECT_REFUSE = "REFUSE"
EXPECT_RECENCY = "RECENCY"

CATEGORIES = (
    "pronoun_simple", "repeat_mention", "topic_shift", "correction",
    "gender_select", "no_antecedent", "gender_mismatch", "ambiguous_pair",
)
#: categories of the second bench (harder constructions, same rubric) — declared
#: here because BenchCase validates against both sets at construction time
CATEGORIES_V2 = (
    "distractor_class", "correction_then_shift", "negation_correction",
    "plural_ambiguous", "long_distance",
)

#: categories whose correct answer is a refusal
REFUSE_CATEGORIES = {"no_antecedent", "gender_mismatch", "plural_ambiguous"}


class CorefBenchError(ValueError):
    """The bench file, its manifest, or the split is invalid."""


# --------------------------------------------------------------------------
# lexical banks — data, not answers
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Person:
    name: str
    accusative: str
    gender: str          # "m" | "f"
    klass: str           # profession used in facts and questions


MALE = (
    Person("Иван", "Ивана", "m", "физик"),
    Person("Пётр", "Петра", "m", "химик"),
    Person("Пушкин", "Пушкина", "m", "поэт"),
    Person("Толстой", "Толстого", "m", "писатель"),
    Person("Гагарин", "Гагарина", "m", "космонавт"),
    Person("Менделеев", "Менделеева", "m", "химик"),
    Person("Дарвин", "Дарвина", "m", "биолог"),
    Person("Ломоносов", "Ломоносова", "m", "учёный"),
    Person("Кеплер", "Кеплера", "m", "астроном"),
    Person("Галилей", "Галилея", "m", "астроном"),
)

FEMALE = (
    Person("Анна", "Анну", "f", "химик"),
    Person("Мария", "Марию", "f", "математик"),
    Person("Елена", "Елену", "f", "биолог"),
    Person("Ольга", "Ольгу", "f", "врач"),
    Person("Софья", "Софью", "f", "математик"),
    Person("Ахматова", "Ахматову", "f", "поэт"),
    Person("Цветаева", "Цветаеву", "f", "поэт"),
    Person("Терешкова", "Терешкову", "f", "космонавт"),
    Person("Екатерина", "Екатерину", "f", "историк"),
    Person("Ирина", "Ирину", "f", "физик"),
)

CITIES = ("Москва", "Париж", "Казань", "Самара", "Тверь", "Омск", "Псков",
          "Тула", "Пермь", "Орёл")

#: question frames per person, chosen by gender so the agreement is real
_WHO = {"m": "Кто такой {name}?", "f": "Кто такая {name}?"}
_PRONOUN = {"m": "он", "f": "она"}
_BORN = {"m": "он родился", "f": "она родилась"}


def _class_question(person: Person, pronoun: str) -> str:
    return f"А {pronoun} {person.klass}?"


def _who(person: Person) -> str:
    return _WHO[person.gender].format(name=person.name)


def _fact(person: Person, city: str) -> str:
    return f"{person.name} — это {person.klass}"


# --------------------------------------------------------------------------
# cases
# --------------------------------------------------------------------------
@dataclass
class BenchCase:
    id: str
    split: str
    category: str
    facts: list[str]
    turns: list[str]
    target: int
    expected: str            # EXPECT_*
    entity: str              # intended referent ("" for REFUSE)
    pronoun: str
    note: str = ""

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES + CATEGORIES_V2:
            raise CorefBenchError(f"{self.id}: неизвестная категория "
                                  f"{self.category!r}")
        if not self.turns:
            raise CorefBenchError(f"{self.id}: нет реплик")
        if not (0 <= self.target < len(self.turns)):
            raise CorefBenchError(f"{self.id}: target вне диапазона реплик")
        if self.expected not in (EXPECT_REFERENT, EXPECT_REFUSE, EXPECT_RECENCY):
            raise CorefBenchError(f"{self.id}: неизвестное ожидание "
                                  f"{self.expected!r}")
        if self.expected == EXPECT_REFUSE and self.entity:
            raise CorefBenchError(f"{self.id}: отказ не может иметь референта")
        if self.expected != EXPECT_REFUSE and not self.entity:
            raise CorefBenchError(f"{self.id}: ожидание без референта")

    def as_dict(self) -> dict:
        return {"id": self.id, "split": self.split, "category": self.category,
                "facts": list(self.facts), "turns": list(self.turns),
                "target": self.target, "expected": self.expected,
                "entity": self.entity, "pronoun": self.pronoun,
                "note": self.note}


# --------------------------------------------------------------------------
# generator
# --------------------------------------------------------------------------
def _pronoun_simple(rng, split, idx) -> BenchCase:
    person = rng.choice(MALE + FEMALE)
    pronoun = _PRONOUN[person.gender]
    return BenchCase(
        id=f"{split}-p{idx:03d}", split=split, category="pronoun_simple",
        facts=[_fact(person, "")],
        turns=[_who(person), _class_question(person, pronoun)],
        target=1, expected=EXPECT_REFERENT, entity=person.name,
        pronoun=pronoun, note="единственный кандидат, род совпадает")


def _repeat_mention(rng, split, idx) -> BenchCase:
    person = rng.choice(MALE + FEMALE)
    pronoun = _PRONOUN[person.gender]
    return BenchCase(
        id=f"{split}-r{idx:03d}", split=split, category="repeat_mention",
        facts=[_fact(person, "")],
        turns=[_who(person), _class_question(person, pronoun),
               _who(person), f"А {pronoun} {person.klass}?"],
        target=3, expected=EXPECT_REFERENT, entity=person.name,
        pronoun=pronoun, note="имя, местоимение, имя снова — затем местоимение")


def _topic_shift(rng, split, idx) -> BenchCase:
    a, b = rng.sample(MALE + FEMALE, 2)
    pronoun = _PRONOUN[b.gender]
    return BenchCase(
        id=f"{split}-t{idx:03d}", split=split, category="topic_shift",
        facts=[_fact(a, ""), _fact(b, "")],
        turns=[_who(a), _who(b), _class_question(b, pronoun)],
        target=2, expected=EXPECT_REFERENT, entity=b.name,
        pronoun=pronoun, note="местоимение относится к последней теме, не к первой")


def _correction(rng, split, idx) -> BenchCase:
    a, b = rng.sample(MALE + FEMALE, 2)
    pronoun = _PRONOUN[b.gender]
    return BenchCase(
        id=f"{split}-c{idx:03d}", split=split, category="correction",
        facts=[_fact(a, ""), _fact(b, "")],
        turns=[_who(a), f"Нет, я про {b.accusative}", _class_question(b, pronoun)],
        target=2, expected=EXPECT_REFERENT, entity=b.name,
        pronoun=pronoun, note="пользователь исправил тему: местоимение — про него")


def _gender_select(rng, split, idx) -> BenchCase:
    male, female = rng.choice(MALE), rng.choice(FEMALE)
    want_male = rng.random() < 0.5
    target_person = male if want_male else female
    other = female if want_male else male
    pronoun = _PRONOUN[target_person.gender]
    other_pronoun = _PRONOUN[other.gender]
    return BenchCase(
        id=f"{split}-g{idx:03d}", split=split, category="gender_select",
        facts=[_fact(male, ""), _fact(female, "")],
        turns=[_who(male), _who(female),
               _class_question(target_person, pronoun)],
        target=2, expected=EXPECT_REFERENT, entity=target_person.name,
        pronoun=pronoun,
        note=f"оба в фокусе; «{pronoun}» должно взять "
             f"{'мужчину' if want_male else 'женщину'}, хотя «{other_pronoun}» "
             f"звучало в предыдущей реплике")


def _no_antecedent(rng, split, idx) -> BenchCase:
    """Conversation starts with a pronoun; the KB knows two people of that gender.

    Two, not one: with a single candidate in the whole base, binding to it would
    be a defensible guess, and the rubric would be arguable.  With two, nothing
    in the conversation selects either one.
    """
    male = rng.random() < 0.5
    pool = MALE if male else FEMALE
    a, b = rng.sample(pool, 2)
    pronoun = _PRONOUN[a.gender]
    return BenchCase(
        id=f"{split}-n{idx:03d}", split=split, category="no_antecedent",
        facts=[_fact(a, ""), _fact(b, "")],
        turns=[_class_question(a, pronoun)],
        target=0, expected=EXPECT_REFUSE, entity="", pronoun=pronoun,
        note="диалог начинается с местоимения: в базе два кандидата того же "
             "рода, но ни один не назван в разговоре")


def _gender_mismatch(rng, split, idx) -> BenchCase:
    male = rng.choice(MALE)
    female = rng.choice(FEMALE)
    # антецедент мужского рода, местоимение женского — связки быть не может
    return BenchCase(
        id=f"{split}-m{idx:03d}", split=split, category="gender_mismatch",
        facts=[_fact(male, "")],
        turns=[_who(male), _class_question(female, "она")],
        target=1, expected=EXPECT_REFUSE, entity="", pronoun="она",
        note="в базе и в разговоре только мужчина: «она» связывать не с чем")


def _ambiguous_pair(rng, split, idx) -> BenchCase:
    pair = rng.choice(MALE) if rng.random() < 0.5 else None
    if pair is None:
        a, b = rng.sample(FEMALE, 2)
    else:
        a, b = rng.sample(MALE, 2)
    pronoun = _PRONOUN[b.gender]
    return BenchCase(
        id=f"{split}-a{idx:03d}", split=split, category="ambiguous_pair",
        facts=[_fact(a, ""), _fact(b, "")],
        turns=[_who(a), _who(b), _class_question(b, pronoun)],
        target=2, expected=EXPECT_RECENCY, entity=b.name,
        pronoun=pronoun,
        note="два кандидата одного рода; засчитывается последний, отказ "
             "допустим, но не засчитывается как верная связка")


_BUILDERS = {
    "pronoun_simple": _pronoun_simple,
    "repeat_mention": _repeat_mention,
    "topic_shift": _topic_shift,
    "correction": _correction,
    "gender_select": _gender_select,
    "no_antecedent": _no_antecedent,
    "gender_mismatch": _gender_mismatch,
    "ambiguous_pair": _ambiguous_pair,
}


def build_cases(split: str, count: int, seed: int) -> list[BenchCase]:
    """``count`` cases for one split, distributed evenly across categories."""
    rng = random.Random(f"{split}:{seed}")
    order = list(CATEGORIES)
    rng.shuffle(order)
    cases: list[BenchCase] = []
    for i in range(count):
        category = order[i % len(order)]
        cases.append(_BUILDERS[category](rng, split, i))
    return cases


# --------------------------------------------------------------------------
# v2 — harder constructions (separate sealed files, same rubric)
# --------------------------------------------------------------------------
def _distractor_class(rng, split, idx) -> BenchCase:
    person = rng.choice(MALE + FEMALE)
    other = rng.choice([p for p in MALE + FEMALE if p.name != person.name])
    pronoun = _PRONOUN[person.gender]
    return BenchCase(
        id=f"{split}-dc{idx:03d}", split=split, category="distractor_class",
        facts=[_fact(person, ""), _fact(other, "")],
        turns=[_who(person), f"Что такое {other.klass}?",
               _class_question(person, pronoun)],
        target=2, expected=EXPECT_REFERENT, entity=person.name,
        pronoun=pronoun,
        note="между антецедентом и местоимением назван класс: «он» должен "
             "остаться человеком, а не стать классом")


def _correction_then_shift(rng, split, idx) -> BenchCase:
    a, b = rng.sample(MALE + FEMALE, 2)
    pronoun = _PRONOUN[a.gender]
    return BenchCase(
        id=f"{split}-cs{idx:03d}", split=split, category="correction_then_shift",
        facts=[_fact(a, ""), _fact(b, "")],
        turns=[_who(b), f"Нет, я про {a.accusative}", _who(a),
               f"А {pronoun} {a.klass}?"],
        target=3, expected=EXPECT_REFERENT, entity=a.name,
        pronoun=pronoun,
        note="после исправления назван новый предмет — местоимение идёт за "
             "последним упоминанием")


def _negation_correction(rng, split, idx) -> BenchCase:
    a, b = rng.sample(MALE + FEMALE, 2)
    pronoun = _PRONOUN[b.gender]
    return BenchCase(
        id=f"{split}-nc{idx:03d}", split=split, category="negation_correction",
        facts=[_fact(a, ""), _fact(b, "")],
        turns=[_who(a), f"Нет, не {a.name}, а {b.name}",
               f"А {pronoun} {b.klass}?"],
        target=2, expected=EXPECT_REFERENT, entity=b.name,
        pronoun=pronoun,
        note="исправление через отрицание: «не A, а B»")


def _plural_ambiguous(rng, split, idx) -> BenchCase:
    male = rng.random() < 0.5
    pool = MALE if male else FEMALE
    a, b = rng.sample(pool, 2)
    return BenchCase(
        id=f"{split}-pl{idx:03d}", split=split, category="plural_ambiguous",
        facts=[_fact(a, ""), _fact(b, "")],
        turns=[_who(a), _who(b), f"А они {a.klass}?"],
        target=2, expected=EXPECT_REFUSE, entity="", pronoun="они",
        note="«они» после двух названных людей не указывает на одного из них")


def _long_distance(rng, split, idx) -> BenchCase:
    person = rng.choice(MALE + FEMALE)
    other = rng.choice([p for p in MALE + FEMALE if p.name != person.name])
    pronoun = _PRONOUN[person.gender]
    return BenchCase(
        id=f"{split}-ld{idx:03d}", split=split, category="long_distance",
        facts=[_fact(person, ""), _fact(other, "")],
        turns=[_who(person), f"Что такое {other.klass}?", "Расскажи ещё",
               _class_question(person, pronoun)],
        target=3, expected=EXPECT_REFERENT, entity=person.name,
        pronoun=pronoun,
        note="между упоминанием и местоимением две реплики без сущностей")


_BUILDERS_V2 = {
    "distractor_class": _distractor_class,
    "correction_then_shift": _correction_then_shift,
    "negation_correction": _negation_correction,
    "plural_ambiguous": _plural_ambiguous,
    "long_distance": _long_distance,
}


def build_cases_v2(split: str, count: int, seed: int) -> list[BenchCase]:
    rng = random.Random(f"v2:{split}:{seed}")
    order = list(CATEGORIES_V2)
    rng.shuffle(order)
    cases: list[BenchCase] = []
    for i in range(count):
        category = order[i % len(order)]
        cases.append(_BUILDERS_V2[category](rng, split, i))
    return cases


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_split(path: Path, cases: list[BenchCase]) -> list[str]:
    lines = [json.dumps({"schema": BENCH_SCHEMA}, ensure_ascii=False)]
    lines += [json.dumps(c.as_dict(), ensure_ascii=False, sort_keys=True)
              for c in cases]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [c.id for c in cases]


# --------------------------------------------------------------------------
# loading + sealing
# --------------------------------------------------------------------------
@dataclass
class Split:
    name: str
    path: str
    sha256: str
    cases: list[BenchCase] = field(default_factory=list)

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for case in self.cases:
            counts[case.category] = counts.get(case.category, 0) + 1
        return {"name": self.name, "path": self.path, "sha256": self.sha256,
                "size": len(self.cases), "categories": counts}


def load_split(path: str | Path, name: str = "") -> Split:
    path = Path(path)
    if not path.exists():
        raise CorefBenchError(f"нет файла {path}")
    raw = path.read_bytes()
    cases: list[BenchCase] = []
    for lineno, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CorefBenchError(f"{path}:{lineno}: не JSON ({exc})") from exc
        if "schema" in data and "id" not in data:
            if data["schema"] != BENCH_SCHEMA:
                raise CorefBenchError(f"{path}: схема {data['schema']!r} "
                                      f"не поддерживается")
            continue
        try:
            cases.append(BenchCase(**data))
        except TypeError as exc:
            raise CorefBenchError(f"{path}:{lineno}: {exc}") from exc
    if not cases:
        raise CorefBenchError(f"{path}: пустой набор")
    return Split(name=name or path.stem, path=str(path),
                 sha256=hashlib.sha256(raw).hexdigest(), cases=cases)


def verify_sealed(split: Split, manifest: dict) -> None:
    """Refuse to evaluate a hidden split whose bytes changed after sealing."""
    entry = (manifest.get("splits") or {}).get(split.name)
    if not entry:
        raise CorefBenchError(f"в манифесте нет сплита {split.name!r}")
    if entry["sha256"] != split.sha256:
        raise CorefBenchError(
            f"{split.name}: sha256 {split.sha256[:12]}… != запечатанного "
            f"{entry['sha256'][:12]}… — набор изменён после фиксации")


def manifest_for(splits: dict[str, Split], seed: int) -> dict:
    return {"schema": MANIFEST_SCHEMA, "seed": seed,
            "generator": "rasm_cim_v17.discourse.coref_bench",
            "splits": {name: {"sha256": s.sha256, "size": len(s.cases),
                              "categories": s.summary()["categories"]}
                       for name, s in splits.items()}}


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson interval — correct for small n and extreme proportions."""
    if total == 0:
        return (0.0, 1.0)
    phat = successes / total
    denom = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total ** 2))
            / denom)
    return (max(0.0, round(centre - half, 4)), min(1.0, round(centre + half, 4)))


def _proportion(successes: int, total: int) -> dict:
    low, high = wilson(successes, total)
    return {"successes": successes, "total": total,
            "rate": round(successes / total, 4) if total else None,
            "ci95": [low, high]}


@dataclass
class Observation:
    case_id: str
    category: str
    expected: str
    entity: str
    bound_to: str
    refused: bool
    verdict: str          # "correct" | "wrong" | "refused" | "ungrounded"


def name_key(name: str) -> str:
    """Names are compared case- and ё-insensitively: the base stores «Петр»."""
    return (name or "").strip().casefold().replace("ё", "е")


def classify(case: BenchCase, bound_to: str, refused: bool) -> str:
    if case.expected == EXPECT_REFUSE:
        return "correct" if refused else "ungrounded"
    if refused:
        return "refused"
    return "correct" if name_key(bound_to) == name_key(case.entity) else "wrong"


def score(observations: list[Observation]) -> dict:
    """Aggregate the rubric into the four numbers the report quotes."""
    referent = [o for o in observations if o.expected == EXPECT_REFERENT]
    recency = [o for o in observations if o.expected == EXPECT_RECENCY]
    refuse = [o for o in observations if o.expected == EXPECT_REFUSE]

    binding_correct = sum(1 for o in referent if o.verdict == "correct")
    binding = _proportion(binding_correct, len(referent))

    recency_correct = sum(1 for o in recency if o.verdict == "correct")
    recency_block = _proportion(recency_correct, len(recency))

    refusal_correct = sum(1 for o in refuse if o.verdict == "correct")
    refusal = _proportion(refusal_correct, len(refuse))

    ungrounded = sum(1 for o in refuse if o.verdict == "ungrounded")
    ungrounded_block = _proportion(ungrounded, len(refuse))

    wrong = sum(1 for o in observations if o.verdict == "wrong")
    safe = binding_correct + recency_correct + refusal_correct
    overall = _proportion(safe, len(observations))

    return {
        "n": len(observations),
        "binding_accuracy": binding,
        "recency_accuracy": recency_block,
        "refusal_accuracy": refusal,
        "ungrounded_rate": ungrounded_block,
        "wrong_bindings": wrong,
        "refusals_on_unique": sum(1 for o in referent if o.verdict == "refused"),
        "safe_accuracy": overall,
        "by_category": _by_category(observations),
    }


def _by_category(observations: list[Observation]) -> dict:
    out: dict[str, dict] = {}
    for category in CATEGORIES + CATEGORIES_V2:
        rows = [o for o in observations if o.category == category]
        if not rows:
            continue
        correct = sum(1 for r in rows if r.verdict == "correct")
        block = _proportion(correct, len(rows))
        block["verdicts"] = {v: sum(1 for r in rows if r.verdict == v)
                             for v in ("correct", "wrong", "refused", "ungrounded")}
        out[category] = block
    return out


# --------------------------------------------------------------------------
# running the bench
# --------------------------------------------------------------------------
PRONOUN_SURFACES = {"он", "она", "оно", "они", "его", "её", "ее", "ему", "ей",
                    "им", "ею", "них", "ними", "него", "неё", "нее"}


def _is_pronoun(surface: str) -> bool:
    return (surface or "").strip().casefold() in PRONOUN_SURFACES


def clear_dialogue(dm) -> None:
    """Facts belong to the base, not to the conversation.

    The bench loads facts as KB setup and then starts a fresh dialogue, so a
    pronoun in the first turn has no antecedent *in the conversation*.  Without
    this step "no antecedent" cases would be measuring the teach turns instead,
    and every one of them would bind by construction.
    """
    dm.context.reset()
    try:
        dm.memory.clear_working()
    except Exception:                     # рабочая память может отсутствовать
        pass


def run_case(case: BenchCase, make_dm, respond_factory,
             reset_after_facts: bool = True) -> Observation:
    """Fresh database, teach the facts, start the dialogue, look at the target."""
    dm = make_dm()
    respond = respond_factory(dm)
    pipe = getattr(respond, "__self__", None)
    for fact in case.facts:
        respond(fact)
    if reset_after_facts:
        clear_dialogue(dm)
    bound_to, refused = "", False
    for index, text in enumerate(case.turns):
        respond(text)
        if index != case.target:
            continue
        meaning = getattr(pipe, "last_meaning", None) if pipe else None
        pronoun_focus = []
        if meaning is not None:
            for focus in meaning.focus:
                if _is_pronoun(focus.surface):
                    pronoun_focus.append(focus)
        resolved = [f for f in pronoun_focus if f.entity_id is not None]
        if resolved:
            bound_to = dm.kg.name_of(resolved[0].entity_id)
        else:
            clarification = (getattr(meaning, "clarification", "") or "") \
                if meaning is not None else ""
            refused = bool(pronoun_focus) or bool(clarification)
            if not refused:
                # парсер не выделил местоимение вовсе: это не связка, но и не
                # отказ; считаем отказом, чтобы не завышать точность
                refused = True
    return Observation(case_id=case.id, category=case.category,
                       expected=case.expected, entity=case.entity,
                       bound_to=bound_to, refused=refused,
                       verdict=classify(case, bound_to, refused))


def evaluate_split(split: Split, make_dm, respond_factory,
                   reset_after_facts: bool = True) -> dict:
    observations = [run_case(case, make_dm, respond_factory, reset_after_facts)
                    for case in split.cases]
    report = score(observations)
    report["split"] = split.summary()
    report["observations"] = [o.__dict__ for o in observations]
    return report


def render_score(report: dict, title: str = "") -> str:
    lines = [title or "Кореференция"]
    b = report["binding_accuracy"]
    r = report["refusal_accuracy"]
    u = report["ungrounded_rate"]
    rc = report["recency_accuracy"]
    s = report["safe_accuracy"]
    lines.append(f"  кейсов: {report['n']}")

    def half(name, block):
        if not block["total"]:
            return f"  {name}: —"
        return (f"  {name}: {block['successes']}/{block['total']} = "
                f"{block['rate']:.1%} (95 % ДИ {block['ci95'][0]:.1%}–"
                f"{block['ci95'][1]:.1%})")

    lines.append(half("связка на уникальный антецедент", b))
    lines.append(half("выбор по свежести (два кандидата)", rc))
    lines.append(half("правильный отказ", r))
    lines.append(half("необоснованные связки", u))
    lines.append(f"  ошибочных связок (не тот референт): {report['wrong_bindings']}; "
                 f"отказов там, где референт был: {report['refusals_on_unique']}")
    lines.append(half("итоговая безопасная точность", s))
    for category, block in report["by_category"].items():
        if not block["total"]:
            continue
        lines.append(f"    {category:16s} {block['successes']:3d}/{block['total']:<3d} "
                     f"= {block['rate']:5.1%}  {block['verdicts']}")
    return "\n".join(lines)
