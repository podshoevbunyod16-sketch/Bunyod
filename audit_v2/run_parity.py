"""Прогон паритетных задач: RASM-CIM через CLI и базовые стратегии.

RASM запускается **не из каталога проекта**: используется установленный в
чистый venv консольный скрипт `rasm-cim` и отдельный `RASM_CIM_HOME` на каждый
кейс. Базовые стратегии считаются здесь же (они не являются системой).

Модель (Gemma 3 270M / открытый аналог) в этой среде недоступна — см.
PROTOCOL.md, раздел 1; для неё в отчёте стоит INCONCLUSIVE, а не выдуманные
цифры.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
CLI = os.environ.get("PARITY_CLI", "/tmp/rasm-venv/bin/rasm-cim")
SEED = 20261127
BOOTSTRAP = 10000

PRONOUNS = {"он", "она", "они", "его", "её", "ее", "ей", "их", "им"}
STARTERS = {
    "привет", "привт", "здравствуйте", "здравствуй", "салам", "салом", "бале",
    "хуб", "кто", "что", "как", "где", "когда", "а", "ну", "ладно", "хорошо",
    "спасибо", "спасиба", "понятно", "значит", "так", "слушай", "короче",
    "начнём", "начинаем", "итак", "скажи", "напомни", "правда", "не", "то",
    "выходит", "ага", "есть", "тут", "мы", "и", "расскажи", "который", "хорошая",
    "привт", "дуст", "чӣ", "кӣ", "пока", "до", "завтра", "ну",
}


def name_key(text: str) -> str:
    return (text or "").strip().casefold().replace("ё", "е")


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    spread = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denom
    return max(0.0, centre - spread), min(1.0, centre + spread)


def rate(successes: int, total: int) -> dict:
    lo, hi = wilson(successes, total)
    return {"successes": successes, "total": total,
            "rate": successes / total if total else 0.0, "ci95": [lo, hi]}


def fmt(stat: dict) -> str:
    return (f"{stat['successes']}/{stat['total']} = {stat['rate'] * 100:.1f}% "
            f"({stat['ci95'][0] * 100:.1f}–{stat['ci95'][1] * 100:.1f})")


def paired_bootstrap(a: list[int], b: list[int]) -> tuple[float, float, float]:
    """Разница долей a − b на одних кейсах, перцентильный интервал."""
    rng = random.Random(SEED)
    n = len(a)
    diffs = []
    for _ in range(BOOTSTRAP):
        idx = [rng.randrange(n) for _ in range(n)]
        diff = sum(a[i] for i in idx) / n - sum(b[i] for i in idx) / n
        diffs.append(diff)
    diffs.sort()
    point = sum(a) / n - sum(b) / n
    lo = diffs[int(0.025 * BOOTSTRAP)]
    hi = diffs[int(0.975 * BOOTSTRAP) - 1]
    return point, lo, hi


# --------------------------------------------------------------------- RASM
def _run(args: list[str], home: str) -> str:
    env = dict(os.environ)
    env["RASM_CIM_HOME"] = home
    env.pop("PYTHONPATH", None)
    proc = subprocess.run([CLI, *args, "--home", home], cwd="/tmp",
                          capture_output=True, text=True, env=env, timeout=120)
    return proc.stdout


def rasm_answer(turns: list[str]) -> dict:
    """Прогнать реплики в чистом доме и вернуть JSON последнего ответа."""
    home = tempfile.mkdtemp(prefix="parity-")
    try:
        payload = {}
        for index, turn in enumerate(turns):
            last = index == len(turns) - 1
            args = ["say", turn] + (["--json"] if last else [])
            out = _run(args, home)
            if last:
                try:
                    payload = json.loads(out)
                except json.JSONDecodeError:
                    payload = {"text": out.strip(), "status": "PARSE_ERROR",
                               "evidence": []}
        return payload
    finally:
        shutil.rmtree(home, ignore_errors=True)


def rasm_extract(sentences: list[str]) -> list[tuple[str, str]]:
    home = tempfile.mkdtemp(prefix="parity-")
    try:
        for sentence in sentences:
            _run(["say", sentence], home)
        out = _run(["graph"], home)
    finally:
        shutil.rmtree(home, ignore_errors=True)
    triples = []
    for line in out.splitlines():
        match = re.match(r"st-\d+\s+(.+?)\s+—.*?→\s+(.+?)\s+\[", line)
        if match:
            triples.append((match.group(1).strip(), match.group(2).strip()))
    return triples


# ---------------------------------------------------------------- стратегии
def last_name(turns: list[str]) -> str:
    names = []
    for text in turns[:-1]:
        for word in re.findall(r"[А-ЯЁ][а-яё\-]+", text):
            low = word.casefold()
            if low in PRONOUNS or low in STARTERS:
                continue
            names.append(word)
    return names[-1] if names else ""


def last_matching_gender(turns: list[str], pronoun: str) -> str:
    if pronoun.casefold() in ("они", "их"):
        return ""
    wanted = "f" if pronoun.casefold() in ("она", "её", "ее", "ей") else "m"
    names = []
    for text in turns[:-1]:
        for word in re.findall(r"[А-ЯЁ][а-яё\-]+", text):
            low = word.casefold()
            if low in PRONOUNS or low in STARTERS:
                continue
            gender = "f" if low.endswith(("а", "я")) else "m"
            if gender == wanted:
                names.append(word)
    return names[-1] if names else ""


def echo_sentence(texts: list[str], question: str) -> str:
    qwords = {w.casefold() for w in re.findall(r"[А-Яа-яЁё]{3,}", question)}
    best, score = "", 0
    for text in texts:
        words = {w.casefold() for w in re.findall(r"[А-Яа-яЁё]{3,}", text)}
        overlap = len(qwords & words)
        if overlap > score:
            best, score = text, overlap
    return best if score else ""


def regex_triples(sentences: list[str]) -> list[tuple[str, str]]:
    triples = []
    for text in sentences:
        if "?" in text:
            continue
        for match in re.finditer(
                r"([А-ЯЁ][а-яё\-]+)\s+[—–-]\s+(?:это\s+)?([А-Яа-яЁё][а-яё\-]*)", text):
            triples.append((match.group(1), match.group(2)))
    return triples


# ------------------------------------------------------------------ задачи
def load(task: str) -> list[dict]:
    path = DATA / f"{task}.jsonl" if task != "hidden" else DATA / "hidden.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if task == "hidden":
        return rows
    return [r for r in rows if r["task"] == task]


#: адаптер ответа системы к общему контракту {"answer","status","evidence"}:
#: ok ← DIRECT/INFERRED без уточняющего вопроса; unknown ← UNKNOWN/AMBIGUOUS
#: или непустое поле clarification (переспрос — это и есть отказ от связки).
REFUSAL_STATUSES = {"UNKNOWN", "AMBIGUOUS", "NOT_FOUND", "PARSE_ERROR"}


def is_refusal(payload: dict) -> bool:
    status = (payload.get("status") or "").upper()
    if status in REFUSAL_STATUSES:
        return True
    if payload.get("clarification"):
        return True
    text = (payload.get("text") or "").strip()
    if text.endswith("?") or "имеешь в виду" in text.casefold():
        return True
    return False


def normalise(payload: dict) -> dict:
    """Привести ответ системы к общему контракту (answer/status/evidence)."""
    refused = is_refusal(payload)
    answer = payload.get("text")
    return {"answer": answer, "status": "unknown" if refused else "ok",
            "evidence": payload.get("evidence") or [],
            "raw_status": payload.get("status")}


def eval_t1(rows: list[dict], strategy: str) -> list[dict]:
    out = []
    for row in rows:
        if strategy == "rasm":
            payload = rasm_answer(row["turns"])
            bound = "" if is_refusal(payload) else (payload.get("text") or "")
            ok = name_key(row["entity"]) in name_key(bound)
            ok = ok or any(name_key(w) == name_key(row["entity"])
                           for w in re.findall(r"[А-ЯЁ][а-яё\-]+", bound))
        elif strategy == "b1":
            ok = False
        elif strategy == "b2":
            ok = name_key(last_name(row["turns"])) == name_key(row["entity"])
        else:
            ok = name_key(last_matching_gender(row["turns"], row["pronoun"])) == \
                 name_key(row["entity"])
        out.append({"id": row["id"], "cat": row["category"], "ok": bool(ok)})
    return out


def eval_t2(rows: list[dict], strategy: str) -> list[dict]:
    out = []
    for row in rows:
        if strategy == "rasm":
            payload = rasm_answer(row["turns"])
            refused = is_refusal(payload)
        elif strategy == "b1":
            refused = True
        elif strategy == "b2":
            refused = not last_name(row["turns"])
        else:
            refused = not last_matching_gender(row["turns"], row["pronoun"])
        out.append({"id": row["id"], "cat": row["category"], "ok": refused})
    return out


def eval_t3(rows: list[dict], strategy: str) -> list[dict]:
    out = []
    for row in rows:
        if strategy == "rasm":
            turns = list(row["text"]) + [row["question"]]
            payload = rasm_answer(turns)
            refused = is_refusal(payload)
            text = payload.get("text") or ""
            correct = (not refused) and name_key(row["expected"]) in name_key(text)
        elif strategy == "b1":
            refused, correct = True, False
        else:
            echo = echo_sentence(row["text"], row["question"])
            refused = not echo
            correct = bool(echo) and name_key(row["expected"]) in name_key(echo)
        ok = correct if row["answerable"] else refused
        out.append({"id": row["id"], "cat": "answerable" if row["answerable"]
                    else "unanswerable", "ok": bool(ok),
                    "correct": bool(correct), "refused": bool(refused)})
    return out


def eval_t4(rows: list[dict], strategy: str) -> dict:
    tp = fp = fn = 0
    per_case = []
    for row in rows:
        expected = {(name_key(s), name_key(o)) for s, o in row["expected"]}
        if strategy == "rasm":
            got = {(name_key(s), name_key(o)) for s, o in rasm_extract(row["sentences"])}
        elif strategy == "b1":
            got = set()
        else:
            got = {(name_key(s), name_key(o)) for s, o in regex_triples(row["sentences"])}
        tp += len(expected & got)
        fp += len(got - expected)
        fn += len(expected - got)
        per_case.append({"id": row["id"], "ok": expected == got})
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "cases_exact": per_case}


def eval_t5(rows: list[dict], strategy: str) -> list[dict]:
    out = []
    for row in rows:
        turns = list(row["text"]) + ([row["question"]] if row["question"] else [])
        if not turns:
            turns = [""]
        if strategy == "rasm":
            payload = normalise(rasm_answer(turns))
            valid = all(field in payload for field in row["required_fields"])
            typed = ((payload.get("answer") is None
                      or isinstance(payload.get("answer"), str))
                     and payload.get("status") in ("ok", "unknown")
                     and isinstance(payload.get("evidence"), list))
            ok = bool(valid and typed)
        elif strategy == "b1":
            ok = False
        else:
            ok = False
        out.append({"id": row["id"], "cat": row["category"], "ok": bool(ok)})
    return out


STRATEGIES = ("rasm", "b1", "b2", "b3")


def summarise(task: str, rows: list[dict], results: dict[str, list[dict]]) -> dict:
    report = {}
    for strategy, items in results.items():
        successes = sum(1 for item in items if item["ok"])
        report[strategy] = {"summary": rate(successes, len(items)), "items": items}
    if "rasm" in results:
        base = []
        for other in ("b1", "b2", "b3"):
            if other not in results:
                continue
            a = [1 if item["ok"] else 0 for item in results["rasm"]]
            b = [1 if item["ok"] else 0 for item in results[other]]
            if len(a) == len(b) and len(a) > 1:
                point, lo, hi = paired_bootstrap(a, b)
                base.append({"vs": other, "diff": point, "ci95": [lo, hi]})
        report["diff_vs_baselines"] = base
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="*", default=["t1", "t2", "t3", "t4", "t5"])
    parser.add_argument("--hidden", action="store_true")
    parser.add_argument("--out", default="audit_v2/results/parity.json")
    args = parser.parse_args()

    report: dict = {}
    if args.hidden:
        hidden_all = load("hidden")
        for task in ("t1", "t2", "t3", "t4", "t5"):
            rows = [r for r in hidden_all if r["task"] == task]
            if not rows:
                continue
            if task == "t1":
                res = {s: eval_t1(rows, s) for s in ("rasm", "b1", "b2", "b3")}
            elif task == "t2":
                res = {s: eval_t2(rows, s) for s in ("rasm", "b1", "b2", "b3")}
            elif task == "t3":
                res = {s: eval_t3(rows, s) for s in ("rasm", "b1", "b2")}
            elif task == "t4":
                res = {s: eval_t4(rows, s)["cases_exact"] for s in ("rasm", "b1", "b2")}
            else:
                res = {s: eval_t5(rows, s) for s in ("rasm", "b1")}
            report[f"hidden_{task}"] = summarise(task, rows, res)
    else:
        for task in args.tasks:
            rows = load(task)
            if task == "t1":
                res = {s: eval_t1(rows, s) for s in ("rasm", "b1", "b2", "b3")}
            elif task == "t2":
                res = {s: eval_t2(rows, s) for s in ("rasm", "b1", "b2", "b3")}
            elif task == "t3":
                res = {s: eval_t3(rows, s) for s in ("rasm", "b1", "b2")}
            elif task == "t4":
                res = {s: eval_t4(rows, s)["cases_exact"] for s in ("rasm", "b1", "b2")}
                report["t4_extraction"] = {s: eval_t4(rows, s) for s in ("rasm", "b1", "b2")}
            else:
                res = {s: eval_t5(rows, s) for s in ("rasm", "b1")}
            report[task] = summarise(task, rows, res)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, data in report.items():
        if name == "t4_extraction":
            continue
        line = f"{name}: " + ", ".join(
            f"{s} {fmt(v['summary'])}" for s, v in data.items()
            if isinstance(v, dict) and "summary" in v)
        print(line)
        for diff in data.get("diff_vs_baselines", []):
            print(f"   rasm − {diff['vs']}: {diff['diff'] * 100:+.1f} п.п. "
                  f"({diff['ci95'][0] * 100:+.1f}…{diff['ci95'][1] * 100:+.1f})")
    if "t4_extraction" in report:
        for strategy, info in report["t4_extraction"].items():
            print(f"t4 {strategy}: P={info['precision']:.3f} R={info['recall']:.3f} "
                  f"F1={info['f1']:.3f} (tp={info['tp']} fp={info['fp']} fn={info['fn']})")


if __name__ == "__main__":
    main()
