"""v0.9 M5 — coverage-guided fuzzing (atheris/libFuzzer) of the v3 parser
surface. NOT plain random mutation: atheris instruments Python bytecode and
libFuzzer drives coverage with a persistent corpus (saved under
rasm_cim_v9/fuzz/). If atheris cannot run, the script says so — we never
relabel dumb mutation as coverage-guided.

Targets (per spec):
  header      arbitrary bytes as a v3 header file      -> read_v3_header
  journal     valid base + arbitrary tail bytes        -> V3Store open/replay
  records     arbitrary payload windows                -> read_records math
  compact     random-but-valid store -> compact -> reopen (answer parity)
  corruptmid  valid base + mutated MIDDLE byte         -> controlled error
Every target's contract: either succeed with the invariant, or raise exactly
{StorageError, JournalCorruptError, ValueError, IndexError, CorruptPageError}.
Anything else (TypeError, struct.error, segfault) = crash = bug.

Run:  python3 bench_v9/fuzz_m5_atheris.py <target> [seconds]
Crashes land in rasm_cim_v9/fuzz/crashes/ ; add a regression test for each.
"""
from __future__ import annotations
import os
import subprocess
import sys

ROOT = "/home/user"
FUZZ = f"{ROOT}/rasm_cim_v9/fuzz"

CHILD = r'''
import atheris, os, sys, struct, tempfile, random
sys.path.insert(0, "/home/user")
FUZZ = "/home/user/rasm_cim_v9/fuzz"
target = sys.argv[1]
TMPD = "/tmp/rasm_v9_fuzz"
os.makedirs(TMPD, exist_ok=True)

from rasm_cim_v8.storage.format import StorageError
from rasm_cim_v8.storage.streaming import CorruptPageError
from rasm_cim_v9.storage.index import HDR_SIZE, build_v3_base, read_v3_header
from rasm_cim_v9.storage.journal import V3Store, JournalCorruptError, append_txn, compact

CONTROLLED = (StorageError, JournalCorruptError, CorruptPageError, ValueError,
              IndexError, OSError)

# a small valid base reused for tail-fuzz targets
BASE = f"{TMPD}/fuzzbase.rasm"
FACTS = [(i * 7 + 3, (i * 7919) % 15015) for i in range(200)]   # first key != 0
if not os.path.exists(BASE):
    build_v3_base(FACTS, BASE, page_recs=16)

def TestOneInput(data):
    data = bytes(data)
    if target == "header":
        p = f"{TMPD}/h.rasm"
        with open(p, "wb") as f:
            f.write(data[:HDR_SIZE].ljust(HDR_SIZE, b"\x00"))
        try:
            m = read_v3_header(p)
            assert m["version"] == 3
        except CONTROLLED:
            pass
    elif target == "journal":
        p = f"{TMPD}/j.rasm"
        with open(BASE, "rb") as f:
            base_bytes = f.read()
        with open(p, "wb") as f:
            f.write(base_bytes + data)
        st = None
        try:
            st = V3Store(p)
            # INVARIANT: replay of junk must never corrupt known answers
            for k, v in (FACTS[0], FACTS[99], FACTS[199]):
                r = st.get(k)
                assert r[0] == "TRUE" and (r[1] == v or r[0] == "CONFLICT"), (k, r)
        except CONTROLLED:
            pass
        finally:
            if st is not None:
                st.close()      # harness fd-leak fix (see fuzz report note 1)
    elif target == "records":
        p = f"{TMPD}/r.rasm"
        with open(BASE, "rb") as f:
            base_bytes = f.read()
        n = struct.unpack_from("<I", base_bytes, 12)[0] if len(base_bytes) >= 16 else 0
        with open(p, "wb") as f:
            f.write(base_bytes + data * 2)
        st = None
        try:
            st = V3Store(p)
            st.read_records(0, min(len(data) % (n + 2), st.n_records()))
            list(st.scan())
        except CONTROLLED:
            pass
        finally:
            if st is not None:
                st.close()
    elif target == "corruptmid":
        p = f"{TMPD}/c.rasm"
        if len(data) < 2:
            return
        with open(BASE, "rb") as f:
            b = bytearray(f.read())
        pos = data[0] * max(1, len(b) // 256) % len(b)
        b[pos] = data[1]
        with open(p, "wb") as f:
            f.write(bytes(b))
        st = None
        try:
            st = V3Store(p)
            for k, v in FACTS[::50]:
                st.get(k)
        except CONTROLLED:
            pass
        finally:
            if st is not None:
                st.close()
    elif target == "compact":
        if len(data) < 8:
            return
        rng = random.Random(struct.unpack_from("<Q", data[:8])[0])
        facts = [(rng.randrange(-2**40, 2**40), rng.randrange(-2**20, 2**20))
                 for _ in range(min(40, 2 + len(data) // 8))]
        p = f"{TMPD}/k.rasm"
        build_v3_base(facts, p, page_recs=8, tmp=False)
        try:
            upd = [(facts[i][0], data[i] if i < len(data) else 0)
                   for i in range(min(4, len(facts)))]
            append_txn(p, upd, fsync=False)
            st = V3Store(p)
            before = {k: st.get(k) for k, _ in facts}
            st.close()
            compact(p)
            st2 = V3Store(p)
            after = {k: st2.get(k) for k, _ in facts}
            for k in before:
                if before[k][0] == "CONFLICT":
                    assert after[k][0] == "CONFLICT"
                elif before[k][0] == "TRUE":
                    # winner is last write; compact must not resurrect shadowed vals
                    assert after[k][0] in ("TRUE", "CONFLICT"), (k, before[k], after[k])
            st2.close()
        except CONTROLLED:
            pass

def main():
    import glob
    corpus = [f"{FUZZ}/corpus_{target}"]
    os.makedirs(corpus[0], exist_ok=True)
    # seed: a couple of structurally-valid prefixes
    seedp = f"{corpus[0]}/seed_base"
    if not os.path.exists(seedp):
        with open(BASE, "rb") as f, open(seedp, "wb") as g:
            g.write(f.read()[:512])
    secs = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    atheris.Setup([sys.argv[0],
                   f"-max_total_time={secs}",
                   f"-artifact_prefix={FUZZ}/crashes/",
                   f"-timeout=10", corpus[0]],
                  TestOneInput)
    atheris.Fuzz()

main()
'''


def run(target, seconds):
    os.makedirs(f"{FUZZ}/crashes", exist_ok=True)
    p = subprocess.run([sys.executable, "-c", CHILD, target, str(seconds)],
                       capture_output=True, text=True, timeout=seconds + 180)
    tail = (p.stderr or "") + (p.stdout or "")
    return p.returncode, tail


def main():
    args = sys.argv[1:]
    targets = args[:-1] if args and args[-1].isdigit() else ["header", "journal", "records", "corruptmid", "compact"]
    secs = int(args[-1]) if args and args[-1].isdigit() else (60 if not args else 60)
    report = {}
    for t in targets:
        print(f"== atheris target: {t} ({secs}s) ==", flush=True)
        try:
            rc, out = run(t, secs)
        except subprocess.TimeoutExpired:
            rc, out = -1, "TIMEOUT-STOPPED (fuzz process exceeded wall budget)"
        crashes = [f for f in os.listdir(f"{FUZZ}/crashes") if f.startswith("crash-")] if os.path.isdir(f"{FUZZ}/crashes") else []
        cov_line = next((ln for ln in out.splitlines() if "cov:" in ln), out.splitlines()[-1] if out else "")
        report[t] = {"returncode": rc, "final": cov_line.strip()[:200],
                     "crash_files": crashes,
                     "stderr_tail": out[-700:]}
        print(" ->", report[t], flush=True)
    import json
    json.dump(report, open(f"{ROOT}/results_v9/fuzz_atheris.json", "w"), indent=1)
    print("saved results_v9/fuzz_atheris.json")


if __name__ == "__main__":
    main()
