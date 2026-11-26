"""M4 — COVERAGE-GUIDED fuzzing with Atheris (the only coverage-guided
harness in v0.8; hypothesis file is property-based, mutators are random).

Targets (spec §4 list mapped to real Python surfaces — see the header note
in tests/test_m4_properties.py about PackedStore.from_bytes):
  t1  storage.deserialize_v1          (bytes → list | StorageError)
  t2  v2 page/index reader             (bytes → temp file → LazyStore.scan)
  t3  RNS.decode_unsigned              (bytes → int via from_bytes)
  t4  TruthGraph op script             (bytes → assert/retract ops, model-free crash search)

Usage: python3 rasm_cim_v8/fuzz/fuzz_targets.py <t1|t2|t3|t4> <seconds>
Run each with a generous wall time; crashes land in /tmp/rasm_v8_atheris/<t>_crash_*
and each crash MUST be re-minimised into a regression test below the pass
criteria line of the report.
"""
from __future__ import annotations
import atheris  # noqa: F401  (must import before instrumented modules)
import os
import sys

sys.path.insert(0, "/home/user")
TMPD = "/tmp/rasm_v8_atheris"
os.makedirs(TMPD, exist_ok=True)

with atheris.instrument_imports():
    from rasm_cim_v8.storage import format as F
    from rasm_cim_v8.storage.format import StorageError
    from rasm_cim_v8.storage.streaming import open_lazy, CorruptPageError
    from rasm_cim_v4.rns4 import RNSv4
    from rasm_cim_v4.graph4 import TruthGraph

import struct
ACCEPT = (StorageError, CorruptPageError, ValueError, struct.error)


def t1(data: bytes) -> None:
    try:
        F.deserialize_v1(data)
    except ACCEPT:
        pass


def t2(data: bytes) -> None:
    if len(data) < 25:
        return
    p = f"{TMPD}/fz.rasm"
    with open(p, "wb") as f:
        f.write(data)
    s = None
    try:
        s = open_lazy(p, on_bad="raise", cache_pages=2)
        for _ in s.scan():
            pass
    except ACCEPT:
        pass
    except MemoryError:
        pass  # huge declared n under a 1.9 GiB box: recorded, not a bug class
    finally:
        if s is not None:
            s.close()


RNS5 = RNSv4((3, 5, 7, 11, 13))


def t3(data: bytes) -> None:
    if len(data) < 10:
        return
    res = tuple(int.from_bytes(data[i * 2:i * 2 + 2], "little", signed=False)
                if i % 2 else -int.from_bytes(data[i * 2:i * 2 + 2], "little")
                for i in range(5))
    x = RNS5.decode_unsigned(res)
    assert 0 <= x < RNS5.M          # violated range = real bug, aborts fuzz


def t4(data: bytes) -> None:
    g = TruthGraph(functional_relations={0, 1})
    for i in range(0, min(len(data), 256) - 3, 4):
        op, s, r, o = data[i], data[i + 1] % 8, data[i + 2] % 4, data[i + 3] % 8
        if op < 86:
            g.assert_true(s, r, o)
        elif op < 171:
            g.assert_false(s, r, o)
        else:
            g.retract(s, r, o)
    for s in range(8):
        for r in range(4):
            g.query(s, r)
            g.get_true(s, r)


T = {"t1": t1, "t2": t2, "t3": t3, "t4": t4}

if __name__ == "__main__":
    name = sys.argv[1]
    secs = sys.argv[2] if len(sys.argv) > 2 else "45"
    corpus = f"{TMPD}/{name}_corpus"
    os.makedirs(corpus, exist_ok=True)
    # seed corpus: one valid artifact so the fuzzer starts near interesting state
    if name == "t1":
        open(f"{corpus}/seed", "wb").write(F.serialize_v1([(1, 2), (3, 4)]))
    elif name == "t2":
        F.write_v2(((i, i) for i in range(40)), f"{corpus}/seed", page_recs=8)
    atheris.Setup([sys.argv[0], f"-max_total_time={secs}",
                   f"-artifact_prefix={TMPD}/{name}_crash_",
                   f"-len_control=1024", corpus],
                  T[name].__call__)
    atheris.Fuzz()
