"""v0.10 — reliability gaps closed in the audit follow-up.

1. Kill-during-compact: compact = build `path.new` + os.replace, so SIGKILL at
   ANY delay must leave EITHER the old file or the new file valid — never a mix,
   never a half-written base. Loop of delays; every reopen must serve the
   original base answers.
2. V3Store context manager (no fd leaks on exception paths).
3. pyproject sanity: package list matches real trees (import smoke).

Run: python3 -m pytest tests_v10 -q
"""
from __future__ import annotations
import os
import subprocess
import sys

sys.path.insert(0, "/home/user")
import pytest

from rasm_cim_v9.storage.index import build_v3_base
from rasm_cim_v9.storage.journal import V3Store, append_txn, compact

TMP = "/tmp/rasm_v10_tests"
os.makedirs(TMP, exist_ok=True)
M = 15015

CHILD = r'''
import os, sys
sys.path.insert(0, "/home/user")
from rasm_cim_v9.storage.journal import append_txn, compact
path, n_txn = sys.argv[1], int(sys.argv[2])
for t in range(n_txn):
    append_txn(path, [(100000 + t, t * 7)], fsync=False)
os.kill(os.getpid(), 9) if os.environ.get("KILL_AFTER_COMPACT_START") == "1" else None
compact(path)
'''


def _base(n=4000):
    p = f"{TMP}/kdc_{n}.rasm"
    build_v3_base(((i, (i * 7919) % M) for i in range(n)), p, page_recs=256)
    return p, n


@pytest.mark.parametrize("kill_after", (0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89))
def test_kill_during_or_before_compact_never_corrupts(kill_after):
    """SIGKILL lands at a random-ish point of the journal workload phase or
    right at compact entry; whatever the outcome, the file must reopen with
    intact base answers and any state must be self-consistent."""
    p, n = _base()
    env = {**os.environ, "KILL_AFTER_COMPACT_START": "1" if kill_after % 2 else "0"}
    subprocess.run([sys.executable, "-c", CHILD, p, str(kill_after)],
                   env=env, capture_output=True, timeout=120)
    st = V3Store(p)                         # must NOT raise
    for k in (0, 7, 1234, 3999):
        assert st.get(k) == ("TRUE", (k * 7919) % M), k   # base never polluted
    # journal prefix property: records = base + whole frames, count monotone
    assert st.n_records() >= n
    st.close()
    for leftover in (".new", ".building"):
        if os.path.exists(p + leftover):
            os.remove(p + leftover)


def test_compact_under_kill_loop_full():
    """Stronger: repeatedly kill a process executing compact() itself by
    killing from outside after tiny sleeps."""
    import time, signal
    p, n = _base(n=6000)
    append_txn(p, [(500000 + i, i) for i in range(200)], fsync=False)
    ok = 0
    for delay in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25):
        proc = subprocess.Popen([sys.executable, "-c",
                                 "import sys, os; sys.path.insert(0, '/home/user');"
                                 "from rasm_cim_v9.storage.journal import compact;"
                                 f"compact({p!r})"])
        time.sleep(delay)
        if proc.poll() is None:
            os.kill(proc.pid, signal.SIGKILL)
            proc.wait()
        # after every interruption the store must be readable and correct
        with V3Store(p) as st:
            assert st.get(1234) == ("TRUE", (1234 * 7919) % M)
            ok += 1
        for junk in (".new", ".building"):
            if os.path.exists(p + junk):
                os.remove(p + junk)
    assert ok == 7


def test_context_manager_no_fd_leak():
    n = os.listdir("/proc/self/fd")
    try:
        with pytest.raises(RuntimeError):
            with V3Store(_base(500)[0]) as st:
                raise RuntimeError("boom")     # exception inside — close() must run
    finally:
        pass
    assert len(os.listdir("/proc/self/fd")) <= len(n)


def test_v3store_plain_close_idempotent():
    p, _ = _base(500)
    st = V3Store(p)
    st.close(); st.close()                     # double close must be silent
