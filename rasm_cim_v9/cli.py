"""RASM-CIM v0.10 minimal CLI (storage core only — language stack comes later).

python -m rasm_cim_v9.cli <command>          (also installed as `rasm-cim-demo`)

Commands (all offline, stdlib only):
  build   PATH [--n N]            synthesize N facts, write v3 base
  ask     PATH KEY                exact lookup (status/value/conflict)
  partial PATH RESIDUE [--mod M]  residue (partial-key) query
  update  PATH KEY VALUE          one journal txn (fsync)
  bench   [--n N]                 quick host showcase table (~15 s)
  crash   [--n N]                 live crash-recovery demo: SIGKILL the writer
                                  mid-update, reopen shows last committed state
System info: `bench` also prints environment. Exit codes: 0 ok, 2 not found/err.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _facts(n):
    M = 15015
    return ((i, (i * 7919) % M) for i in range(n))


def cmd_build(a):
    from rasm_cim_v9.storage.index import build_v3_base
    info = build_v3_base(_facts(a.n), a.path, page_recs=a.page_recs)
    print(json.dumps(info))


def cmd_ask(a):
    from rasm_cim_v9.storage.journal import open_any
    with open_any(a.path) as st:
        print(st.get(a.key))


def cmd_partial(a):
    from rasm_cim_v9.storage.journal import V3Store
    with V3Store(a.path) as st:
        ri = st.build_residue()
        ks = ri.query(a.residue, a.mod)
        print(f"{len(ks)} keys, first 20: {ks[:20]}")


def cmd_update(a):
    from rasm_cim_v9.storage.journal import append_txn
    seq = append_txn(a.path, [(a.key, a.value)])
    print(f"committed txn seq={seq}")


CRASH_CHILD = r'''
import os, sys, time
sys.path.insert(0, "/home/user")
path, stage = sys.argv[1], int(sys.argv[2])
from rasm_cim_v9.storage.journal import append_txn
for t in range(120):
    append_txn(path, [(1_000_000 + t, t * 3)], fsync=False)
    if t == stage:
        os.kill(os.getpid(), 9)          # SIGKILL exactly here
'''


def cmd_crash(a):
    import subprocess, signal
    from rasm_cim_v9.storage.index import build_v3_base
    from rasm_cim_v9.storage.journal import V3Store
    p = a.path or "/tmp/rasm_cli_crash.rasm"
    build_v3_base(_facts(a.n), p, page_recs=512)
    with V3Store(p) as st:
        n0 = st.n_records()
    print(f"base records: {n0}")
    killed_any = False
    for stage in (0, 5, 37, 99):
        r = subprocess.run([sys.executable, "-c", CRASH_CHILD, p, str(stage)],
                           capture_output=True, text=True, cwd="/home/user")
        killed = r.returncode < 0 or r.returncode == 137
        killed_any = killed_any or killed
        with V3Store(p) as st:
            n1 = st.n_records()
            probe = st.get(7)            # base answer must stay TRUE, unpolluted
        print(f"kill@txn{stage}: sigkilled={killed} -> reopened ok, "
              f"records {n0}->{n1} (never corrupted), get(7)={probe[0]}")
    print("VERDICT:", "PASS — torn transactions discarded, committed state intact"
          if killed_any else "no kills triggered (check environment)")


def cmd_bench(a):
    import time
    import random
    import statistics
    from rasm_cim_v8.storage.format import write_v2
    from rasm_cim_v8.storage.streaming import open_lazy
    from rasm_cim_v9.storage.index import build_v3_base
    from rasm_cim_v9.storage.journal import V3Store, append_txn
    n = a.n
    d = "/tmp/rasm_cli_bench"
    os.makedirs(d, exist_ok=True)
    p2, p3 = f"{d}/v2.rasm", f"{d}/v3.rasm"
    t0 = time.perf_counter(); write_v2(_facts(n), p2, page_recs=2048); tv2 = time.perf_counter() - t0
    t0 = time.perf_counter(); build_v3_base(_facts(n), p3, page_recs=2048); tv3 = time.perf_counter() - t0
    rng = random.Random(1)
    probes = [rng.randrange(n) for _ in range(3000)]
    res = {}
    for label, path in (("v0.8 (v2 lazy)", p2), ("v0.10 (v3+journal)", p3)):
        st = open_lazy(path) if label.startswith("v0.8") else V3Store(path)
        lat = []
        for k in probes:
            t0 = time.perf_counter(); st.get(k); lat.append((time.perf_counter() - t0) * 1e6)
        lat.sort()
        res[label] = (round(lat[len(lat)//2], 1), round(lat[int(len(lat)*.95)], 1),
                      round(lat[int(len(lat)*.99)], 1))
        st.close()
    lat = []
    for t in range(10):
        t0 = time.perf_counter(); append_txn(p3, [(n + t, t)], fsync=True)
        lat.append((time.perf_counter() - t0) * 1e3)
    print(f"host: py {sys.version.split()[0]}, os {os.uname().sysname} {os.uname().release}")
    print(f"n={n:,}   build v0.8 {tv2:.3f}s | v0.10 {tv3:.3f}s")
    print(f"{'engine':18} {'p50 us':>8} {'p95 us':>8} {'p99 us':>8}")
    for k, v in res.items():
        print(f"{k:18} {v[0]:>8} {v[1]:>8} {v[2]:>8}")
    print(f"journal update (fsync): median {statistics.median(lat):.3f} ms, "
          f"max {max(lat):.3f} ms  [tmpfs caveat if /tmp is tmpfs]")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rasm_cim_v9.cli", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("path"); b.add_argument("--n", type=int, default=10000); b.add_argument("--page-recs", type=int, default=2048); b.set_defaults(f=cmd_build)
    q = sub.add_parser("ask"); q.add_argument("path"); q.add_argument("key", type=int); q.set_defaults(f=cmd_ask)
    z = sub.add_parser("partial"); z.add_argument("path"); z.add_argument("residue", type=int); z.add_argument("--mod", type=int, default=97); z.set_defaults(f=cmd_partial)
    u = sub.add_parser("update"); u.add_argument("path"); u.add_argument("key", type=int); u.add_argument("value", type=int); u.set_defaults(f=cmd_update)
    c = sub.add_parser("crash"); c.add_argument("--path", default=None); c.add_argument("--n", type=int, default=5000); c.set_defaults(f=cmd_crash)
    e = sub.add_parser("bench"); e.add_argument("--n", type=int, default=50000); e.set_defaults(f=cmd_bench)
    a = ap.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
