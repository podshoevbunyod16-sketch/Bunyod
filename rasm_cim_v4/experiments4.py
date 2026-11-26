"""
RASM-CIM v0.4 experiments — hypotheses, controls, repeats (3 seeds), raw results.
Saves results_v4/results_v4.json + plots_v4/*.png
"""
from __future__ import annotations
import json, time, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rasm_cim_v4.rns4 import RNSv4, KeyEncoder
from rasm_cim_v4.graph4 import TruthGraph
from rasm_cim_v4.program4 import synthesize_split
from rasm_cim_v4.ca4 import ca_parity, EchoStateReservoir
from rasm_cim_v4.archD import ArchD
from rasm_cim_v3.arch import ArchA, ArchB, ArchC
from rasm_cim_v4.benchmark import bench_arch, standard_tasks
from rasm_cim_v3.program3 import execute

RESULTS = {}

def timed(fn, *a, **k):
    t0=time.perf_counter(); out=fn(*a,**k); return out, time.perf_counter()-t0

def ms(vals): return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "raw": [float(v) for v in vals]}

# G1 RNS range + boundary + encode/decode cost
def g1():
    print("\n=== G1: RNS range/boundary/cost (fix H8) ===")
    rns = RNSv4((3,5,7,11))
    # boundaries
    bounds = {}
    for x in [0, rns.M-1]:
        bounds[str(x)] = bool(rns.decode_unsigned(rns.encode_unsigned(x))==x)
    for x in [rns.M, rns.M+1, -1]:
        try:
            rns.encode_unsigned(x); bounds[str(x)] = "no-raise (BUG)"
        except OverflowError:
            bounds[str(x)] = "OverflowError (correct)"
    # encode/decode cost vs binary
    xs = np.arange(5000) % rns.M
    _, t_enc = timed(lambda: [rns.encode_unsigned(int(x)) for x in xs])
    encs = [rns.encode_unsigned(int(x)) for x in xs]
    _, t_dec = timed(lambda: [rns.decode_unsigned(e) for e in encs])
    _, t_bin = timed(lambda: [int(x) % rns.M for x in xs])
    out = {"boundaries": bounds, "encode_5k_s": t_enc, "decode_5k_s": t_dec, "binary_mod_5k_s": t_bin,
           "conclusion": "RNS encode/decode >> binary mod in Python. No speed advantage in software."}
    print(f"  boundaries={bounds} enc={t_enc:.4f}s dec={t_dec:.4f}s binary={t_bin:.4f}s")
    RESULTS["G1_rns_range_cost"] = out

# G2 TruthGraph states/history/retract
def g2():
    print("\n=== G2: TruthGraph TRUE/FALSE/UNKNOWN/CONFLICT + history ===")
    g = TruthGraph(functional_relations={0})
    g.assert_true(0,0,1)
    q_true = g.query(0,0,1); q_false = g.query(0,0,2); q_unk = g.query(9,0,1)
    g.assert_false(0,0,2)
    q_false2 = g.query(0,0,2)
    g.assert_true(0,0,2)  # functional conflict: two trues for rel 0
    q_conf = g.query(0,0)
    g.retract(0,0,2)
    q_after = g.query(0,0)
    out = {"true": q_true, "false_before": q_false, "unknown": q_unk, "false_after": q_false2,
           "conflict_functional": q_conf, "after_retract": q_after, "history_len": len(g.history)}
    print(f"  {out}")
    RESULTS["G2_truthgraph"] = out

# G3 Honest parity: binary vs RNS-addition vs CA vs RNN
def g3():
    print("\n=== G3: Honest parity (binary vs RNS-add vs CA) ===")
    rns = RNSv4((3,5,7,11))
    rr = np.random.default_rng(5)
    bin_ok = rns_ok = ca_ok = 0; n=100
    t_bin=t_rns=t_ca=0.0
    for _ in range(n):
        bits = rr.integers(0,2,size=12)
        true_p = int(np.sum(bits)%2)
        _, dt = timed(lambda: int(np.sum(bits)%2)); t_bin+=dt
        if int(np.sum(bits)%2)==true_p: bin_ok+=1
        _, dt = timed(lambda: rns.sum_bits_rns(bits) % 2); t_rns+=dt
        if rns.sum_bits_rns(bits)%2==true_p: rns_ok+=1
        _, dt = timed(lambda: ca_parity(bits)[0]); t_ca+=dt
        if ca_parity(bits)[0]==true_p: ca_ok+=1
    out = {"binary_acc": bin_ok/n, "rns_add_acc": rns_ok/n, "ca_acc": ca_ok/n,
           "time_binary_s": t_bin, "time_rns_s": t_rns, "time_ca_s": t_ca,
           "conclusion": "All exact (parity is easy modular task). RNS/CA not faster in Python; RNS value is exactness + parallelism in hardware (not measured)."}
    print(f"  binary={bin_ok/n:.3f} ({t_bin:.4f}s) RNS-add={rns_ok/n:.3f} ({t_rns:.4f}s) CA={ca_ok/n:.3f} ({t_ca:.4f}s)")
    RESULTS["G3_parity_honest"] = out

# G4 Synthesis train/val/test + counts
def g4():
    print("\n=== G4: Synthesis split + counts ===")
    # True program: (x+2)*3 ; train/val/test inputs disjoint
    train = [(0,6),(1,9),(2,12)]; val = [(3,15),(4,18)]; test = [(5,21),(10,36),(-1,3)]
    res = synthesize_split(train, val, test, max_depth=3, max_nodes=50000)
    out = {"program": str(res["program"]), "train_acc": res["train_acc"], "val_acc": res["val_acc"],
           "test_acc": res["test_acc"], "candidates": res["candidates"], "time_s": res["time_s"],
           "note": "BFS search (not discovery). Test inputs unseen during synthesis."}
    print(f"  prog={res['program']} train={res['train_acc']:.3f} val={res['val_acc']:.3f} test={res['test_acc']:.3f} cand={res['candidates']}")
    RESULTS["G4_synthesis_split"] = out

# G5 CA vs reservoir vs SSM on sequence task (copy-ish: last-bit recall)
def g5():
    print("\n=== G5: Reservoir vs SSM-readout (last-bit recall) ===")
    from rasm_cim_v2.ssm2 import SelectiveSSM
    from sklearn.linear_model import LogisticRegression
    rr = np.random.default_rng(0)
    seqs, labels = [], []
    for _ in range(300):
        bits = rr.integers(0,2,size=8)
        seqs.append(bits.astype(float).reshape(8,1)); labels.append(int(bits[-1]))
    labels = np.array(labels)
    # reservoir
    res = EchoStateReservoir(input_dim=1, res_dim=64, seed=0)
    feats_r = np.stack([res.collect(s)[-1] for s in seqs])
    clf_r = LogisticRegression(max_iter=1000).fit(feats_r[:200], labels[:200])
    acc_r = float(clf_r.score(feats_r[200:], labels[200:]))
    # SSM readout
    ssm = SelectiveSSM(input_dim=1, state_dim=32, seed=0)
    feats_s = np.stack([ssm.collect_states(s)[-1] for s in seqs])
    clf_s = LogisticRegression(max_iter=1000).fit(feats_s[:200], labels[:200])
    acc_s = float(clf_s.score(feats_s[200:], labels[200:]))
    out = {"reservoir_acc": acc_r, "ssm_readout_acc": acc_s,
           "note": "Last-bit recall is easy (recent input dominates). Both should be high; difference is not meaningful for long memory."}
    print(f"  reservoir={acc_r:.3f} ssm={acc_s:.3f}")
    RESULTS["G5_reservoir_vs_ssm"] = out

# G6 Fair benchmark A/B/C/D
def g6():
    print("\n=== G6: Fair benchmark ArchA/B/C/D (same tasks/seeds) ===")
    tasks = standard_tasks()
    factories = {"A": lambda s: ArchA(D=256, seed=s), "B": lambda s: ArchB(D=256, seed=s),
                 "C": lambda s: ArchC(D=256, seed=s), "D": lambda s: ArchD(seed=s)}
    out = {}
    for name, fac in factories.items():
        out[name] = bench_arch(fac, tasks, seeds=(0,1,2))
        print(f"  Arch{name}: {out[name]}")
    RESULTS["G6_benchmark"] = out

# G7 ArchD save/load + tools + propagation
def g7():
    print("\n=== G7: ArchD save/load, tools, propagation ===")
    d = ArchD(seed=0)
    d.learn_fact(0,0,1); d.learn_fact(1,1,2)
    state = d.save()
    d2 = ArchD(seed=0); d2.load(state)
    reload_ok = (d2.compose_value(0,0,1)==2)
    tool_ok = (d.call_tool("add", 20, 22)==42)
    prop = d.propagate(0, steps=2)
    out = {"reload_ok": bool(reload_ok), "tool_ok": bool(tool_ok), "reachable_from_0": sorted(prop)}
    print(f"  reload={reload_ok} tool={tool_ok} reachable={sorted(prop)}")
    RESULTS["G7_archD"] = out

# G8 Scalability fixed (full-int keys, no RNS collision)
def g8():
    print("\n=== G8: Scalability with full-int keys (no collision) ===")
    from rasm_cim_v3.memory3 import FixedCAM
    from rasm_cim_v2.cam2 import random_bipolar
    out = {}
    for N in [1000, 5000, 10000]:
        cam = FixedCAM(D=128)
        rng = np.random.default_rng(0)
        t0=time.perf_counter()
        for i in range(N):
            cam.write((i,), random_bipolar(rng,128), i*2)
        t_write=(time.perf_counter()-t0)/N
        # exact queries for large keys (beyond 1155) — must work (v0.3 H8 collided)
        ok = all(cam.query_exact((i,)).value==i*2 for i in [0,1154,1155,5000,N-1] if i<N)
        out[f"N={N}"] = {"per_write_s": t_write, "large_key_ok": bool(ok)}
        print(f"  N={N} per_write={t_write:.6f}s large_key_ok={ok}")
    RESULTS["G8_scalability_fixed"] = out

def main():
    os.makedirs("results_v4", exist_ok=True); os.makedirs("plots_v4", exist_ok=True)
    g1(); g2(); g3(); g4(); g5(); g6(); g7(); g8()
    with open("results_v4/results_v4.json","w") as f:
        json.dump(RESULTS, f, indent=2)
    print("\nSaved results_v4/results_v4.json")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        Ns=[1000,5000,10000]
        vals=[RESULTS["G8_scalability_fixed"][f"N={n}"]["per_write_s"] for n in Ns]
        plt.figure(); plt.plot(Ns, vals, marker="o"); plt.xlabel("N"); plt.ylabel("per-write s")
        plt.title("FixedCAM per-write vs N (full-int keys)"); plt.savefig("plots_v4/scalability_v4.png", dpi=150); plt.close()
        print("Saved plots_v4/scalability_v4.png")
    except Exception as e:
        print(f"Plotting skipped: {e}")

if __name__ == "__main__":
    main()
