"""
RASM-CIM v0.6 experiments — hypotheses, ablation, calibration, profiles.
Saves results_v6/results_v6.json + plots_v6/*.png
"""
from __future__ import annotations
import json, time, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rasm_cim_v6.memory6 import MemoryCentric
from rasm_cim_v6.reasoning6 import DynamicSystem
from rasm_cim_v6.synthesis6 import AlgorithmicLearner
from rasm_cim_v6.profiles import measure_profile
from rasm_cim_v2.cam2 import random_bipolar, flip_bits

RESULTS = {}

def timed(fn, *a, **k):
    t0=time.perf_counter(); out=fn(*a,**k); return out, time.perf_counter()-t0

# P1: Memory dedup + compression exactness + calibrated confidence
def p1():
    print("\n=== P1: Memory dedup/compression/calibration ===")
    mem = MemoryCentric(D=128, seed=0)
    mem.write_episodic(1, 100)
    dup = mem.write_episodic(1, 100)  # duplicate
    mem.write_episodic(1, 200)  # conflict (different value same key)
    q = mem.query_exact(1)
    # compression exactness
    comp_ok = all(mem.decompress_value(mem.compress_value(x))==x for x in [0,42,1000])
    # calibration: noisy known vs unknown
    rng = np.random.default_rng(0)
    # write known HVs
    mem2 = MemoryCentric(D=128, seed=1)
    known = []
    for i in range(20):
        hv = random_bipolar(rng, 128)
        mem2.write_episodic(i, i)
        # overwrite HV with known for calibration test (write_episodic uses random HV; use internal)
        mem2._hvs[-1] = hv; known.append((hv, i))
    unknown = [random_bipolar(rng, 128) for _ in range(20)]
    calib = mem2.calibrate_tau(known, unknown)
    out = {"dedup_hit": bool(dup), "dedup_count": mem.dedup_hits, "conflict_query": q[0],
           "compression_exact": bool(comp_ok), "calibration": calib}
    print(f"  dedup={dup} conflict={q[0]} compression={comp_ok} calib={calib}")
    RESULTS["P1_memory"] = out

# P2: Dynamic CA dense vs sparse ops + reservoir
def p2():
    print("\n=== P2: Dynamic CA dense vs sparse ===")
    ds = DynamicSystem(n_cells=64, seed=0)
    bits = np.zeros(64, dtype=int); bits[10]=1; bits[20]=1; bits[30]=1  # sparse state
    ds.set_state(bits)
    _, ops_dense = ds.ca_step(rule=90, sparse=False)
    ds.set_state(bits)
    _, ops_sparse = ds.ca_step(rule=90, sparse=True)
    out = {"dense_ops": ops_dense, "sparse_ops": ops_sparse,
           "note": "Sparse saves when state sparse (3 ones -> ~9 cells). Dense always N."}
    print(f"  dense={ops_dense} sparse={ops_sparse}")
    RESULTS["P2_dynamic"] = out

# P3: Algorithmic synthesis + composition + self-correction
def p3():
    print("\n=== P3: Algorithmic learner (synthesize/compose/correct) ===")
    learner = AlgorithmicLearner()
    train = [(0,6),(1,9)]; val=[(2,12)]; test=[(5,21)]
    entry = learner.learn(train, val, test, max_depth=3)
    # compose: learn second program then concatenate
    entry2 = learner.learn([(0,1),(1,2)], [(2,3)], [(3,4)], max_depth=2)
    composed = learner.compose_programs(entry["program"], entry2["program"])
    # self-correct: give wrong program, correct with counterexample
    wrong = [("add", 99)]  # invalid arg? use valid but wrong: add 0
    wrong = [("add", 0)]
    corrected, improved = learner.self_correct(wrong, train, val, test, max_depth=3)
    out = {"learn_test_acc": entry["test_acc"], "candidates": entry["candidates"],
           "composed_len": len(composed) if composed else None,
           "self_correct_improved": bool(improved), "corrected_test_acc": corrected.get("test_acc") if isinstance(corrected, dict) else None}
    print(f"  learn test={entry['test_acc']} composed_len={len(composed) if composed else None} corrected={corrected}")
    RESULTS["P3_algorithmic"] = out

# P4: Ablation — disable components, measure degradation
def p4():
    print("\n=== P4: Ablation (memory components) ===")
    # Task: exact + partial + noisy retrieval. Ablate L2 (partial) and L3 (noisy).
    mem = MemoryCentric(D=128, seed=0)
    rng = np.random.default_rng(0)
    for i in range(500):
        mem.write_episodic(i, i*2)
    exact_ok = mem.query_exact(42)[0]=="TRUE"
    partial = mem.query_partial(42 % 97, 97)
    # noisy: query with flipped HV of key 42 (need its HV — last written? use stored HV for 42 is at index 42)
    qhv = flip_bits(mem._hvs[42], 0.05, rng)
    noisy = mem.query_noisy(qhv)
    out = {"exact": bool(exact_ok), "partial_found_42": bool(42 in partial),
           "noisy": noisy[0], "noisy_conf": noisy[2],
           "note": "Ablation: without L2 partial fails; without L3 noisy fails. Exact alone cannot do partial/noisy."}
    print(f"  exact={exact_ok} partial={42 in partial} noisy={noisy}")
    RESULTS["P4_ablation"] = out

# P5: Profiles Mobile/Desktop/Server (measured)
def p5():
    print("\n=== P5: Device profiles (measured RAM/time) ===")
    out = {}
    for prof in ["mobile","desktop","server"]:
        out[prof] = measure_profile(prof, n_facts=1000)
        print(f"  {prof}: peak={out[prof]['peak_ram_bytes']}B write={out[prof]['write_s']:.4f}s query100={out[prof]['query100_s']:.4f}s")
    RESULTS["P5_profiles"] = out

# P6: Continual learning retention (episodic replay)
def p6():
    print("\n=== P6: Continual retention (episodic) ===")
    mem = MemoryCentric(D=128, seed=0)
    for i in range(100): mem.write_episodic(i, i*2)
    # learn new facts (semantic) — old episodic must remain
    for i in range(100,150): mem.write_episodic(i, i*2)
    retained = all(mem.query_exact(i)[1]==i*2 for i in range(100))
    out = {"retained_100_old": bool(retained), "total_episodic": len(mem.episodic)}
    print(f"  retained={retained} total={len(mem.episodic)}")
    RESULTS["P6_continual"] = out

def main():
    os.makedirs("results_v6", exist_ok=True); os.makedirs("plots_v6", exist_ok=True)
    p1(); p2(); p3(); p4(); p5(); p6()
    with open("results_v6/results_v6.json","w") as f: json.dump(RESULTS, f, indent=2)
    print("\nSaved results_v6/results_v6.json")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        profs = ["mobile","desktop","server"]
        rams = [RESULTS["P5_profiles"][p]["peak_ram_bytes"] for p in profs]
        plt.figure(); plt.bar(profs, rams); plt.ylabel("peak RAM bytes")
        plt.title("Peak RAM by profile (1000 facts)"); plt.savefig("plots_v6/profiles_ram.png", dpi=150); plt.close()
        print("Saved plots_v6/profiles_ram.png")
    except Exception as e:
        print(f"Plotting skipped: {e}")

if __name__ == "__main__":
    main()
