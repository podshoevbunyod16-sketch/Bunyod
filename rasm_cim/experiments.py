"""
RASM-CIM experimental audit — runs E1..E9, saves results/results.json + plots.
All numbers are measured in this run. Theoretical claims are kept separate
in the final report; this file only reports measurements.
"""
from __future__ import annotations
import json, time, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rasm_cim.rns import RNS
from rasm_cim.fpe import IntegerFPE, ResidueFPE, max_offdiag_similarity
from rasm_cim.cam import AssociativeMemory, random_bipolar, flip_bits, hopfield_retrieve
from rasm_cim.state import DiagonalSSM, NonlinearRNNBaseline, cumulative_sum_baseline
from rasm_cim.resonator import Resonator, make_codebooks, bind_factors
from rasm_cim.logic import LogicMemory, DictLogicBaseline, ProgramMemory, reference_apply, generate_code
from rasm_cim.system import RASMCIM

rng_global = np.random.default_rng(42)
RESULTS = {}

def timed(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t0

# ---------------- E1: RNS correctness ----------------
def e1_rns():
    print("\n=== E1: RNS correctness ===")
    moduli_sets = [(3,5,7), (3,5,7,11), (11,13,17), (7,11,13,17,19)]
    out = {}
    for moduli in moduli_sets:
        rns = RNS(moduli)
        M = rns.M
        n = 2000
        xs = rng_global.integers(0, M, size=n)
        ys = rng_global.integers(0, M, size=n)
        # roundtrip
        rt_ok = all(rns.decode(rns.encode(int(x))) == int(x) for x in xs)
        # add/mul/sub modulo M
        add_ok = mul_ok = sub_ok = cmp_ok = True
        for x, y in zip(xs, ys):
            x, y = int(x), int(y)
            if rns.decode(rns.add(rns.encode(x), rns.encode(y))) != (x + y) % M:
                add_ok = False; break
        for x, y in zip(xs, ys):
            x, y = int(x), int(y)
            if rns.decode(rns.mul(rns.encode(x), rns.encode(y))) != (x * y) % M:
                mul_ok = False; break
        for x, y in zip(xs, ys):
            x, y = int(x), int(y)
            if rns.decode(rns.sub(rns.encode(x), rns.encode(y))) != (x - y) % M:
                sub_ok = False; break
        # compare on 200 pairs
        for _ in range(200):
            x, y = int(rng_global.integers(0, M)), int(rng_global.integers(0, M))
            expected = (x > y) - (x < y)
            if rns.compare(rns.encode(x), rns.encode(y)) != expected:
                cmp_ok = False; break
        key = "x".join(map(str, moduli))
        out[key] = {"M": M, "K": len(moduli), "roundtrip": bool(rt_ok),
                    "add": bool(add_ok), "mul": bool(mul_ok), "sub": bool(sub_ok),
                    "compare": bool(cmp_ok), "n_tested": n}
        print(f"  moduli={moduli} M={M} roundtrip={rt_ok} add={add_ok} mul={mul_ok} sub={sub_ok} compare={cmp_ok}")
    # division test: RNS has no native division — demonstrate failure mode
    rns = RNS((3,5,7))
    # 7 / 2 = 3.5 is not an integer, so RNS (integer ring Z_M) cannot represent it.
    # Naive componentwise "division" via modular inverses gives a wrong integer.
    a, b = rns.encode(7), rns.encode(2)
    try:
        naive = tuple((ai * pow(bi, -1, mi)) % mi if bi % mi != 0 else 0
                      for ai, bi, mi in zip(a, b, rns.moduli))
        naive_dec = rns.decode(naive)
    except Exception as e:
        naive_dec = f"error: {e}"
    out["division_naive_example"] = {"7/2 naive componentwise decode": naive_dec, "expected": "3.5 (not in Z_M)",
                                     "note": "RNS division is not closed; naive componentwise returns a spurious integer — documented limitation. Even exact division needs base-extension/CRT."}
    print(f"  division naive 7/2 -> {naive_dec} (expected 3.5, not representable) — limitation confirmed")
    RESULTS["E1_rns"] = out

# ---------------- E2: FPE ----------------
def e2_fpe():
    print("\n=== E2: FPE orthogonality & addition decoding ===")
    out = {}
    for D in [64, 256, 1024]:
        fpe = IntegerFPE(D=D, seed=1)
        vals = list(range(20))
        maxsim = max_offdiag_similarity(fpe, vals)
        # addition decoding: encode(a)⊙encode(b) should decode to a+b
        codebook = {v: fpe.encode(v) for v in range(40)}
        correct = 0; total = 0
        for a in range(10):
            for b in range(10):
                pred = fpe.decode(fpe.bind(fpe.encode(a), fpe.encode(b)), codebook)
                total += 1
                if pred == a + b:
                    correct += 1
        out[f"D={D}"] = {"max_offdiag_similarity": float(maxsim),
                          "addition_decoding_accuracy": correct/total, "n": total}
        print(f"  D={D} max_offdiag={maxsim:.3f} add_decode_acc={correct/total:.3f}")
    # Residue FPE periodicity
    rfpe = ResidueFPE(m=7, D=512, seed=2)
    periodic = np.allclose(rfpe.encode(3), rfpe.encode(10))  # 3 mod 7 == 10 mod 7
    out["residue_periodicity_m7"] = {"encode(3)==encode(10)": bool(periodic)}
    print(f"  ResidueFPE m=7 periodicity: {periodic}")
    RESULTS["E2_fpe"] = out

# ---------------- E3: CAM ----------------
def e3_cam():
    print("\n=== E3: CAM capacity, noise, timing ===")
    out = {}
    for D in [256, 512]:
        for N in [100, 500, 1000]:
            rng = np.random.default_rng(7)
            cam = AssociativeMemory(D=D)
            keys = [random_bipolar(rng, D) for _ in range(N)]
            for i, k in enumerate(keys):
                cam.write_hv(k, i)
            # clean retrieval (vectorized)
            correct_clean = 0
            for i in range(min(N, 100)):
                val, _ = cam.search_cam_vectorized(keys[i])
                if val == i:
                    correct_clean += 1
            # noisy retrieval at 10% flips
            correct_noisy = 0
            trials = 100
            for t in range(trials):
                i = int(rng.integers(0, N))
                noisy = flip_bits(keys[i], 0.10, rng)
                val, _ = cam.search_cam_vectorized(noisy)
                if val == i:
                    correct_noisy += 1
            out[f"D={D},N={N}"] = {"clean_acc_100": correct_clean/100,
                                    "noisy10_acc": correct_noisy/trials}
            print(f"  D={D} N={N} clean={correct_clean/100:.2f} noisy10={correct_noisy/trials:.2f}")
    # Timing: dict vs vectorized CAM vs linear scan
    print("  Timing (exact lookup, N=5000, D=512, 200 queries):")
    rng = np.random.default_rng(3)
    D, N = 512, 5000
    cam = AssociativeMemory(D=D)
    keys = [random_bipolar(rng, D) for _ in range(N)]
    for i, k in enumerate(keys):
        cam.write_hv(k, i)
        cam.write_residue((i % 97, i % 89, i % 83), i)
    queries = [keys[int(rng.integers(0, N))] for _ in range(200)]
    res_queries = [(i % 97, i % 89, i % 83) for i in range(200)]
    _, t_vec = timed(lambda: [cam.search_cam_vectorized(q) for q in queries])
    _, t_lin = timed(lambda: [cam.search_linear_hv(q) for q in queries])
    _, t_dict = timed(lambda: [cam.search_dict(q) for q in res_queries])
    out["timing_N5000"] = {"vectorized_cam_s": t_vec, "linear_scan_s": t_lin, "dict_s": t_dict}
    print(f"    vectorized={t_vec:.4f}s linear={t_lin:.4f}s dict={t_dict:.6f}s")
    # Hopfield retrieval sanity
    rng2 = np.random.default_rng(5)
    pats = rng2.choice(np.array([-1.0, 1.0]), size=(20, 256))
    q = pats[3] + rng2.normal(0, 0.3, size=256)
    retrieved, w = hopfield_retrieve(q, pats, beta=5.0)
    out["hopfield_sanity"] = {"top1_is_pattern3": bool(int(np.argmax(w)) == 3)}
    print(f"  Hopfield top1 correct: {int(np.argmax(w))==3}")
    RESULTS["E3_cam"] = out

# ---------------- E4: SSM state ----------------
def e4_ssm():
    print("\n=== E4: SSM sequence tasks ===")
    out = {}
    # Cumulative sum: linear SSM with lam=1 should track sum via readout? Test state norm correlation.
    # Simpler: compare SSM state sum vs true cumsum using B projection — we test that SSM
    # with lam=1 preserves cumulative information (state = sum of inputs projected).
    ssm = DiagonalSSM(input_dim=1, state_dim=8, decay=1.0, seed=0)
    # Override B to identity-like for interpretability: B = ones
    ssm.B = np.ones((8, 1))
    xs = rng_global.normal(size=(50, 1))
    ssm.reset()
    states = []
    for x in xs:
        ssm.step(x, event=True)
        states.append(ssm.h.copy())
    states = np.stack(states)
    true_cumsum = np.cumsum(xs[:, 0])
    # correlation between mean state and true cumsum should be ~1 for lam=1
    corr = float(np.corrcoef(states.mean(axis=1), true_cumsum)[0, 1])
    out["cumsum_lam1_state_corr"] = corr
    print(f"  cumsum lam=1 state-vs-true corr={corr:.4f} (expect ~1.0)")
    # Decay: with lam=0.5, old inputs fade — test impulse response
    ssm2 = DiagonalSSM(input_dim=1, state_dim=4, decay=0.5, seed=0)
    ssm2.B = np.ones((4, 1))
    ssm2.reset()
    ssm2.step(np.array([1.0]), event=True)
    h_after = float(ssm2.h.mean())
    for _ in range(10):
        ssm2.step(np.array([0.0]), event=False)  # no event: decay only
    h_decayed = float(ssm2.h.mean())
    out["decay_test"] = {"h_after_impulse": h_after, "h_after_10_decays": h_decayed,
                         "expected_ratio_0.5^10": 0.5**10}
    print(f"  decay: {h_after:.3f} -> {h_decayed:.6f} (ratio {h_decayed/h_after:.6f}, expect {0.5**10:.6f})")
    # Parity: linear SSM cannot solve parity — demonstrate with linear readout probe
    # Train linear probe (least squares) on SSM states to predict parity; expect near chance.
    from sklearn.linear_model import LogisticRegression
    ssm3 = DiagonalSSM(input_dim=1, state_dim=16, decay=0.9, seed=1)
    seqs, labels, feats = [], [], []
    for _ in range(200):
        bits = rng_global.integers(0, 2, size=8)
        ssm3.reset()
        for b in bits:
            ssm3.step(np.array([float(b)]), event=True)
        feats.append(ssm3.h.copy())
        labels.append(int(np.sum(bits) % 2))
    feats = np.stack(feats); labels = np.array(labels)
    # simple train/test split
    clf = LogisticRegression(max_iter=1000).fit(feats[:150], labels[:150])
    acc = float(clf.score(feats[150:], labels[150:]))
    out["parity_linear_probe_acc"] = acc
    print(f"  parity linear probe acc={acc:.3f} (chance=0.50) — linear SSM limitation")
    # Nonlinear RNN baseline on same parity task (untrained random — also expected poor without training)
    rnn = NonlinearRNNBaseline(input_dim=1, hidden=16, seed=1)
    feats_rnn = []
    for _ in range(200):
        bits = rng_global.integers(0, 2, size=8)  # note: different seqs, indicative only
        rnn.reset()
        for b in bits:
            rnn.step(np.array([float(b)]))
        feats_rnn.append(rnn.h.copy())
    out["note"] = "Parity probe uses linear readout on fixed (untrained) SSM states; failure is expected for linear systems and does not prove SSMs cannot do parity with nonlinearity/training."
    RESULTS["E4_ssm"] = out

# ---------------- E5: Resonator ----------------
def e5_resonator():
    print("\n=== E5: Resonator factorization vs brute force ===")
    out = {}
    for D, sizes in [(256, [10,10]), (256, [50,50]), (512, [50,50]), (512, [20,20,20])]:
        books = make_codebooks(sizes, D, seed=11)
        # single-restart (classic resonator) vs multi-restart (stochastic factorizer style)
        for label, n_restarts in [("single", 1), ("multi20", 20)]:
            res = Resonator(books, max_iter=50, seed=0, n_restarts=n_restarts)
            rng = np.random.default_rng(9)
            n_trials = 30
            correct = 0; iters_list = []
            for _ in range(n_trials):
                true_idx = tuple(int(rng.integers(0, s)) for s in sizes)
                comp = bind_factors(books, true_idx)
                pred, iters = res.factorize(comp)
                iters_list.append(iters)
                if pred == true_idx:
                    correct += 1
            acc = correct / n_trials
            key = f"D={D},sizes={sizes},{label}"
            out[key] = {"accuracy": acc, "mean_iters": float(np.mean(iters_list)), "n": n_trials, "n_restarts": n_restarts}
            print(f"  {key} acc={acc:.3f} mean_iters={np.mean(iters_list):.1f}")
    # Brute force timing for small case
    books = make_codebooks([10,10], 256, seed=11)
    res = Resonator(books, seed=0)
    comp = bind_factors(books, (3, 7))
    _, t_res = timed(lambda: res.factorize(comp))
    _, t_bf = timed(lambda: res.brute_force(comp))
    out["timing_10x10"] = {"resonator_s": t_res, "bruteforce_s": t_bf}
    print(f"  timing 10x10 resonator={t_res:.4f}s bruteforce={t_bf:.4f}s")
    RESULTS["E5_resonator"] = out

# ---------------- E6: Logic ----------------
def e6_logic():
    print("\n=== E6: Logic / compositional + noise ===")
    rng = np.random.default_rng(13)
    lm = LogicMemory(n_subjects=20, n_relations=3, n_objects=20, D=512, seed=4)
    dm = DictLogicBaseline()
    facts = []
    for s in range(15):
        r = int(rng.integers(0, 3))
        o = int(rng.integers(0, 20))
        facts.append((s, r, o))
        lm.add_fact(s, r, o)
        dm.add_fact(s, r, o)
    # clean query accuracy
    correct_hv = sum(1 for (s,r,o) in facts if lm.query(s,r)[0] == o)
    correct_dict = sum(1 for (s,r,o) in facts if dm.query(s,r) == o)
    # noisy query (20% flips on subj/rel vectors)
    correct_noisy = 0
    for (s,r,o) in facts:
        pred, _ = lm.query_noisy(s, r, 0.20, rng)
        if pred == o:
            correct_noisy += 1
    out = {"hv_clean_acc": correct_hv/len(facts), "dict_clean_acc": correct_dict/len(facts),
           "hv_noisy20_acc": correct_noisy/len(facts), "n_facts": len(facts)}
    print(f"  HV clean={correct_hv/len(facts):.3f} dict clean={correct_dict/len(facts):.3f} HV noisy20={correct_noisy/len(facts):.3f}")
    # Compositional: A->B, B->C stored; query A->C via two steps (dict transitive vs HV two-step)
    # Build chain facts
    lm2 = LogicMemory(n_subjects=30, n_relations=2, n_objects=30, D=512, seed=6)
    dm2 = DictLogicBaseline()
    chain = [(i, 0, i+1) for i in range(10)] + [(i+1, 1, 20+i) for i in range(10)]
    for (s,r,o) in chain:
        lm2.add_fact(s, r, o); dm2.add_fact(s, r, o)
    # dict transitive
    dict_comp = sum(1 for i in range(10) if dm2.transitive(i, 0, 1) == 20+i)
    # HV two-step: query (i,0)->mid, then (mid,1)->o
    hv_comp = 0
    for i in range(10):
        mid, _ = lm2.query(i, 0)
        if mid is None:
            continue
        o, _ = lm2.query(mid, 1)
        if o == 20+i:
            hv_comp += 1
    out["composition_dict"] = dict_comp/10
    out["composition_hv_twostep"] = hv_comp/10
    print(f"  composition dict={dict_comp/10:.3f} HV two-step={hv_comp/10:.3f}")
    RESULTS["E6_logic"] = out

# ---------------- E7: Programs ----------------
def e7_programs():
    print("\n=== E7: Program generation ===")
    pm = ProgramMemory(D=512, seed=8)
    rng = np.random.default_rng(15)
    ops = ["add","sub","mul","max","min","neg"]
    correct_decode = 0; correct_exec = 0; n = 60
    for _ in range(n):
        op = ops[int(rng.integers(0, len(ops)))]
        arg = int(rng.integers(-5, 6))
        hv = pm.encode(op, arg)
        (pop, parg), sim = pm.decode(hv)
        if (pop, parg) == (op, arg):
            correct_decode += 1
        # generate code from decoded program and execute
        code = generate_code(pop, parg)
        fn = eval(code)
        xs = [int(rng.integers(-10, 11)) for _ in range(5)]
        if all(fn(x) == reference_apply(op, arg, x) for x in xs):
            correct_exec += 1
    out = {"decode_acc": correct_decode/n, "exec_acc": correct_exec/n, "n": n}
    print(f"  decode={correct_decode/n:.3f} exec={correct_exec/n:.3f}")
    RESULTS["E7_programs"] = out

# ---------------- E8: Memory / scalability ----------------
def e8_memory():
    print("\n=== E8: Memory per item (measured) ===")
    import sys as _sys
    rns = RNS((3,5,7,11))
    # RNS residues: K small ints
    sample_res = rns.encode(12345)
    rns_bytes = _sys.getsizeof(sample_res) + sum(_sys.getsizeof(x) for x in sample_res)
    # HV D=512 float64
    hv = np.zeros(512, dtype=np.float64)
    hv_bytes = hv.nbytes
    hv_bytes_f32 = np.zeros(512, dtype=np.float32).nbytes
    # Python int + dict entry overhead (rough)
    py_int_bytes = _sys.getsizeof(12345)
    out = {"rns_residues_getsizeof": rns_bytes, "rns_K": rns.K, "rns_M": rns.M,
           "hv_D512_float64_bytes": int(hv_bytes), "hv_D512_float32_bytes": int(hv_bytes_f32),
           "python_int_getsizeof": py_int_bytes,
           "note": "getsizeof is Python object overhead, not hardware bits. Hardware RNS needs sum(log2(mi)) bits; HV needs D bits (bipolar) or D*32 bits (float)."}
    # Theoretical bits
    import math
    out["rns_theoretical_bits"] = sum(math.ceil(math.log2(m)) for m in rns.moduli)
    out["hv_bipolar_bits_D512"] = 512
    print(f"  RNS residues getsizeof={rns_bytes}B theoretical bits={out['rns_theoretical_bits']} M={rns.M}")
    print(f"  HV D=512 float64={hv_bytes}B float32={hv_bytes_f32}B bipolar bits=512")
    RESULTS["E8_memory"] = out

# ---------------- E9: End-to-end ----------------
def e9_e2e():
    print("\n=== E9: End-to-end associative recall (RASM-CIM) ===")
    sysm = RASMCIM(moduli=(3,5,7,11), D=512, seed=0)
    rng = np.random.default_rng(21)
    N = 200
    pairs = [(int(rng.integers(0, 1000)), int(rng.integers(0, 1000))) for _ in range(N)]
    # deduplicate keys for exact test
    seen = {}
    for k, v in pairs:
        seen[k] = v
    pairs = list(seen.items())[:N]
    for k, v in pairs:
        sysm.write_pair(k, v)
    # exact queries
    exact_correct = sum(1 for k, v in pairs[:50] if sysm.query_exact(k) == v)
    # noisy queries
    noisy_correct = 0
    for k, v in pairs[:50]:
        pred, _ = sysm.query_noisy(k, noise_frac=0.10)
        if pred == v:
            noisy_correct += 1
    out = {"exact_acc_50": exact_correct/50, "noisy10_acc_50": noisy_correct/50, "N_stored": len(pairs)}
    print(f"  exact={exact_correct/50:.3f} noisy10={noisy_correct/50:.3f} N={len(pairs)}")
    RESULTS["E9_e2e"] = out

def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    e1_rns(); e2_fpe(); e3_cam(); e4_ssm(); e5_resonator(); e6_logic(); e7_programs(); e8_memory(); e9_e2e()
    with open("results/results.json", "w") as f:
        json.dump(RESULTS, f, indent=2)
    print("\nSaved results/results.json")
    # Simple plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # FPE orthogonality vs D
        Ds = [64, 256, 1024]
        sims = [RESULTS["E2_fpe"][f"D={d}"]["max_offdiag_similarity"] for d in Ds]
        plt.figure(); plt.plot(Ds, sims, marker="o"); plt.xlabel("D"); plt.ylabel("max off-diag similarity")
        plt.title("FPE orthogonality vs D (lower=better)"); plt.savefig("plots/fpe_orthogonality.png", dpi=150); plt.close()
        # Resonator accuracy (exclude timing key)
        labels = [k for k in RESULTS["E5_resonator"].keys() if k != "timing_10x10"]
        accs = [RESULTS["E5_resonator"][k]["accuracy"] for k in labels]
        plt.figure(figsize=(10,4)); plt.bar(range(len(labels)), accs); plt.xticks(range(len(labels)), labels, rotation=25, ha="right")
        plt.ylabel("accuracy"); plt.title("Resonator factorization accuracy (single vs multi-restart)"); plt.tight_layout(); plt.savefig("plots/resonator_accuracy.png", dpi=150); plt.close()
        print("Saved plots/*.png")
    except Exception as e:
        print(f"Plotting skipped: {e}")

if __name__ == "__main__":
    main()
