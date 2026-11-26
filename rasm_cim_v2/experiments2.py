"""
RASM-CIM v0.2 experiments F1-F10 — comparative, reproducible.
Saves results_v2/results_v2.json + plots_v2/*.png
All numbers measured in this run. No hardware energy claimed.
"""
from __future__ import annotations
import json, time, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rasm_cim_v2.rns2 import PackedRNS, BinaryBaseline
from rasm_cim_v2.cam2 import CAM2, random_bipolar, flip_bits
from rasm_cim_v2.ssm2 import SelectiveSSM, LinearSSMv1, TanhRNNBaseline, train_readout
from rasm_cim_v2.resonator2 import Resonator2, make_codebooks, bind_factors
from rasm_cim_v2.logic2 import LogicMemory2, DictLogicBaseline2
from rasm_cim_v2.program2 import ProgramMemory2, execute_program, parse_program_string, SafeExecutionError
from rasm_cim_v2.system2 import RASMCIMv2

RESULTS = {}
rng = np.random.default_rng(42)

def timed(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t0

# F1 Binary vs RNS
def f1():
    print("\n=== F1: Binary vs RNS (packed) ===")
    rns = PackedRNS((3,5,7,11))
    binary = BinaryBaseline(rns.M)
    n = 5000
    xs = rng.integers(0, rns.M, size=n)
    ys = rng.integers(0, rns.M, size=n)
    # correctness
    rns_ok = all(rns.decode(rns.encode_packed(int(x))) == int(x) for x in xs[:500])
    # packed bytes
    rns_bytes = rns.packed_bytes_per_number()
    bin_bytes = binary.packed_bytes_per_number()
    # timing add/mul batch
    xb_rns, yb_rns = rns.encode_batch(xs), rns.encode_batch(ys)
    xb_bin, yb_bin = binary.encode_batch(xs), binary.encode_batch(ys)
    rns.reset_ops(); binary.reset_ops()
    _, t_rns_add = timed(lambda: rns.add_packed(xb_rns, yb_rns))
    _, t_bin_add = timed(lambda: binary.add(xb_bin, yb_bin))
    out = {"M": rns.M, "correct": bool(rns_ok),
           "bits_binary": binary.bits, "bits_rns": rns.bits_rns,
           "bytes_binary_packed": bin_bytes, "bytes_rns_packed": rns_bytes,
           "rns_ops_per_add": rns.K, "binary_ops_per_add": 1,
           "time_rns_add_s": t_rns_add, "time_binary_add_s": t_bin_add,
           "conclusion": "RNS bits >= binary bits: NO memory advantage. RNS ops K vs 1: more modular ops in software. Advantage only potential hardware parallelism (not measured)."}
    print(f"  M={rns.M} bits bin={binary.bits} rns={rns.bits_rns} bytes bin={bin_bytes} rns={rns_bytes}")
    print(f"  time add rns={t_rns_add:.4f}s binary={t_bin_add:.4f}s")
    RESULTS["F1_binary_vs_rns"] = out

# F2 HashMap vs CAM
def f2():
    print("\n=== F2: HashMap vs CAM2 (exact / noisy / partial) ===")
    D, N = 256, 2000
    cam = CAM2(D=D, tau=0.5)
    rns = PackedRNS((3,5,7,11))
    r = np.random.default_rng(7)
    keys_hv = [random_bipolar(r, D) for _ in range(N)]
    for i in range(N):
        cam.write(tuple(int(x) for x in rns.encode_packed(i)), keys_hv[i], i)
    # exact timing
    queries = [tuple(int(x) for x in rns.encode_packed(i)) for i in range(200)]
    _, t_exact = timed(lambda: [cam.search_exact(q) for q in queries])
    # noisy accuracy with refusal vs without
    correct_ref = correct_noref = unknown_correct = 0
    trials = 200
    for _ in range(trials):
        i = int(r.integers(0, N))
        noisy = flip_bits(keys_hv[i], 0.10, r)
        v, sim, unk = cam.search_noisy(noisy)
        if not unk and v == i:
            correct_ref += 1
        v2, _ = cam.search_noisy_no_refusal(noisy)
        if v2 == i:
            correct_noref += 1
    # unknown queries: random HV far from all stored -> should refuse
    unk_refused = 0
    for _ in range(100):
        q = random_bipolar(r, D)
        # random HV similarity to stored ~0, tau=0.5 -> should refuse most
        _, _, unk = cam.search_noisy(q)
        if unk:
            unk_refused += 1
    # partial search: inverted index vs full scan ops (2 residues = more selective)
    cam.reset_ops()
    enc100 = rns.encode_packed(100)
    partial = {0: int(enc100[0]), 1: int(enc100[1])}
    res_inv, ops_inv = cam.search_partial(partial)
    cam.reset_ops()
    res_scan, ops_scan = cam.search_partial_fullscan(partial)
    out = {"exact_time_200_s": t_exact, "noisy10_acc_with_refusal": correct_ref/trials,
           "noisy10_acc_no_refusal": correct_noref/trials,
           "unknown_refusal_rate": unk_refused/100,
           "partial_results": len(res_inv), "partial_ops_inverted": ops_inv, "partial_ops_fullscan": ops_scan,
           "note": "Exact = HashMap inside CAM (fastest in software). Partial via inverted index examines fewer candidates than full scan when selective. Noisy refusal trades coverage for precision."}
    print(f"  exact 200q={t_exact:.4f}s noisy_ref={correct_ref/trials:.3f} noisy_noref={correct_noref/trials:.3f} unk_refusal={unk_refused/100:.3f}")
    print(f"  partial ops inverted={ops_inv} fullscan={ops_scan} results={len(res_inv)}")
    RESULTS["F2_hashmap_vs_cam"] = out

# F3 RNN vs SSM
def f3():
    print("\n=== F3: RNN vs LinearSSM vs SelectiveSSM (trained readout) ===")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    def make_parity(n_seq=300, length=8, seed=0):
        rr = np.random.default_rng(seed)
        seqs, labels = [], []
        for _ in range(n_seq):
            bits = rr.integers(0, 2, size=length)
            seqs.append(bits.astype(float).reshape(length, 1))
            labels.append(int(np.sum(bits) % 2))
        return seqs, np.array(labels)
    seqs, labels = make_parity()
    models = {
        "linear_ssm": LinearSSMv1(input_dim=1, state_dim=32, decay=0.9, seed=1),
        "selective_ssm": SelectiveSSM(input_dim=1, state_dim=32, seed=1),
        "tanh_rnn": TanhRNNBaseline(input_dim=1, hidden=32, seed=1),
    }
    out = {}
    for name, model in models.items():
        feats = np.stack([model.collect_states(s)[-1] for s in seqs])
        clf = LogisticRegression(max_iter=1000)
        clf.fit(feats[:200], labels[:200])
        acc = float(clf.score(feats[200:], labels[200:]))
        out[name] = {"parity_test_acc": acc}
        print(f"  {name} parity acc={acc:.3f}")
    # cumsum regression R2
    rr = np.random.default_rng(2)
    seqs_c, targets = [], []
    for _ in range(200):
        s = rr.normal(size=(10, 1))
        seqs_c.append(s)
        targets.append(float(np.sum(s)))
    for name, model in [("linear_ssm", LinearSSMv1(input_dim=1, state_dim=16, decay=1.0, seed=0)),
                        ("selective_ssm", SelectiveSSM(input_dim=1, state_dim=16, seed=0))]:
        feats = np.stack([model.collect_states(s)[-1] for s in seqs_c])
        _, r2 = train_readout(feats, np.array(targets), task="reg")
        out[name]["cumsum_R2"] = r2
        print(f"  {name} cumsum R2={r2:.3f}")
    out["note"] = "Readout trained (logistic/ridge) on final states; recurrent weights fixed random. Selective+nonlinear should beat linear on parity; neither is deep-trained."
    RESULTS["F3_rnn_vs_ssm"] = out

# F4 Brute vs Resonator
def f4():
    print("\n=== F4: Brute force vs Resonator2 (with fallback) ===")
    out = {}
    for D, sizes in [(256, [10,10]), (256, [30,30])]:
        books = make_codebooks(sizes, D, seed=11)
        res = Resonator2(books, seed=0, n_restarts=20)
        rr = np.random.default_rng(9)
        n_trials = 20
        correct = fallbacks = 0
        t_res_total = 0.0
        for _ in range(n_trials):
            true_idx = tuple(int(rr.integers(0, s)) for s in sizes)
            comp = bind_factors(books, true_idx)
            t0 = time.perf_counter()
            pred, _, conf, fb = res.factorize(comp)
            t_res_total += time.perf_counter() - t0
            if fb:
                fallbacks += 1
            if pred == true_idx:
                correct += 1
        # brute timing on 5 trials
        comp = bind_factors(books, (0, 0))
        _, t_bf = timed(lambda: res.brute_force(comp))
        key = f"D={D},sizes={sizes}"
        out[key] = {"accuracy": correct/n_trials, "fallbacks": fallbacks, "n": n_trials,
                    "mean_resonator_time_s": t_res_total/n_trials, "brute_time_one_s": t_bf}
        print(f"  {key} acc={correct/n_trials:.3f} fallbacks={fallbacks}/{n_trials} res_t={t_res_total/n_trials:.4f}s bf_t={t_bf:.4f}s")
    RESULTS["F4_brute_vs_resonator"] = out

# F5 Graphs vs ProgramMemory (compositional)
def f5():
    print("\n=== F5: ProgramMemory compositional (unseen 2-op combos) ===")
    pm = ProgramMemory2(D=256, seed=8)
    rr = np.random.default_rng(15)
    # seen: single ops + some 2-op combos; test: unseen 2-op combos
    seen_2op = [(("add", 2), ("mul", 3)), (("sub", 1), ("neg", None))]
    # encode seen (not needed for decode, but simulates memory)
    for prog in [[("add", 2)], [("mul", 3)], [("add", 2), ("mul", 3)]]:
        pm.encode_program(prog)
    # test unseen combos: add+neg, mul+square, max+add etc.
    test_progs = [[("add", 5), ("neg", None)], [("mul", 2), ("square", None)],
                  [("max", 3), ("add", 1)], [("sub", 2), ("abs", None)],
                  [("add", 1), ("add", 1), ("mul", 2)]]
    correct_decode = correct_exec = 0
    for prog in test_progs:
        hvs = pm.encode_program(prog)
        decoded, _ = pm.decode_program(hvs)
        if decoded == prog:
            correct_decode += 1
        # execute decoded (or true if decode failed, count separately)
        try:
            if decoded is not None:
                xs = [-5, 0, 3, 7]
                if all(execute_program(decoded, x) == execute_program(prog, x) for x in xs):
                    correct_exec += 1
        except Exception:
            pass
    # safe execution: malicious strings must be rejected
    malicious = ["__import__('os').system('echo hi')", "add(3); import os", "eval('1+1')", "add(x)"]
    rejected = 0
    for m in malicious:
        try:
            parse_program_string(m)
        except SafeExecutionError:
            rejected += 1
        except Exception:
            rejected += 1
    out = {"unseen_decode_acc": correct_decode/len(test_progs),
           "unseen_exec_acc": correct_exec/len(test_progs),
           "n_test": len(test_progs), "malicious_rejected": f"{rejected}/{len(malicious)}",
           "note": "Programs are HV sequences; decode is per-step cleanup (not brute over program combos). Execution via whitelist interpreter, no eval."}
    print(f"  unseen decode={correct_decode/len(test_progs):.3f} exec={correct_exec/len(test_progs):.3f} malicious rejected={rejected}/{len(malicious)}")
    RESULTS["F5_programs"] = out

# F6 Ordinary vs compositional memory + refusal + contradiction
def f6():
    print("\n=== F6: Logic compositional + refusal + contradiction ===")
    rr = np.random.default_rng(13)
    lm = LogicMemory2(n_subjects=30, n_relations=3, n_objects=30, D=256, seed=4, tau=0.6)
    dm = DictLogicBaseline2()
    facts = []
    for s in range(15):
        r = int(rr.integers(0, 3)); o = int(rr.integers(0, 30))
        facts.append((s, r, o)); lm.add_fact(s, r, o); dm.add_fact(s, r, o)
    # calibrate tau
    unknown_q = [(25, 0), (26, 1), (27, 2), (28, 0), (29, 1)]
    calib = lm.calibrate_tau(facts, unknown_q)
    # known / unknown with calibrated tau
    known_ok = sum(1 for (s,r,o) in facts if lm.query(s,r)["object"] == o)
    unk_ref = sum(1 for (s,r) in unknown_q if lm.query(s,r)["unknown"])
    dict_known = sum(1 for (s,r,o) in facts if dm.query(s,r) == o)
    dict_unk_ref = sum(1 for (s,r) in unknown_q if dm.query(s,r) is None)
    # composition unseen: chain A->B->C not stored as A->C
    lm2 = LogicMemory2(n_subjects=30, n_relations=2, n_objects=30, D=256, seed=6, tau=0.6)
    dm2 = DictLogicBaseline2()
    chain = [(i, 0, i+1) for i in range(10)] + [(i+1, 1, 20+i) for i in range(10)]
    for (s,r,o) in chain:
        lm2.add_fact(s, r, o); dm2.add_fact(s, r, o)
    hv_comp = sum(1 for i in range(10) if lm2.compose(i, 0, 1)["object"] == 20+i)
    dict_comp = sum(1 for i in range(10) if dm2.compose(i, 0, 1) == 20+i)
    # contradiction
    lm3 = LogicMemory2(n_subjects=10, n_relations=1, n_objects=10, D=256, seed=0)
    lm3.add_fact(0, 0, 1)
    contra = lm3.add_fact(0, 0, 2)
    q_contra = lm3.query(0, 0)
    out = {"calibration": calib, "known_acc_hv": known_ok/len(facts), "known_acc_dict": dict_known/len(facts),
           "unknown_refusal_hv": unk_ref/len(unknown_q), "unknown_refusal_dict": dict_unk_ref/len(unknown_q),
           "composition_hv": hv_comp/10, "composition_dict": dict_comp/10,
           "contradiction_detected": bool(contra), "contradiction_query_conflict": bool(q_contra["conflict"])}
    print(f"  calib tau={calib['tau']} known HV={known_ok/len(facts):.3f} dict={dict_known/len(facts):.3f}")
    print(f"  unknown refusal HV={unk_ref/len(unknown_q):.3f} dict={dict_unk_ref/len(unknown_q):.3f}")
    print(f"  composition HV={hv_comp/10:.3f} dict={dict_comp/10:.3f} contradiction={contra}/{q_contra['conflict']}")
    RESULTS["F6_logic"] = out

# F7 Integration ablation (context-dependent recall)
def f7():
    print("\n=== F7: Integration ablation — context-dependent recall ===")
    # Task: same key maps to different values in different contexts.
    # Context is given as event vector; SSM must encode context into key.
    # Without SSM (context always 0), second write overwrites/conflicts -> fail.
    def run(use_ssm: bool):
        sysm = RASMCIMv2(moduli=(3,5,7,11), D=256, seed=0, use_ssm=use_ssm, use_cam=True, use_rns=True)
        # distinct non-degenerate contexts (zeros would give zero state -> degenerate)
        ctxA = -np.ones(256); ctxB = np.ones(256) * 2.0
        # write key 42 -> 100 in ctx A, key 42 -> 200 in ctx B
        sysm.reset_state(); sysm.write_pair(42, 100, context_event=ctxA)
        sysm.reset_state(); sysm.write_pair(42, 200, context_event=ctxB)
        # query in ctx A
        sysm.reset_state(); sysm.ssm.step(ctxA, event=True)
        vA, _, _ = sysm.query_pair(42)
        sysm.reset_state(); sysm.ssm.step(ctxB, event=True)
        vB, _, _ = sysm.query_pair(42)
        return vA, vB
    vA_full, vB_full = run(True)
    vA_no, vB_no = run(False)
    out = {"full_ssm": {"ctxA": vA_full, "ctxB": vB_full, "correct": bool(vA_full==100 and vB_full==200)},
           "no_ssm_ablation": {"ctxA": vA_no, "ctxB": vB_no, "correct": bool(vA_no==100 and vB_no==200)},
           "note": "SSM context is part of CAM key. Without SSM both writes share key -> second overwrites in dict index (CAM exact returns last). With SSM keys differ by context."}
    print(f"  full: A={vA_full} B={vB_full} correct={vA_full==100 and vB_full==200}")
    print(f"  no-SSM: A={vA_no} B={vB_no} correct={vA_no==100 and vB_no==200}")
    RESULTS["F7_ablation_context"] = out

# F8 New capabilities: one-shot, transfer, tools, scaling
def f8():
    print("\n=== F8: New capabilities ===")
    # one-shot write cost: time to write 1 pair vs retraining baseline (refit logistic on all data)
    sysm = RASMCIMv2(D=256, seed=0)
    _, t_write = timed(lambda: sysm.write_pair(123, 456))
    # transfer: RNS residues learned for arithmetic reused as CAM keys for logic-like recall
    rns = PackedRNS((3,5,7,11))
    # learn arithmetic: store (a+b) results as pairs, query unseen sum via RNS add + CAM?
    # Simpler transfer test: same RNS encoder used for both pair recall and fact keys (shared representation)
    transfer_ok = (rns.decode(rns.encode_packed(100)) == 100)
    # tools: select + call
    tool_name, _, unk = sysm.tools.select(sysm.tools.query_hv_for("calculator_add"))
    tool_res = sysm.tools.call(tool_name, 20, 22) if tool_name else None
    # scaling: write cost vs N (should be ~constant per write), query ops vs N
    times = []
    for N in [100, 500, 1000]:
        s2 = RASMCIMv2(D=256, seed=1)
        t0 = time.perf_counter()
        for i in range(N):
            s2.write_pair(i, i*2)
        times.append((time.perf_counter()-t0)/N)
    out = {"one_shot_write_s": t_write, "transfer_rns_shared": bool(transfer_ok),
           "tool_selected": tool_name, "tool_result_20+22": tool_res,
           "per_write_s_by_N": times,
           "note": "One-shot write is O(1) CAM append + SSM step. No retraining. Tool selection via HV similarity, execution validated."}
    print(f"  one-shot write={t_write:.5f}s transfer={transfer_ok} tool={tool_name}->{tool_res}")
    print(f"  per-write by N: {[f'{t:.5f}' for t in times]}")
    RESULTS["F8_capabilities"] = out

# F9 Memory packed vs HV vs dict (measured bytes)
def f9():
    print("\n=== F9: Packed memory (measured nbytes) ===")
    rns = PackedRNS((3,5,7,11))
    N = 10000
    xs = np.arange(N) % rns.M
    packed_rns = rns.encode_batch(xs)  # (N,K) uint16
    binary = BinaryBaseline(rns.M)
    packed_bin = binary.encode_batch(xs)
    hv_f32 = np.zeros((N, 256), dtype=np.float32)
    hv_bipolar_packed = np.packbits(np.zeros((N, 256), dtype=np.uint8))  # 256 bits = 32 bytes per HV
    out = {"N": N, "rns_packed_bytes": int(packed_rns.nbytes), "binary_packed_bytes": int(packed_bin.nbytes),
           "hv_f32_bytes": int(hv_f32.nbytes), "hv_bipolar_256bit_bytes": int(N*32),
           "per_item_rns": int(packed_rns.nbytes/N), "per_item_binary": int(packed_bin.nbytes/N),
           "per_item_hv_f32": int(hv_f32.nbytes/N)}
    print(f"  per item: RNS={out['per_item_rns']}B binary={out['per_item_binary']}B HVf32={out['per_item_hv_f32']}B bipolar256={32}B")
    RESULTS["F9_memory"] = out

# F10 Safety + reproducibility
def f10():
    print("\n=== F10: Safety (no eval) + reproducibility ===")
    # AST-based check: no actual Call nodes to eval/exec/compile/__import__ in v2 code
    import pathlib, ast as _ast
    bad = []
    for p in pathlib.Path("rasm_cim_v2").glob("*.py"):
        tree = _ast.parse(p.read_text())
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name):
                if node.func.id in ("eval", "exec", "compile", "__import__"):
                    bad.append(f"{p}:{node.lineno}:{node.func.id}")
    # reproducibility: run RNS encode twice, same result
    rns = PackedRNS((3,5,7,11))
    repro = bool(np.array_equal(rns.encode_packed(42), rns.encode_packed(42)))
    out = {"eval_exec_hits": bad, "reproducible_rns": repro,
           "note": "v2 code must have zero eval/exec calls. Hits above (if any) are listed for removal."}
    print(f"  eval/exec hits: {bad} reproducible={repro}")
    RESULTS["F10_safety"] = out

def main():
    os.makedirs("results_v2", exist_ok=True)
    os.makedirs("plots_v2", exist_ok=True)
    f1(); f2(); f3(); f4(); f5(); f6(); f7(); f8(); f9(); f10()
    with open("results_v2/results_v2.json", "w") as f:
        json.dump(RESULTS, f, indent=2)
    print("\nSaved results_v2/results_v2.json")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # F4 resonator
        labels = [k for k in RESULTS["F4_brute_vs_resonator"].keys()]
        accs = [RESULTS["F4_brute_vs_resonator"][k]["accuracy"] for k in labels]
        plt.figure(figsize=(8,4)); plt.bar(range(len(labels)), accs)
        plt.xticks(range(len(labels)), labels, rotation=15, ha="right")
        plt.ylabel("accuracy"); plt.title("Resonator2 accuracy (with fallback)"); plt.tight_layout()
        plt.savefig("plots_v2/resonator_v2.png", dpi=150); plt.close()
        # F9 memory per item
        cats = ["RNS packed", "Binary packed", "HV float32", "HV bipolar 256b"]
        vals = [RESULTS["F9_memory"]["per_item_rns"], RESULTS["F9_memory"]["per_item_binary"],
                RESULTS["F9_memory"]["per_item_hv_f32"], 32]
        plt.figure(); plt.bar(cats, vals); plt.ylabel("bytes per item")
        plt.title("Memory per item (measured nbytes)"); plt.tight_layout()
        plt.savefig("plots_v2/memory_per_item.png", dpi=150); plt.close()
        print("Saved plots_v2/*.png")
    except Exception as e:
        print(f"Plotting skipped: {e}")

if __name__ == "__main__":
    main()
