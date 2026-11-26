"""CLI demo v0.6. Run: python3 -m rasm_cim_v6.demo6"""
import numpy as np
from rasm_cim_v6.memory6 import MemoryCentric
from rasm_cim_v6.reasoning6 import DynamicSystem
from rasm_cim_v6.synthesis6 import AlgorithmicLearner

def main():
    print("="*64); print("RASM-CIM v0.6 — Arch A/B/C demo"); print("="*64)
    print("\n[A] Memory-Centric: dedup, conflict, calibrated noisy")
    mem = MemoryCentric(D=128, seed=0)
    mem.write_episodic(1, 100); print(f"  dup write detected: {mem.write_episodic(1,100)}")
    mem.write_episodic(1, 200); print(f"  query key 1: {mem.query_exact(1)} (CONFLICT)")
    print(f"  RNS compression 42 -> {mem.decompress_value(mem.compress_value(42))} (exact)")

    print("\n[B] Dynamic: sparse CA saves ops")
    ds = DynamicSystem(n_cells=64, seed=0)
    bits = np.zeros(64, dtype=int); bits[10]=1; bits[20]=1
    ds.set_state(bits); _, dense = ds.ca_step(rule=90, sparse=False)
    ds.set_state(bits); _, sparse = ds.ca_step(rule=90, sparse=True)
    print(f"  dense ops={dense}, sparse ops={sparse} (sparse wins on sparse state)")

    print("\n[C] Algorithmic: synthesize + verify + compose")
    learner = AlgorithmicLearner()
    e = learner.learn([(0,6),(1,9)], [(2,12)], [(5,21)], max_depth=3)
    print(f"  program={e['program']} test_acc={e['test_acc']} candidates={e['candidates']}")
    print(f"  run on 5: {learner.run(e['program'], 5)} (expect 21)")

    print("\nDemo complete. Experiments: python3 -m rasm_cim_v6.experiments6")

if __name__ == "__main__":
    main()
