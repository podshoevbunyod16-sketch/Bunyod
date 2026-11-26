"""CLI demo v0.6 full. Run: python3 -m rasm_cim_v6_v2.demo6"""
import numpy as np
from rasm_cim_v6_v2.architectures.archA_memory import ArchA_MemoryCentric
from rasm_cim_v6_v2.architectures.archB_dynamic import ArchB_Dynamic
from rasm_cim_v6_v2.architectures.archC_algorithmic import ArchC_Algorithmic

def main():
    print("=" * 64); print("RASM-CIM v0.6 FULL — Arch A/B/C + 14 capabilities"); print("=" * 64)
    print("\n[A] Memory-Centric")
    A = ArchA_MemoryCentric(D=128, seed=0)
    A.write(1, 100); print(f"  dup: {A.write(1, 100)}"); A.write(1, 200)
    print(f"  query 1: {A.query_exact(1)} (CONFLICT)")
    print(f"  RNS 42 -> {A.decompress(A.compress(42))} (exact)")

    print("\n[B] Dynamic")
    B = ArchB_Dynamic(n_cells=64, seed=0)
    bits = np.zeros(64, dtype=int); bits[10] = 1; bits[20] = 1
    B.set_state(bits); _, dense = B.ca_step(rule=90, sparse=False)
    B.set_state(bits); _, sparse = B.ca_step(rule=90, sparse=True)
    print(f"  dense={dense} sparse={sparse}")

    print("\n[C] Algorithmic")
    C = ArchC_Algorithmic()
    e = C.learn([(0, 6), (1, 9)], [(2, 12)], [(5, 21)], max_depth=3)
    print(f"  program={e['program']} test={e['test_acc']} run(5)={C.run(e['program'], 5)}")

    print("\nExperiments: python3 -m rasm_cim_v6_v2.experiments.experiments6")

if __name__ == "__main__":
    main()
