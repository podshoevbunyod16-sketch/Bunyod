"""CLI demo v0.3. Run: python3 -m rasm_cim_v3.demo3"""
import numpy as np
from rasm_cim_v3.memory3 import RNSv3, FixedCAM, GraphMemory
from rasm_cim_v3.program3 import synthesize, execute, program_str
from rasm_cim_v3.arch import ArchB

def main():
    print("="*64); print("RASM-CIM v0.3 — platform demo"); print("="*64)
    print("\n[1] RNS overflow-safe + signed + multibank")
    rns = RNSv3((3,5,7,11))
    print(f"  signed -100 -> {rns.decode_signed(rns.encode_signed(-100))}")
    try:
        rns.encode_unsigned(99999)
    except OverflowError as e:
        print(f"  overflow correctly raised: {e}")

    print("\n[2] CAM collision -> CONFLICT (not silent overwrite)")
    from rasm_cim_v2.cam2 import random_bipolar
    cam = FixedCAM(D=64); k = random_bipolar(np.random.default_rng(0), 64)
    cam.write((1,2,3), k, 100); cam.write((1,2,3), k, 200)
    print(f"  query same key: conflict={cam.query_exact((1,2,3)).conflict}")

    print("\n[3] Graph logic: compose + refuse + contradiction")
    g = GraphMemory(); g.add_edge(0,0,1); g.add_edge(1,1,2)
    print(f"  compose 0->1->2: {g.compose_bfs(0,0,1).value} (expect 2)")
    print(f"  unknown 9: unknown={g.query(9,0).unknown}")
    print(f"  contradiction add 0->99: {g.add_edge(0,0,99)}")

    print("\n[4] BFS program synthesis (length 4, beyond v0.2 max 3)")
    prog = synthesize([(0,9),(1,13),(2,17)], max_depth=4, max_nodes=50000)
    print(f"  synthesized: {program_str(prog)} -> on 2: {execute(prog, 2)} (expect 17)")

    print("\n[5] ArchB parity via RNS (exact) vs RNN (chance) — see H5/H6")
    b = ArchB()
    print(f"  parity [1,0,1,1] via RNS: {b.parity_rns([1,0,1,1])} (expect 1)")

    print("\nDemo complete. Experiments: python3 -m rasm_cim_v3.experiments3")

if __name__ == "__main__":
    main()
