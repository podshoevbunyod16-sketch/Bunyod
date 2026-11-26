"""CLI demo v0.4. Run: python3 -m rasm_cim_v4.demo4"""
import numpy as np
from rasm_cim_v4.rns4 import RNSv4
from rasm_cim_v4.graph4 import TruthGraph
from rasm_cim_v4.program4 import synthesize_split
from rasm_cim_v4.archD import ArchD
from rasm_cim_v3.program3 import execute

def main():
    print("="*64); print("RASM-CIM v0.4 — platform demo"); print("="*64)
    print("\n[1] RNS range-aware (no silent collisions)")
    rns = RNSv4((3,5,7,11))
    print(f"  M={rns.M}, encode 1154 ok, encode 1155 -> OverflowError (explicit)")
    try: rns.encode_unsigned(1155)
    except OverflowError as e: print(f"  {e}")
    print(f"  honest parity via RNS addition [1,0,1,1] sum={rns.sum_bits_rns([1,0,1,1])} parity={rns.sum_bits_rns([1,0,1,1])%2}")

    print("\n[2] TruthGraph: TRUE/FALSE/UNKNOWN/CONFLICT + history")
    g = TruthGraph(functional_relations={0})
    g.assert_true(0,0,1); print(f"  query (0,0,1)={g.query(0,0,1)} (TRUE)")
    g.assert_false(0,0,2); print(f"  query (0,0,2)={g.query(0,0,2)} (FALSE)")
    print(f"  query (9,0,1)={g.query(9,0,1)} (UNKNOWN)")
    g.assert_true(0,0,2); print(f"  query (0,0) functional={g.query(0,0)} (CONFLICT)")
    g.retract(0,0,2); print(f"  after retract query (0,0)={g.query(0,0)} (TRUE), history={len(g.history)} steps")

    print("\n[3] BFS synthesis with train/val/test split")
    res = synthesize_split([(0,6),(1,9)], [(2,12)], [(5,21)], max_depth=3)
    print(f"  program={res['program']} train={res['train_acc']} val={res['val_acc']} test={res['test_acc']} candidates={res['candidates']}")

    print("\n[4] Arch-D (no RNS/CAM/HDC): graph + BFS + tools")
    d = ArchD(seed=0); d.learn_fact(0,0,1); d.learn_fact(1,1,2)
    print(f"  compose 0->1->2: {d.compose_value(0,0,1)} (expect 2)")
    print(f"  tool add(20,22)={d.call_tool('add',20,22)}")
    print(f"  propagate from 0: {sorted(d.propagate(0, steps=2))}")

    print("\nDemo complete. Experiments: python3 -m rasm_cim_v4.experiments4")

if __name__ == "__main__":
    main()
