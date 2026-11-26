"""CLI demo v0.5 NEXTGEN. Run: python3 -m rasm_cim_v5.demo5"""
import numpy as np
from rasm_cim_v5.system5 import NEXTGEN
from rasm_cim_v5.learn5 import make_task

def main():
    print("="*64); print("NEXT-GEN v0.5 — competing system (no LLM core)"); print("="*64)
    s = NEXTGEN(seed=0)
    s.learn_fact(0, (1,), 2)
    s.learn_pair(42, 999)
    seqs, labels = make_task(200, 1, "lastbit")
    s.train_sequence_classifier(seqs, labels)
    queries = [
        {"type":"math","op":"mul","a":12,"b":11},
        {"type":"parity","bits":[1,0,1,1]},
        {"type":"fact","rel":0,"args":[1]},
        {"type":"memory","key":42},
        {"type":"program","train":[(0,6),(1,9)],"val":[(2,12)],"test":[(5,21)]},
        {"type":"sequence","seq":[0,1,0,1,0,1,0,1]},
        {"type":"unknown_type"},
    ]
    for q in queries:
        print(f"  {q} -> {s.ask(q)}")
    print("\nNo LLM used as core. Language generation NOT implemented (honest limitation).")
    print("Experiments: python3 -m rasm_cim_v5.experiments5")

if __name__ == "__main__":
    main()
