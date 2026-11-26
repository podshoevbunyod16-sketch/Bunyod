"""
CLI demo for RASM-CIM prototype.
Run: python3 -m rasm_cim.demo
"""
import numpy as np
from rasm_cim.rns import RNS
from rasm_cim.fpe import IntegerFPE
from rasm_cim.system import RASMCIM
from rasm_cim.logic import ProgramMemory, generate_code
from rasm_cim.resonator import Resonator, make_codebooks, bind_factors

def main():
    print("="*60)
    print("RASM-CIM prototype v0.1 — live demo")
    print("="*60)

    print("\n[1] RNS: encode/decode + carry-free arithmetic")
    rns = RNS((3,5,7,11))
    for x in [42, 100, 999]:
        print(f"  {x} -> residues {rns.encode(x)} -> decode {rns.decode(rns.encode(x))}")
    a, b = rns.encode(100), rns.encode(23)
    print(f"  100+23 in RNS -> {rns.decode(rns.add(a,b))} (expect 123)")
    print(f"  100*23 in RNS -> {rns.decode(rns.mul(a,b))} (expect 2300 mod {rns.M} = {2300 % rns.M})")

    print("\n[2] FPE: addition via phasor binding")
    fpe = IntegerFPE(D=512, seed=0)
    codebook = {v: fpe.encode(v) for v in range(30)}
    bound = fpe.bind(fpe.encode(7), fpe.encode(8))
    print(f"  encode(7)⊙encode(8) decodes to {fpe.decode(bound, codebook)} (expect 15)")

    print("\n[3] Associative memory: one-shot write + noisy query")
    sysm = RASMCIM(moduli=(3,5,7,11), D=512, seed=0)
    pairs = [(10,100),(20,200),(30,300),(42,999)]
    for k,v in pairs:
        sysm.write_pair(k,v)
    for k,v in pairs:
        exact = sysm.query_exact(k)
        noisy, sim = sysm.query_noisy(k, noise_frac=0.10)
        print(f"  key {k}: exact->{exact} (expect {v}), noisy10->{noisy} sim={sim:.2f}")

    print("\n[4] Resonator factorization (2 factors, multi-restart)")
    books = make_codebooks([10,10], 256, seed=11)
    res = Resonator(books, seed=0, n_restarts=20)
    true_idx = (3,7)
    comp = bind_factors(books, true_idx)
    pred, iters = res.factorize(comp)
    print(f"  true={true_idx} pred={pred} iters={iters} correct={pred==true_idx}")

    print("\n[5] Program generation: encode -> decode -> Python code -> exec")
    pm = ProgramMemory(D=512, seed=8)
    for op, arg in [("add",3),("mul",2),("max",5)]:
        hv = pm.encode(op, arg)
        (pop, parg), sim = pm.decode(hv)
        code = generate_code(pop, parg)
        fn = eval(code)
        print(f"  ({op},{arg}) -> decoded ({pop},{parg}) -> code: {code} -> f(10)={fn(10)} (expect {eval(generate_code(op,arg))(10)})")

    print("\nDemo complete. Run experiments for full audit: python3 -m rasm_cim.experiments")

if __name__ == "__main__":
    main()
