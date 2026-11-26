"""
CLI demo for RASM-CIM v0.2.
Run: python3 -m rasm_cim_v2.demo2
"""
import numpy as np
from rasm_cim_v2.rns2 import PackedRNS
from rasm_cim_v2.system2 import RASMCIMv2
from rasm_cim_v2.program2 import ProgramMemory2, execute_program, parse_program_string, program_to_string

def main():
    print("="*64)
    print("RASM-CIM v0.2 — integrated demo (no eval, with refusal)")
    print("="*64)

    print("\n[1] Packed RNS vs binary (honest: no memory advantage)")
    rns = PackedRNS((3,5,7,11))
    print(f"  {rns} bits_binary={rns.bits_binary} bits_rns={rns.bits_rns} bytes_rns={rns.packed_bytes_per_number()}")

    print("\n[2] Context-dependent recall (SSM+CAM integration)")
    sysm = RASMCIMv2(D=256, seed=0, use_ssm=True)
    ctxA = -np.ones(256); ctxB = np.ones(256)*2.0
    sysm.reset_state(); sysm.write_pair(42, 100, context_event=ctxA)
    sysm.reset_state(); sysm.write_pair(42, 200, context_event=ctxB)
    sysm.reset_state(); sysm.ssm.step(ctxA, event=True); vA,_,_ = sysm.query_pair(42)
    sysm.reset_state(); sysm.ssm.step(ctxB, event=True); vB,_,_ = sysm.query_pair(42)
    print(f"  key 42 in ctxA -> {vA} (expect 100), in ctxB -> {vB} (expect 200)")

    print("\n[3] Logic: learn, query, refuse unknown, detect contradiction")
    sysm.learn_fact(0, 0, 1)
    print(f"  query (0,0): {sysm.query_fact(0,0)}")
    print(f"  query unknown (99,0) [out of range would error; use (5,0) unseen]: {sysm.query_fact(5,0)}")
    contra = sysm.learn_fact(0, 0, 2)
    print(f"  add contradictory (0,0)->2: detected={contra}, query conflict={sysm.query_fact(0,0)['conflict']}")

    print("\n[4] Compositional program (unseen 2-op) + safe execution")
    pm = ProgramMemory2(D=256, seed=8)
    prog = [("add", 5), ("neg", None)]
    hvs = pm.encode_program(prog)
    decoded, _ = pm.decode_program(hvs)
    print(f"  true: {program_to_string(prog)}  decoded: {program_to_string(decoded) if decoded else None}")
    print(f"  execute decoded on 10: {execute_program(decoded, 10)} (expect -15)")
    parsed = parse_program_string("add(3) -> mul(2)")
    print(f"  parsed 'add(3) -> mul(2)': {parsed} -> on 10: {execute_program(parsed, 10)} (expect 26)")
    try:
        parse_program_string("__import__('os').system('echo hi')")
        print("  ERROR: malicious not rejected!")
    except Exception as e:
        print(f"  malicious rejected: {e}")

    print("\n[5] Tool use via associative selection")
    name, sim, unk = sysm.tools.select(sysm.tools.query_hv_for("calculator_add"))
    print(f"  selected tool: {name} sim={sim:.2f} unknown={unk} -> call(20,22)={sysm.tools.call(name, 20, 22)}")

    print("\nDemo complete. Full audit: python3 -m rasm_cim_v2.experiments2")

if __name__ == "__main__":
    main()
