"""Unit tests for RASM-CIM v0.2. Run: python3 -m pytest test_rasm_v2.py -v"""
import numpy as np
import pytest
from rasm_cim_v2.rns2 import PackedRNS, BinaryBaseline
from rasm_cim_v2.cam2 import CAM2, random_bipolar, flip_bits
from rasm_cim_v2.ssm2 import SelectiveSSM
from rasm_cim_v2.resonator2 import Resonator2, make_codebooks, bind_factors
from rasm_cim_v2.logic2 import LogicMemory2
from rasm_cim_v2.program2 import ProgramMemory2, execute_program, parse_program_string, SafeExecutionError
from rasm_cim_v2.system2 import RASMCIMv2

def test_packed_rns_roundtrip():
    rns = PackedRNS((3,5,7,11))
    for x in [0, 42, 1000]:
        assert rns.decode(rns.encode_packed(x)) == x

def test_rns_no_memory_advantage_documented():
    rns = PackedRNS((3,5,7,11))
    # Honest: RNS bits >= binary bits
    assert rns.bits_rns >= rns.bits_binary
    assert rns.packed_bytes_per_number() >= BinaryBaseline(rns.M).packed_bytes_per_number()

def test_cam_refusal():
    rng = np.random.default_rng(0)
    cam = CAM2(D=128, tau=0.5)
    k = random_bipolar(rng, 128)
    cam.write((1,2,3), k, 99)
    # known noisy query should succeed
    v, sim, unk = cam.search_noisy(flip_bits(k, 0.05, rng))
    assert v == 99 and not unk
    # random far query should refuse (sim ~0 < 0.5)
    v2, sim2, unk2 = cam.search_noisy(random_bipolar(rng, 128))
    assert unk2 is True

def test_cam_partial_inverted():
    cam = CAM2(D=64)
    rns = PackedRNS((3,5,7,11))
    for i in range(100):
        cam.write(tuple(int(x) for x in rns.encode_packed(i)), random_bipolar(np.random.default_rng(i), 64), i)
    enc = rns.encode_packed(10)
    res, _ = cam.search_partial({0: int(enc[0]), 1: int(enc[1])})
    assert 10 in res

def test_ssm_context_distinct():
    s = SelectiveSSM(input_dim=64, state_dim=32, seed=0)
    s.reset(); s.step(-np.ones(64), event=True); a = s.context_id()
    s.reset(); s.step(np.ones(64)*2, event=True); b = s.context_id()
    assert a != b

def test_resonator_fallback_accuracy():
    books = make_codebooks([10,10], 256, seed=11)
    res = Resonator2(books, seed=0, n_restarts=20)
    comp = bind_factors(books, (3,7))
    pred, _, conf, fb = res.factorize(comp)
    assert pred == (3,7) and conf > 0.9

def test_logic_refusal_and_contradiction():
    lm = LogicMemory2(n_subjects=10, n_relations=1, n_objects=10, D=256, seed=0, tau=0.6)
    lm.add_fact(0, 0, 1)
    assert lm.query(0, 0)["object"] == 1
    # unknown subject (never seen as subject with this relation in facts? use unseen combo)
    # Our query searches over facts, so unseen (9,0) will have low sim -> unknown
    q = lm.query(9, 0)
    assert q["unknown"] is True
    assert lm.add_fact(0, 0, 2) is True  # contradiction
    assert lm.query(0, 0)["conflict"] is True

def test_program_compositional_safe():
    pm = ProgramMemory2(D=256, seed=8)
    prog = [("add", 5), ("neg", None)]
    decoded, _ = pm.decode_program(pm.encode_program(prog))
    assert decoded == prog
    assert execute_program(decoded, 10) == -15
    assert execute_program(parse_program_string("add(3) -> mul(2)"), 10) == 26

def test_program_rejects_malicious():
    for bad in ["__import__('os')", "add(x)", "add(999)", "unknown_op(1)"]:
        with pytest.raises(SafeExecutionError):
            parse_program_string(bad)

def test_integration_context_ablation():
    ctxA = -np.ones(256); ctxB = np.ones(256)*2.0
    full = RASMCIMv2(D=256, seed=0, use_ssm=True)
    full.reset_state(); full.write_pair(42, 100, context_event=ctxA)
    full.reset_state(); full.write_pair(42, 200, context_event=ctxB)
    full.reset_state(); full.ssm.step(ctxA, event=True); vA,_,_ = full.query_pair(42)
    full.reset_state(); full.ssm.step(ctxB, event=True); vB,_,_ = full.query_pair(42)
    assert vA == 100 and vB == 200
    # ablation: without SSM both queries return last write
    no = RASMCIMv2(D=256, seed=0, use_ssm=False)
    no.reset_state(); no.write_pair(42, 100, context_event=ctxA)
    no.reset_state(); no.write_pair(42, 200, context_event=ctxB)
    no.reset_state(); no.ssm.step(ctxA, event=True); wA,_,_ = no.query_pair(42)
    assert wA == 200  # wrong for ctxA — proves SSM matters

def test_no_eval_in_v2():
    import ast, pathlib
    for p in pathlib.Path("rasm_cim_v2").glob("*.py"):
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in ("eval", "exec", "compile", "__import__"), f"{p}:{node.lineno}"
