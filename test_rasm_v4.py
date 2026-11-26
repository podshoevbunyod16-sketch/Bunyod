"""Tests for RASM-CIM v0.4. Run: python3 -m pytest test_rasm_v4.py -v"""
import numpy as np
import pytest
from rasm_cim_v4.rns4 import RNSv4, KeyEncoder
from rasm_cim_v4.graph4 import TruthGraph
from rasm_cim_v4.program4 import synthesize_split
from rasm_cim_v4.ca4 import ca_parity
from rasm_cim_v4.archD import ArchD
from rasm_cim_v3.program3 import execute

def test_rns_boundaries():
    rns = RNSv4((3,5,7,11))
    assert rns.decode_unsigned(rns.encode_unsigned(0)) == 0
    assert rns.decode_unsigned(rns.encode_unsigned(rns.M-1)) == rns.M-1
    with pytest.raises(OverflowError):
        rns.encode_unsigned(rns.M)
    with pytest.raises(OverflowError):
        rns.encode_unsigned(-1)
    assert rns.decode_signed(rns.encode_signed(-100)) == -100

def test_rns_parity_actual_addition():
    rns = RNSv4((3,5,7,11))
    assert rns.sum_bits_rns([1,0,1,1]) == 3
    assert rns.sum_bits_rns([1,0,1,1]) % 2 == 1

def test_key_encoder_no_collision():
    rns = RNSv4((3,5,7,11))
    ke = KeyEncoder(rns)
    # exact keys distinct even beyond M
    assert ke.exact_key(42) != ke.exact_key(42 + rns.M)
    assert ke.in_range(42) is True and ke.in_range(99999) is False
    with pytest.raises(OverflowError):
        ke.partial_residues(99999)

def test_truthgraph_states():
    g = TruthGraph(functional_relations={0})
    g.assert_true(0,0,1)
    assert g.query(0,0,1) == "TRUE"
    assert g.query(0,0,2) == "UNKNOWN"
    assert g.query(9,0,1) == "UNKNOWN"
    g.assert_false(0,0,2)
    assert g.query(0,0,2) == "FALSE"
    g.assert_true(0,0,2)  # functional conflict
    assert g.query(0,0) == "CONFLICT"
    g.retract(0,0,2)
    assert g.query(0,0) == "TRUE"
    assert len(g.history) == 4

def test_synthesis_split():
    res = synthesize_split([(0,6),(1,9)], [(2,12)], [(5,21)], max_depth=3)
    assert res["train_acc"] == 1.0 and res["test_acc"] == 1.0
    assert res["candidates"] > 0 and res["program"] is not None

def test_ca_parity():
    assert ca_parity(np.array([1,0,1,1]))[0] == 1
    assert ca_parity(np.array([1,0,1,0]))[0] == 0

def test_archD_save_load_tools():
    d = ArchD(seed=0)
    d.learn_fact(0,0,1); d.learn_fact(1,1,2)
    state = d.save()
    d2 = ArchD(seed=0); d2.load(state)
    assert d2.compose_value(0,0,1) == 2
    assert d.call_tool("add", 20, 22) == 42
    assert d.propagate(0, steps=2) == {0,1,2}

def test_no_eval_v4():
    import ast, pathlib
    for p in pathlib.Path("rasm_cim_v4").glob("*.py"):
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in ("eval","exec","compile","__import__"), f"{p}:{node.lineno}"
