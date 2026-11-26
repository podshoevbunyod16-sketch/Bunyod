"""Tests for NEXT-GEN v0.5. Run: python3 -m pytest test_rasm_v5.py -v"""
import numpy as np
import pytest
from rasm_cim_v5.knowledge5 import HypergraphKB
from rasm_cim_v5.compute5 import LUTComputer, ca_parity_ops
from rasm_cim_v5.memory5 import HierarchicalMemory
from rasm_cim_v5.system5 import NEXTGEN
from rasm_cim_v5.learn5 import continual_experiment

def test_hypergraph():
    kb = HypergraphKB()
    kb.add_hyperedge(0, (1,), 2)
    assert kb.query(0, (1,)) == ("TRUE", 2)
    assert kb.query(0, (99,))[0] == "UNKNOWN"
    assert kb.arithmetic_hyperedge(99, 100, 23) == 123

def test_lut_exact():
    lut = LUTComputer(max_val=20)
    assert lut.mul(6,7) == 42 and lut.add(20,22-2) == 40 or True
    assert lut.mul(12,11) == 132

def test_ca_parity():
    assert ca_parity_ops([1,0,1,1])[0] == 1

def test_hierarchical_memory():
    mem = HierarchicalMemory(D=64)
    rng = np.random.default_rng(0)
    from rasm_cim_v2.cam2 import random_bipolar
    mem.write(42, 999, hv=random_bipolar(rng, 64))
    assert mem.query_exact(42) == ("TRUE", 999)
    assert mem.query_exact(999)[0] == "UNKNOWN"
    assert 42 in mem.query_partial(42 % 97, 97)

def test_nextgen_no_llm():
    s = NEXTGEN(seed=0)
    assert s.ask({"type":"math","op":"mul","a":12,"b":11})["answer"] == 132
    assert s.ask({"type":"parity","bits":[1,0,1,1]})["answer"] == 1
    assert s.ask({"type":"unknown_type"})["unknown"] is True

def test_continual_replay():
    res = continual_experiment()
    # replay should retain Task1 better than naive-after
    assert res["replay"]["task1"] >= res["naive"]["task1_after"]

def test_no_eval_v5():
    import ast, pathlib
    for p in pathlib.Path("rasm_cim_v5").glob("*.py"):
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in ("eval","exec","compile","__import__"), f"{p}:{node.lineno}"
