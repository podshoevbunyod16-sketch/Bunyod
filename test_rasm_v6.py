"""Tests for RASM-CIM v0.6. Run: python3 -m pytest test_rasm_v6.py -v"""
import numpy as np
import pytest
from rasm_cim_v6.memory6 import MemoryCentric
from rasm_cim_v6.reasoning6 import DynamicSystem
from rasm_cim_v6.synthesis6 import AlgorithmicLearner
from rasm_cim_v6.profiles import measure_profile

def test_memory_dedup_conflict_compression():
    mem = MemoryCentric(D=64, seed=0)
    assert mem.write_episodic(1, 100) is False
    assert mem.write_episodic(1, 100) is True  # duplicate
    mem.write_episodic(1, 200)
    assert mem.query_exact(1)[0] == "CONFLICT"
    assert mem.decompress_value(mem.compress_value(42)) == 42

def test_dynamic_sparse_saves_ops():
    ds = DynamicSystem(n_cells=64, seed=0)
    bits = np.zeros(64, dtype=int); bits[10]=1
    ds.set_state(bits)
    _, ops_dense = ds.ca_step(rule=90, sparse=False)
    ds.set_state(bits)
    _, ops_sparse = ds.ca_step(rule=90, sparse=True)
    assert ops_sparse < ops_dense

def test_algorithmic_learn_compose_correct():
    learner = AlgorithmicLearner()
    entry = learner.learn([(0,6),(1,9)], [(2,12)], [(5,21)], max_depth=3)
    assert entry["test_acc"] == 1.0
    assert learner.run(entry["program"], 5) == 21

def test_profiles_measure():
    m = measure_profile("mobile", n_facts=100)
    assert m["peak_ram_bytes"] > 0 and m["write_s"] >= 0

def test_no_eval_v6():
    import ast, pathlib
    for p in pathlib.Path("rasm_cim_v6").glob("*.py"):
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in ("eval","exec","compile","__import__"), f"{p}:{node.lineno}"
