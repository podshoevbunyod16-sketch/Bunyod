"""Tests for RASM-CIM v0.6 full (rasm_cim_v6_v2). Run: python3 -m pytest test_rasm_v6_v2.py -v"""
import numpy as np
import pytest
from rasm_cim_v6_v2.architectures.archA_memory import ArchA_MemoryCentric
from rasm_cim_v6_v2.architectures.archB_dynamic import ArchB_Dynamic
from rasm_cim_v6_v2.architectures.archC_algorithmic import ArchC_Algorithmic
from rasm_cim_v6_v2.learning.selective_ssm import SelectiveSSM
from rasm_cim_v6_v2.compute.compute_core import PackedStore
from rasm_cim_v6_v2.mobile.profiles_mobile import measure_profile

def test_archA_dedup_conflict_compression():
    A = ArchA_MemoryCentric(D=64, seed=0)
    assert A.write(1, 100) is False
    assert A.write(1, 100) is True
    A.write(1, 200)
    assert A.query_exact(1)[0] == "CONFLICT"
    assert A.decompress(A.compress(42)) == 42

def test_archB_sparse_saves():
    B = ArchB_Dynamic(n_cells=64, seed=0)
    bits = np.zeros(64, dtype=int); bits[10] = 1
    B.set_state(bits); _, dense = B.ca_step(rule=90, sparse=False)
    B.set_state(bits); _, sparse = B.ca_step(rule=90, sparse=True)
    assert sparse < dense

def test_archC_learn():
    C = ArchC_Algorithmic()
    e = C.learn([(0, 6), (1, 9)], [(2, 12)], [(5, 21)], max_depth=3)
    assert e["test_acc"] == 1.0 and C.run(e["program"], 5) == 21

def test_selective_ssm_runs():
    ssm = SelectiveSSM(D=4, seed=0)
    seqs = [np.array([[0.0], [1.0]]), np.array([[1.0], [0.0]])]
    loss = ssm.train_batch(seqs, [1, 0], lr=0.01)
    assert loss >= 0 and ssm.predict_final(seqs[0]) in (0, 1)

def test_packed_store():
    ps = PackedStore()
    for i in range(100): ps.put(i, i * 2)
    assert ps.get(42) == 84 and ps.bytes_used() == 1600

def test_profile_measure():
    m = measure_profile("mobile", n_facts=50)
    assert m["peak_ram_bytes"] > 0

def test_no_eval_v6v2():
    import ast, pathlib
    for p in pathlib.Path("rasm_cim_v6_v2").rglob("*.py"):
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in ("eval", "exec", "compile", "__import__"), f"{p}:{node.lineno}"
