"""Tests for RASM-CIM v0.3. Run: python3 -m pytest test_rasm_v3.py -v"""
import numpy as np
import pytest
from rasm_cim_v3.memory3 import RNSv3, MultiBankRNS, FixedCAM, GraphMemory
from rasm_cim_v3.program3 import synthesize, execute, SafeError
from rasm_cim_v3.resonator3 import Resonator3, make_codebooks, bind_factors
from rasm_cim_v3.arch import ArchA, ArchB, ArchC
from rasm_cim_v3.ssm3 import TanhRNNBPTT, make_parity

def test_rns_overflow_raises():
    rns = RNSv3((3,5,7,11))
    with pytest.raises(OverflowError):
        rns.encode_unsigned(2000)
    assert rns.decode_signed(rns.encode_signed(-100)) == -100

def test_multibank():
    mb = MultiBankRNS([(3,5,7),(11,13,17)])
    assert mb.decode(mb.encode(100000)) == 100000

def test_cam_collision_conflict():
    from rasm_cim_v2.cam2 import random_bipolar
    cam = FixedCAM(D=64)
    k = random_bipolar(np.random.default_rng(0), 64)
    cam.write((1,2,3), k, 100); cam.write((1,2,3), k, 200)
    res = cam.query_exact((1,2,3))
    assert res.conflict is True and res.value is None

def test_graph_compose_refusal_contradiction():
    g = GraphMemory()
    g.add_edge(0,0,1); g.add_edge(1,1,2)
    assert g.compose_bfs(0,0,1).value == 2
    assert g.query(9,0).unknown is True
    assert g.add_edge(0,0,99) is True

def test_bfs_synthesis_length4():
    prog = synthesize([(0,9),(1,13),(2,17)], max_depth=4, max_nodes=50000)
    assert prog is not None and len(prog) <= 4
    assert all(execute(prog, x)==y for x,y in [(0,9),(1,13),(2,17)])

def test_resonator_honest_stats():
    books = make_codebooks([10,10], 256, seed=11)
    res = Resonator3(books, seed=0, n_restarts=20)
    pred, conf, fb = res.factorize(bind_factors(books, (3,7)))
    assert pred == (3,7)
    assert res.stats()["calls"] == 1

def test_archB_parity_rns():
    b = ArchB()
    assert b.parity_rns([1,0,1,1]) == 1  # 3 ones -> odd
    assert b.parity_rns([1,0,1,0]) == 0

def test_archA_overflow_safe():
    a = ArchA()
    with pytest.raises(OverflowError):
        a.query_pair(99999)

def test_rnn_bptt_runs():
    seqs, labels = make_parity(20, 4, seed=0)
    m = TanhRNNBPTT(input_dim=1, hidden=8, seed=0)
    loss = m.train_batch(seqs, labels, lr=0.02)
    assert np.isfinite(loss)

def test_no_eval_v3():
    import ast, pathlib
    for p in pathlib.Path("rasm_cim_v3").glob("*.py"):
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in ("eval","exec","compile","__import__"), f"{p}:{node.lineno}"
