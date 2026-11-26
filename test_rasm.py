"""Unit tests for RASM-CIM prototype. Run: python3 -m pytest test_rasm.py -v"""
import numpy as np
from rasm_cim.rns import RNS
from rasm_cim.fpe import IntegerFPE, ResidueFPE
from rasm_cim.cam import AssociativeMemory, random_bipolar, flip_bits
from rasm_cim.state import DiagonalSSM
from rasm_cim.resonator import Resonator, make_codebooks, bind_factors
from rasm_cim.logic import ProgramMemory, generate_code

def test_rns_roundtrip():
    rns = RNS((3,5,7,11))
    for x in [0, 1, 42, 1000, 1154]:
        assert rns.decode(rns.encode(x)) == x

def test_rns_arithmetic():
    rns = RNS((3,5,7,11))
    a, b = rns.encode(100), rns.encode(23)
    assert rns.decode(rns.add(a,b)) == 123
    assert rns.decode(rns.mul(a,b)) == 2300 % rns.M
    assert rns.decode(rns.sub(a,b)) == 77

def test_rns_compare():
    rns = RNS((3,5,7))
    assert rns.compare(rns.encode(10), rns.encode(20)) == -1
    assert rns.compare(rns.encode(20), rns.encode(10)) == 1
    assert rns.compare(rns.encode(15), rns.encode(15)) == 0

def test_fpe_addition():
    fpe = IntegerFPE(D=256, seed=0)
    codebook = {v: fpe.encode(v) for v in range(30)}
    assert fpe.decode(fpe.bind(fpe.encode(7), fpe.encode(8)), codebook) == 15

def test_residue_fpe_periodicity():
    rfpe = ResidueFPE(m=7, D=256, seed=0)
    assert np.allclose(rfpe.encode(3), rfpe.encode(10))

def test_cam_retrieval():
    rng = np.random.default_rng(0)
    cam = AssociativeMemory(D=256)
    keys = [random_bipolar(rng, 256) for _ in range(20)]
    for i, k in enumerate(keys):
        cam.write_hv(k, i)
    for i in range(20):
        val, _ = cam.search_cam_vectorized(keys[i])
        assert val == i

def test_ssm_decay():
    ssm = DiagonalSSM(input_dim=1, state_dim=4, decay=0.5, seed=0)
    ssm.B = np.ones((4,1))
    ssm.step(np.array([1.0]), event=True)
    h0 = float(ssm.h.mean())
    for _ in range(10):
        ssm.step(np.array([0.0]), event=False)
    assert abs(float(ssm.h.mean())/h0 - 0.5**10) < 1e-9

def test_resonator_small():
    books = make_codebooks([10,10], 256, seed=11)
    res = Resonator(books, seed=0, n_restarts=20)
    comp = bind_factors(books, (3,7))
    pred, _ = res.factorize(comp)
    assert pred == (3,7)

def test_program_roundtrip():
    pm = ProgramMemory(D=256, seed=8)
    hv = pm.encode("add", 3)
    (op, arg), _ = pm.decode(hv)
    assert (op, arg) == ("add", 3)
    assert eval(generate_code(op, arg))(10) == 13
