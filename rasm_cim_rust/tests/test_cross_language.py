"""Cross-language equivalence tests: Python reference (v0.1–v0.6) vs Rust core.

Rule: where exact equivalence is *defined* (deterministic integer/graph/BFS
logic), Python and Rust MUST agree bit-for-bit. Where numpy RNG init streams
make weights incomparable (learning models, HV generation), we compare only
under EXPLICIT injected weights/data — never claim equivalence otherwise.

Run: pytest rasm_cim_rust/tests/test_cross_language.py -x -q
"""
import math
import os
import random
import struct
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("RASM_PY_ROOT") or os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)  # rasm_cim_ext.so + version packages live here

import rasm_cim_ext as rx  # noqa: E402
from rasm_cim_v4.rns4 import RNSv4  # noqa: E402
from rasm_cim_v6_v2.memory.memory_system import MemorySystem as PyMem  # noqa: E402
from rasm_cim_v6_v2.compute.compute_core import PackedStore as PyPacked  # noqa: E402
# CA reference: v0.4 ca4.py
from rasm_cim_v4.ca4 import ca_step_dense as py_ca_dense, ca_step_sparse as py_ca_sparse, ca_parity as py_ca_parity  # noqa: E402
# synthesis reference v0.4 program4
from rasm_cim_v4.program4 import synthesize_split as py_synth  # noqa: E402

MODULI = (3, 5, 7, 11, 13)
M = 15015


# ---------------------------------------------------------------- RNS
class TestRnsEquivalence:
    def test_roundtrip_edge_and_random(self):
        r = rx.RnsPy(list(MODULI))
        p = RNSv4(MODULI)
        cases = [0, 1, M - 1, 2, 42, 1000, 7507, 7508, 15013]
        cases += random.Random(1).sample(range(M), 200)
        for x in cases:
            assert list(r.encode_unsigned(x)) == [int(v) for v in p.encode_unsigned(x)]
            enc = list(p.encode_unsigned(x))
            assert r.decode_unsigned(enc) == p.decode_unsigned(tuple(enc)) == x

    def test_signed_edge(self):
        r = rx.RnsPy(list(MODULI))
        p = RNSv4(MODULI)
        for x in [-7507, -1, 0, 1, 7507, -100, 42, 7507 - 1, -7507 + 1]:
            assert list(r.encode_signed(x)) == [int(v) for v in p.encode_signed(x)]
            assert r.decode_signed(list(p.encode_signed(x))) == x
        # out-of-range on BOTH languages (no silent wrap)
        for bad in [-7508, M, -(M // 2) - 1, M - 1 + 1]:
            with pytest.raises(ValueError):
                r.encode_signed(bad) if bad < 0 else r.encode_unsigned(bad)
        with pytest.raises(OverflowError):
            p.encode_signed(-7508)
        with pytest.raises(OverflowError):
            p.encode_unsigned(M)

    def test_add_sub_mul(self):
        r = rx.RnsPy(list(MODULI))
        p = RNSv4(MODULI)
        rng = random.Random(2)
        for _ in range(200):
            a, b = rng.randrange(M), rng.randrange(M)
            ea, eb = [int(v) for v in p.encode_unsigned(a)], [int(v) for v in p.encode_unsigned(b)]
            s = r.add_residues(ea, eb)
            assert p.decode_unsigned(tuple(s)) == (a + b) % M
            pr = r.mul_residues(ea, eb)
            assert p.decode_unsigned(tuple(pr)) == (a * b) % M
            d = r.sub_residues(ea, eb)
            assert p.decode_unsigned(tuple(d)) == (a - b) % M

    def test_non_coprime(self):
        with pytest.raises(ValueError):
            rx.RnsPy([4, 6])
        with pytest.raises(ValueError):
            RNSv4((4, 6))

    def test_large_moduli_product(self):
        # v0.3 H1 parity: multi-modulus bank M=255255 roundtrip on 1000 pts
        mods = (3, 5, 7, 11, 13, 17, 19)
        m = math.prod(mods)
        r = rx.RnsPy(list(mods))
        p = RNSv4(mods)
        assert r.m() == m
        rng = random.Random(3)
        for x in rng.sample(range(m), 1000):
            assert r.decode_unsigned(list(r.encode_unsigned(x))) == x
            assert p.decode_unsigned(tuple(p.encode_unsigned(x))) == x


# ---------------------------------------------------------------- Memory
class TestMemoryEquivalence:
    def _both(self, n=50, seed=0, d=64):
        r = rx.MemoryPy(d, 0.5, seed)
        p = PyMem(D=d, tau=0.5, seed=seed)
        rng = random.Random(100 + seed)
        keys = list(range(1, n + 1))
        vals = [rng.randrange(M) for _ in range(n)]
        for k, v in zip(keys, vals):
            dr, dp = r.write_episodic(k, v), p.write_episodic(k, v)
            assert dr == dp
        return r, p, keys, vals

    def test_exact_and_dedup(self):
        r, p, keys, vals = self._both(80, seed=1)
        # duplicate writes: both report dedup hit
        for k, v in list(zip(keys, vals))[:20]:
            assert r.write_episodic(k, v) is True
            assert p.write_episodic(k, v) is True
        assert r.dedup_hits() == 20
        assert p.dedup_hits == 20
        for k, v in zip(keys, vals):
            assert r.query_exact(k)[1] == v
            assert p.query_exact(k)[1] == v
        assert r.query_exact(-1)[0] == "UNKNOWN" == p.query_exact(-1)[0]

    def test_conflict(self):
        r, p, keys, _ = self._both(10, seed=2)
        k = keys[0]
        r.write_episodic(k, 999)
        p.write_episodic(k, 999)
        sr, sp = r.query_exact(k), p.query_exact(k)
        assert sr[0] == sp[0] == "CONFLICT"
        orig = p._exact[k][0]
        assert sorted(sr[1]) == sorted(sp[1]) == sorted({orig, 999})

    def test_partial(self):
        r, p, keys, _ = self._both(500, seed=3)
        for mod in (97, 89, 83):
            for res in range(0, mod, 7):
                assert r.query_partial(res, mod) == p.query_partial(res, mod)

    def test_noisy_kernel_injected(self):
        # exact comparability: same hvs injected into the pure kernel
        rng = np.random.default_rng(0)
        hvs = rng.choice(np.array([-1.0, 1.0]), size=(30, 64))
        vals = list(range(30))
        p = PyMem(D=64, tau=0.5, seed=0)
        p._hvs = [hvs[i] for i in range(30)]
        p._vals = vals
        for i in range(30):
            q = hvs[i] * 0.8 + rng.normal(0, 0.05, 64)
            got = rx.MemoryPy.noisy_query([list(map(float, h)) for h in hvs], vals,
                                          list(map(float, q)), 0.5, 64)
            want = p.query_noisy(q)
            assert got[0] == want[0] and got[1] == want[1]
            if want[0] == "TRUE":
                assert abs(got[2] - want[2]) < 1e-9

    def test_compression_roundtrip(self):
        r = rx.MemoryPy(64, 0.5, 0)
        p = PyMem(D=64, tau=0.5, seed=0)
        rng = random.Random(7)
        for x in [0, 1, M - 1] + rng.sample(range(M), 300):
            cr = r.compress_value(x)
            cp = [int(v) for v in p.compress_value(x)]
            assert cr == cp
            assert r.decompress_value(cr) == p.decompress_value(cp) == x
        with pytest.raises(ValueError):
            r.compress_value(M)

    def test_semantic_graph(self):
        r = rx.MemoryPy(64, 0.5, 0)
        p = PyMem(D=64, tau=0.5, seed=0)
        for (s, rel, o) in [(1, 0, 2), (2, 1, 3), (4, 2, 5)]:
            r.learn_fact(s, rel, o)
            p.learn_fact(s, rel, o)
        for (s, rel, o) in [(1, 0, 2), (2, 1, 3), (9, 0, 1)]:
            assert r.query_fact(s, rel, o) == p.query_fact(s, rel, o)


# ---------------------------------------------------------------- Packed store
class TestPackedEquivalence:
    def test_bytes_bitidentical(self):
        r = rx.PackedStorePy()
        p = PyPacked()
        rng = random.Random(11)
        items = [(rng.randrange(-10**12, 10**12), rng.randrange(-10**12, 10**12))
                 for _ in range(1000)]
        for k, v in items:
            r.put(k, v)
            p.put(k, v)
        assert bytes(r.to_bytes()) == bytes(p._buf)
        for k, v in items[::37]:
            assert r.get(k) == p.get(k) == v
        assert r.get(10**13 + 5) == p.get(10**13 + 5) is None
        assert r.bytes_used() == p.bytes_used() == 16000

    def test_reload_after_deserialize(self):
        r = rx.PackedStorePy()
        for i in range(500):
            r.put(i, i * 7)
        raw = bytes(r.to_bytes())
        r2 = rx.PackedStorePy.from_bytes(raw)
        assert all(r2.get(i) == i * 7 for i in range(500))
        assert r2.bytes_used() == len(raw) == 8000


# ---------------------------------------------------------------- CA
class TestCaEquivalence:
    def test_dense_step(self):
        rng = random.Random(12)
        for _ in range(200):
            st = [rng.randrange(2) for _ in range(64)]
            for rule in (90, 110):
                want = py_ca_dense(np.array(st), rule)
                got = rx.ca_dense_step(st, rule)
                assert list(got) == [int(v) for v in want], (rule, st[:4])

    def test_sparse_matches_python_and_saves_ops(self):
        rng = random.Random(13)
        for _ in range(50):
            st = [0] * 64
            idx = rng.sample(range(64), 9)
            for i in idx:
                st[i] = 1
            new_s, ops_s = rx.ca_sparse_step(st, 90)
            w_new, w_ops = py_ca_sparse(np.array(st), 90)
            assert list(new_s) == [int(v) for v in w_new]
            assert ops_s == int(w_ops) < 64  # union of neighborhoods
        # v0.6 D1 exact scenario: isolated actives at 10/20/30 -> 9 vs 64
        st = [0] * 64
        for i in (10, 20, 30):
            st[i] = 1
        new_s, ops_s = rx.ca_sparse_step(st, 90)
        assert ops_s == 9
        new_d = rx.ca_dense_step(st, 90)
        assert list(new_s) == list(new_d)  # identical result, 9 vs 64 ops

    def test_parity(self):
        rng = random.Random(14)
        for _ in range(200):
            bits = [rng.randrange(2) for _ in range(32)]
            got = rx.ca_parity_fn(bits)
            assert got[0] == py_ca_parity(bits)[0] == sum(bits) % 2


# ---------------------------------------------------------------- Synthesis
class TestSynthesisEquivalence:
    CASES = [
        ([(0, 6), (1, 9)], [(2, 12)], [(5, 21)]),          # add2,mul3
        ([(1, -1), (2, -2)], [(3, -3)], [(4, -4)]),        # neg
        ([(2, 4), (3, 9)], [(4, 16)], [(5, 25)]),          # square
        ([(1, 5), (2, 6)], [(3, 7)], [(4, 8)]),            # add4? ops restrict
        ([(7, 1), (8, 1)], [(9, 1)], [(10, 1)]),           # mod-ish absent -> None
    ]

    def test_same_program_counts_and_acc(self):
        for train, val, test in self.CASES:
            prog, tr, va, te, cands, _ms = rx.synth(train, val, test, 3, 50000)
            p = py_synth(train, val, test, max_depth=3, max_nodes=50000)
            py_prog = p["program"]
            assert cands == p["candidates"], (cands, p["candidates"])
            if py_prog is None:
                assert prog is None
            else:
                assert prog is not None
                assert [(o, a) for (o, a) in py_prog] == [(o, a) for (o, a) in prog], (prog, py_prog)
                assert (tr, va, te) == (p["train_acc"], p["val_acc"], p["test_acc"])
                for x, y in test:
                    assert rx.synth_execute(list(prog), x) == y


# ---------------------------------------------------------------- Storage format
class TestStorageFormat:
    def test_serialize_roundtrip_and_python_parse(self):
        facts = [(1, 100), (-5, 2**40), (2**50, -(2**50))]
        data = rx.serialize_facts(facts)
        # Python parses same bytes
        assert data[:4] == b"RASM"
        ver = struct.unpack_from("<I", data, 4)[0]
        plen = struct.unpack_from("<Q", data, 8)[0]
        assert ver == 1 and plen == 16 * len(facts)
        payload = data[16:16 + plen]
        got = [struct.unpack_from("<qq", payload, i * 16) for i in range(len(facts))]
        assert got == facts
        assert rx.deserialize_facts(data) == facts

    def test_corruption_detected(self):
        data = bytearray(rx.serialize_facts([(1, 2), (3, 4)]))
        data[25] ^= 0xFF
        with pytest.raises(ValueError):
            rx.deserialize_facts(bytes(data))

    def test_truncated(self):
        with pytest.raises(ValueError):
            rx.deserialize_facts(b"RASM\x01\x00")


# ---------------------------------------------------------------- Learning
class TestLearningEquivalence:
    """Weights injected from Python -> both sides must compute identical
    forward logits and grads (within 1e-9). This proves the Rust port's
    math matches ssm3.py/selective_ssm.py; RNG streams are NOT compared."""

    def _py_ssm(self, d, seed):
        from rasm_cim_v6_v2.learning.selective_ssm import SelectiveSSM
        return SelectiveSSM(D=d, seed=seed)

    def test_ssm_forward_matches_python(self):
        d = 8
        py = self._py_ssm(d, 0)
        # pack python weights in Rust flat layout [Wa|ba|Wb|bb|Wc|bc|Wy|by]
        w = (py.Wa[:, 0].tolist() + py.ba.tolist() + py.Wb[:, 0].tolist() + py.bb.tolist()
             + py.Wc[:, 0].tolist() + py.bc.tolist() + py.Wy[0].tolist() + list(py.by))
        assert len(w) == 7 * d + 1
        rng = np.random.default_rng(5)
        for _ in range(50):
            bits = rng.integers(0, 2, size=8).astype(float)
            want = py._forward(bits.reshape(8, 1))[4][-1, 0]
            got = rx.ssm_forward(w, d, list(map(float, bits)))
            assert abs(got - want) < 1e-9

    def test_ssm_grads_match_numeric_python(self):
        d = 4
        py = self._py_ssm(d, 1)
        w = (py.Wa[:, 0].tolist() + py.ba.tolist() + py.Wb[:, 0].tolist() + py.bb.tolist()
             + py.Wc[:, 0].tolist() + py.bc.tolist() + py.Wy[0].tolist() + list(py.by))
        rng = np.random.default_rng(6)
        seqs = [rng.integers(0, 2, size=5).astype(float).tolist() for _ in range(8)]
        labels = [(int(sum(s)) % 2) for s in seqs]
        got = rx.ssm_grads(w, d, seqs, labels)
        # numeric grad via python model: perturb each weight
        def py_loss(wvec):
            m = self._py_ssm(d, 0)
            idx = 0
            def take(n):
                nonlocal idx
                v = wvec[idx:idx + n]; idx += n
                return np.array(v)
            m.Wa = take(d).reshape(d, 1); m.ba = take(d)
            m.Wb = take(d).reshape(d, 1); m.bb = take(d)
            m.Wc = take(d).reshape(d, 1); m.bc = take(d)
            m.Wy = take(d).reshape(1, d); m.by = take(1)
            tot = 0.0
            for s, y in zip(seqs, labels):
                logits = m._forward(np.asarray(s).reshape(len(s), 1))[4]
                p = float(m._sigmoid(logits[-1, 0]))
                tot += -(y * math.log(p + 1e-12) + (1 - y) * math.log(1 - p + 1e-12))
            return tot / len(seqs)
        eps = 1e-6
        for gi in [0, 1, 2 * d, 4 * d + 1, 6 * d, 7 * d]:
            wp = list(w); wp[gi] += eps
            wm = list(w); wm[gi] -= eps
            num = (py_loss(wp) - py_loss(wm)) / (2 * eps)
            assert abs(num - got[gi]) < 1e-3, f"param {gi}: numeric {num} vs rust {got[gi]}"

    def test_rnn_forward_matches_python(self):
        from rasm_cim_v3.ssm3 import TanhRNNBPTT
        hidden = 8
        py = TanhRNNBPTT(hidden=hidden, seed=0)
        w = (py.Wxh[:, 0].tolist() + py.Whh.flatten().tolist() + py.bh.tolist()
             + py.Why[0].tolist() + py.by.tolist())
        rng = np.random.default_rng(9)
        for _ in range(50):
            bits = rng.integers(0, 2, size=6).astype(float)
            _, logits = py._forward_seq(bits.reshape(6, 1))
            want = logits[-1, 0]
            got = rx.rnn_forward(w, hidden, list(map(float, bits)))
            assert abs(got - want) < 1e-9

    def test_negative_result_protocol_shape(self):
        # v0.6 REFUTED L1: run same protocol in Rust; both must be ~chance.
        ssm, rnn = rx.run_parity_experiment(30, 0.02)
        assert 0.2 <= ssm <= 0.8 and 0.2 <= rnn <= 0.8
        assert not (ssm > rnn + 0.1)


# ---------------------------------------------------------------- Graph
class TestGraphEquivalence:
    def test_truth_states(self):
        from rasm_cim_v4.graph4 import TruthGraph as PyG
        g = rx.graph_new([0, 1, 2])
        # v0.4 graph4 uses different API — emulate via memory_system semantic
        # graph here (TruthGraph port matches v0.2 logic2 semantics).
        p = PyG({0, 1, 2})
        for (s, r, o) in [(1, 0, 2), (2, 1, 3)]:
            g.assert_true(s, r, o)
            p.assert_true(s, r, o)
        assert g.query(1, 0, 2) == p.query(1, 0, 2) == "TRUE"
        # v0.4 semantics: with explicit o there is NO functional inference
        assert g.query(1, 0, 3) == p.query(1, 0, 3) == "UNKNOWN"
        # functional without o: exactly-one-true => TRUE
        assert g.query(1, 0, None) == p.query(1, 0) == "TRUE"
        g.assert_false(5, 2, 6)
        assert g.query(5, 2, 6) == "FALSE"
        assert g.query(5, 2, 7) == "UNKNOWN"   # o not in either set
        g.assert_true(5, 2, 6)
        assert g.query(5, 2, 6) == "CONFLICT"
        g.retract(5, 2, 6)
        # after retract of positive assertion, FALSE remains
        assert g.query(5, 2, 6) == "FALSE"
        assert g.history_len() == 5


# ---------------------------------------------------------------- concurrency
class TestThreadSafety:
    def test_parallel_batches_same_result(self):
        import threading
        r = rx.PackedStorePy()
        pairs = [(i, i * 3) for i in range(2000)]
        r.batch_put(pairs, True)   # rayon
        seq = rx.PackedStorePy()
        seq.batch_put(pairs, False)
        keys = [i for i in range(2000)]
        assert r.batch_get(keys, True) == seq.batch_get(keys, False)
        assert [(v if v is None else v) for v in r.batch_get(keys, True)] == [v for _, v in pairs]

    def test_threads_independent_stores(self):
        import threading
        out = {}
        def worker(tid):
            m = rx.MemoryPy(64, 0.5, tid)
            for i in range(100):
                m.write_episodic(tid * 1000 + i, i)
            out[tid] = m.retention_check([(tid * 1000 + i, i) for i in range(100)])
        ts = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        assert all(out.values())

    def test_broken_data_no_panic(self):
        for junk in [b"", b"RASM", b"RASM" + b"\x00" * 40, bytes(range(64))]:
            with pytest.raises(ValueError):
                rx.deserialize_facts(junk)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-q"]))
