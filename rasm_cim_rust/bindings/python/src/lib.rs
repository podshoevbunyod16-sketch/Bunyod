//! Python bindings (PyO3) for the RASM-CIM v0.7 Rust core.
//! Module name: `rasm_cim_ext`.
//! Python stays the research interface; compute runs in Rust.
//!
//! Equivalence-friendly design: every constructor takes explicit seeds /
//! data, no hidden global state; RNG-based streams are exposed through the
//! same API so tests can inject identical inputs and compare outputs.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use rasm_cim_rust::ca::{ca_parity, ca_step_dense, ca_step_sparse, EchoStateReservoir};
use rasm_cim_rust::compute::PackedStore;
use rasm_cim_rust::graph::TruthGraph;
use rasm_cim_rust::learning;
use rasm_cim_rust::memory::{MemorySystem, QueryResult};
use rasm_cim_rust::rns::Rns;
use rasm_cim_rust::storage;
use rasm_cim_rust::synthesis::{execute, synthesize_split, Op};

fn to_py_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

// ------------------------------ RNS ------------------------------

#[pyclass]
struct RnsPy {
    inner: Rns,
}

#[pymethods]
impl RnsPy {
    #[new]
    fn new(moduli: Vec<u32>) -> PyResult<Self> {
        Ok(Self { inner: Rns::new(moduli).map_err(to_py_err)? })
    }
    fn m(&self) -> u64 { self.inner.m() }
    fn ops(&self) -> u64 { self.inner.ops() }
    fn reset_ops(&mut self) { self.inner.reset_ops() }
    fn encode_unsigned(&mut self, x: u64) -> PyResult<Vec<u32>> {
        self.inner.encode_unsigned(x).map_err(to_py_err)
    }
    fn encode_signed(&mut self, x: i64) -> PyResult<Vec<u32>> {
        self.inner.encode_signed(x).map_err(to_py_err)
    }
    fn decode_unsigned(&mut self, residues: Vec<u32>) -> u64 {
        self.inner.decode_unsigned(&residues)
    }
    fn decode_signed(&mut self, residues: Vec<u32>) -> i64 {
        self.inner.decode_signed(&residues)
    }
    fn add_residues(&mut self, a: Vec<u32>, b: Vec<u32>) -> Vec<u32> {
        self.inner.add_residues(&a, &b)
    }
    fn sub_residues(&mut self, a: Vec<u32>, b: Vec<u32>) -> Vec<u32> {
        self.inner.sub_residues(&a, &b)
    }
    fn mul_residues(&mut self, a: Vec<u32>, b: Vec<u32>) -> Vec<u32> {
        self.inner.mul_residues(&a, &b)
    }
}

// ------------------------------ Memory ------------------------------

#[pyclass]
struct MemoryPy {
    inner: MemorySystem,
}

fn result_to_tuple(r: QueryResult) -> PyObject {
    Python::with_gil(|py| match r {
        QueryResult::True { value, confidence } => (
            "TRUE", Some(value), confidence).into_pyobject(py).unwrap().into_any().unbind(),
        QueryResult::Unknown { confidence } => (
            "UNKNOWN", None::<i64>, confidence).into_pyobject(py).unwrap().into_any().unbind(),
        QueryResult::Conflict { values } => (
            "CONFLICT", Some(values), 0.0f64).into_pyobject(py).unwrap().into_any().unbind(),
    })
}

#[pymethods]
impl MemoryPy {
    #[new]
    fn new(d: usize, tau: f64, seed: u64) -> Self {
        Self { inner: MemorySystem::new(d, tau, seed) }
    }
    /// true if duplicate (dedup hit)
    fn write_episodic(&mut self, key: i64, value: i64) -> bool {
        self.inner.write_episodic(key, value)
    }
    fn query_exact(&self, key: i64) -> PyObject { result_to_tuple(self.inner.query_exact(key)) }
    fn query_partial(&self, residue: i64, modulus: i64) -> PyResult<Vec<i64>> {
        if matches!(modulus, 97 | 89 | 83) { Ok(self.inner.query_partial(residue, modulus)) }
        else { Err(PyValueError::new_err("modulus must be 97/89/83")) }
    }
    fn query_noisy(&self, qhv: Vec<f64>) -> PyObject {
        result_to_tuple(self.inner.query_noisy(&qhv))
    }
    fn calibrate_tau(&mut self, known: Vec<(Vec<f64>, i64)>, unknown: Vec<Vec<f64>>) -> (f64, f64) {
        self.inner.calibrate_tau(&known, &unknown)
    }
    fn compress_value(&mut self, x: u64) -> PyResult<Vec<u32>> {
        self.inner.compress_value(x).map_err(to_py_err)
    }
    fn decompress_value(&mut self, residues: Vec<u32>) -> u64 {
        self.inner.decompress_value(&residues)
    }
    fn retention_check(&self, kv: Vec<(i64, i64)>) -> bool { self.inner.retention_check(&kv) }
    fn learn_fact(&mut self, s: i64, r: i64, o: i64) { self.inner.learn_fact(s, r, o); }
    fn store_program(&mut self, program: Vec<(String, Option<i64>)>, train_acc: f64, val_acc: f64) {
        self.inner.store_program(program, train_acc, val_acc);
    }
    fn procedural_len(&self) -> usize { self.inner.procedural_len() }
    fn query_fact(&self, s: i64, r: i64, o: Option<i64>) -> String { self.inner.query_fact(s, r, o) }
    fn dedup_hits(&self) -> u64 { self.inner.dedup_hits() }
    fn episodic_len(&self) -> usize { self.inner.episodic_len() }
    fn tau(&self) -> f64 { self.inner.tau() }
    /// Deterministic snapshot of all keys (sorted) for cross-language compare
    fn all_keys(&self) -> Vec<i64> { self.inner.all_keys() }
    /// Pure noisy-retrieval kernel: inject exact hvs/vals for equivalence.
    #[staticmethod]
    fn noisy_query(hvs: Vec<Vec<f64>>, vals: Vec<i64>, qhv: Vec<f64>, tau: f64, d: usize) -> PyObject {
        result_to_tuple(MemorySystem::noisy_query(&hvs, &vals, &qhv, tau, d))
    }
}

// ------------------------------ PackedStore ------------------------------

#[pyclass]
struct PackedStorePy {
    inner: PackedStore,
}

#[pymethods]
impl PackedStorePy {
    #[new]
    fn new() -> Self { Self { inner: PackedStore::new() } }
    fn put(&mut self, key: i64, value: i64) { self.inner.put(key, value); }
    fn get(&self, key: i64) -> Option<i64> { self.inner.get(key) }
    fn len(&self) -> usize { self.inner.len() }
    fn bytes_used(&self) -> usize { self.inner.bytes_used() }
    /// Little-endian "<qq" record bytes (bit-identical to Python struct.pack)
    fn to_bytes(&self) -> Vec<u8> { self.inner.to_bytes() }
    #[staticmethod]
    fn from_bytes(data: Vec<u8>) -> PyResult<Self> {
        PackedStore::from_bytes(&data).map(|inner| Self { inner }).map_err(to_py_err)
    }
    fn batch_put(&mut self, pairs: Vec<(i64, i64)>, parallel: bool) {
        if parallel {
            let r = std::thread::scope(|_| self.inner.batch_put_parallel(&pairs));
            let _ = r;
        } else {
            for (k, v) in pairs { self.inner.put(k, v); }
        }
    }
    fn batch_get(&self, keys: Vec<i64>, parallel: bool) -> Vec<Option<i64>> {
        if parallel { self.inner.batch_get_parallel(&keys) }
        else { keys.iter().map(|k| self.inner.get(*k)).collect() }
    }
}

// ------------------------------ free functions ------------------------------

#[pyfunction]
fn ca_dense_step(state: Vec<u8>, rule: u32) -> Vec<u8> { ca_step_dense(&state, rule) }

#[pyfunction]
fn ca_sparse_step(state: Vec<u8>, rule: u32) -> PyResult<(Vec<u8>, usize)> {
    Ok(ca_step_sparse(&state, rule))
}

#[pyfunction]
fn ca_parity_fn(bits: Vec<u8>) -> (u8, usize) { ca_parity(&bits) }

#[pyfunction]
fn synth(train: Vec<(i64, i64)>, val: Vec<(i64, i64)>, test: Vec<(i64, i64)>, max_depth: usize, max_nodes: usize)
    -> PyResult<(Option<Vec<(String, Option<i64>)>>, f64, f64, f64, usize, f64)> {
    let r = synthesize_split(&train, &val, &test, max_depth, max_nodes);
    let prog = r.program.map(|p| p.iter().map(|op| match op {
        Op::Arg(name, arg) => (name.to_string(), Some(*arg)),
        Op::NoArg(name) => (name.to_string(), None),
    }).collect());
    Ok((prog, r.train_acc, r.val_acc, r.test_acc, r.candidates, r.time_ms))
}

#[pyfunction]
fn synth_execute(program: Vec<(String, Option<i64>)>, x: i64) -> PyResult<i64> {
    let mut ops: Vec<Op> = Vec::new();
    for (name, arg) in program {
        ops.push(match arg {
            Some(a) => Op::arg_from(&name, a).ok_or_else(|| PyValueError::new_err(format!("bad op {name}")))?,
            None => Op::noarg_from(&name).ok_or_else(|| PyValueError::new_err(format!("bad op {name}")))?,
        });
    }
    Ok(execute(&ops, x))
}

#[pyfunction]
fn serialize_facts(facts: Vec<(i64, i64)>) -> Vec<u8> { storage::serialize_facts(&facts) }

#[pyfunction]
fn deserialize_facts(data: Vec<u8>) -> PyResult<Vec<(i64, i64)>> {
    storage::deserialize_facts(&data).map_err(to_py_err)
}

#[pyfunction]
fn graph_new(functional: Vec<i64>) -> TruthGraphPy { TruthGraphPy { inner: TruthGraph::new(functional) } }

#[pyclass]
struct TruthGraphPy { inner: TruthGraph }

#[pymethods]
impl TruthGraphPy {
    fn assert_true(&mut self, s: i64, r: i64, o: i64) { self.inner.assert_true(s, r, o); }
    fn assert_false(&mut self, s: i64, r: i64, o: i64) { self.inner.assert_false(s, r, o); }
    fn retract(&mut self, s: i64, r: i64, o: i64) { self.inner.retract(s, r, o); }
    fn query(&self, s: i64, r: i64, o: Option<i64>) -> String {
        self.inner.query(s, r, o).to_string()
    }
    fn history_len(&self) -> usize { self.inner.history_len() }
}

// ------------------------------ learning ------------------------------

#[pyfunction]
fn ssm_forward(weights: Vec<f64>, d: usize, seq: Vec<f64>) -> PyResult<f64> {
    let mut m = learning::Ssm::new(d, 0);
    m.set_weights(weights);
    Ok(m.forward(&seq))
}

#[pyfunction]
fn ssm_grads(weights: Vec<f64>, d: usize, seqs: Vec<Vec<f64>>, labels: Vec<u8>) -> PyResult<Vec<f64>> {
    let mut m = learning::Ssm::new(d, 0);
    m.set_weights(weights);
    Ok(m.grads_only(&seqs, &labels))
}

#[pyfunction]
fn rnn_forward(weights: Vec<f64>, hidden: usize, seq: Vec<f64>) -> PyResult<f64> {
    let mut m = learning::TanhRnnBptt::new(hidden, 0);
    m.set_weights(weights);
    Ok(m.forward(&seq).0)
}

#[pyfunction]
fn rnn_grads(weights: Vec<f64>, hidden: usize, seqs: Vec<Vec<f64>>, labels: Vec<u8>) -> PyResult<Vec<f64>> {
    let mut m = learning::TanhRnnBptt::new(hidden, 0);
    m.set_weights(weights);
    Ok(m.grads_only(&seqs, &labels))
}

#[pyfunction]
fn run_parity_experiment(epochs: usize, lr: f64) -> (f64, f64) {
    learning::run_parity_experiment(epochs, lr)
}

// ------------------------------ reservoir ------------------------------

#[pyclass]
struct ReservoirPy { inner: EchoStateReservoir }

#[pymethods]
impl ReservoirPy {
    #[new]
    fn new(input_dim: usize, res_dim: usize, radius: f64, seed: u64) -> Self {
        Self { inner: EchoStateReservoir::new(input_dim, res_dim, radius, seed) }
    }
    fn step(&mut self, x: Vec<f64>) -> PyResult<Vec<f64>> {
        if x.len() != self.inner.input_dim { return Err(PyValueError::new_err("input dim mismatch")); }
        Ok(self.inner.step(&x))
    }
    fn collect(&mut self, seq: Vec<Vec<f64>>) -> Vec<Vec<f64>> { self.inner.collect(&seq) }
}

// ------------------------------ mobile/profiles ------------------------------

#[pyfunction]
fn profile_max_threads(profile: &str) -> PyResult<usize> {
    use rasm_cim_rust::profiles::Profile;
    let p = match profile {
        "mobile" => Profile::Mobile,
        "desktop" => Profile::Desktop,
        "server" => Profile::Server,
        _ => return Err(PyValueError::new_err("profile: mobile|desktop|server")),
    };
    Ok(p.settings().max_threads)
}

#[pymodule]
fn rasm_cim_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RnsPy>()?;
    m.add_class::<MemoryPy>()?;
    m.add_class::<PackedStorePy>()?;
    m.add_class::<TruthGraphPy>()?;
    m.add_class::<ReservoirPy>()?;
    m.add_function(wrap_pyfunction!(ca_dense_step, m)?)?;
    m.add_function(wrap_pyfunction!(ca_sparse_step, m)?)?;
    m.add_function(wrap_pyfunction!(ca_parity_fn, m)?)?;
    m.add_function(wrap_pyfunction!(synth, m)?)?;
    m.add_function(wrap_pyfunction!(synth_execute, m)?)?;
    m.add_function(wrap_pyfunction!(serialize_facts, m)?)?;
    m.add_function(wrap_pyfunction!(deserialize_facts, m)?)?;
    m.add_function(wrap_pyfunction!(graph_new, m)?)?;
    m.add_function(wrap_pyfunction!(ssm_forward, m)?)?;
    m.add_function(wrap_pyfunction!(ssm_grads, m)?)?;
    m.add_function(wrap_pyfunction!(rnn_forward, m)?)?;
    m.add_function(wrap_pyfunction!(rnn_grads, m)?)?;
    m.add_function(wrap_pyfunction!(run_parity_experiment, m)?)?;
    m.add_function(wrap_pyfunction!(profile_max_threads, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
