//! Hierarchical memory — port of memory5.py / memory_system.py (v0.6 §5).
//!
//! L1 exact (HashMap), L2 partial (inverted index on key residues),
//! L3 noisy associative (bipolar HV + calibrated tau).
//! Semantic = TruthGraph, episodic = append-only log, procedural = verified programs.
//! Dedup without losing unique info; CONFLICT on distinct values for same key.
//! Confidence: exact = 1.0 by construction (deterministic lookup);
//! noisy = calibrated (sim+1)/2, NOT hardcoded 1.0.

use std::collections::{HashMap, HashSet};
use crate::graph::TruthGraph;
use crate::rns::Rns;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Query result with calibrated confidence.
#[derive(Debug, Clone, PartialEq)]
pub enum QueryResult {
    True { value: i64, confidence: f64 },
    Unknown { confidence: f64 },
    Conflict { values: Vec<i64> },
}

pub struct MemorySystem {
    d: usize,
    tau: f64,
    rng: StdRng,
    episodic: Vec<(u64, i64, i64)>, // (step, key, value)
    semantic: TruthGraph,
    procedural: Vec<(Vec<(String, Option<i64>)>, f64, f64)>,
    exact: HashMap<i64, Vec<i64>>,
    inv: HashMap<(usize, i64), HashSet<i64>>,
    hvs: Vec<Vec<f64>>,
    vals: Vec<i64>,
    rns: Rns,
    dedup_hits: u64,
    step: u64,
}

impl MemorySystem {
    pub fn new(d: usize, tau: f64, seed: u64) -> Self {
        Self {
            d, tau,
            rng: StdRng::seed_from_u64(seed),
            episodic: Vec::new(),
            semantic: TruthGraph::new(vec![0, 1, 2]),
            procedural: Vec::new(),
            exact: HashMap::new(),
            inv: HashMap::new(),
            hvs: Vec::new(),
            vals: Vec::new(),
            rns: Rns::new(vec![3, 5, 7, 11, 13]).unwrap(),
            dedup_hits: 0,
            step: 0,
        }
    }

    /// Write episodic fact. Returns true if duplicate (dedup hit).
    pub fn write_episodic(&mut self, key: i64, value: i64) -> bool {
        if let Some(vals) = self.exact.get(&key) {
            if vals.contains(&value) {
                self.dedup_hits += 1;
                return true;
            }
        }
        self.step += 1;
        self.episodic.push((self.step, key, value));
        self.exact.entry(key).or_default().push(value);
        for (pos, m) in [97i64, 89, 83].iter().enumerate() {
            self.inv.entry((pos, key.rem_euclid(*m))).or_default().insert(key);
        }
        let hv: Vec<f64> = (0..self.d).map(|_| if self.rng.gen_bool(0.5) { 1.0 } else { -1.0 }).collect();
        self.hvs.push(hv);
        self.vals.push(value);
        false
    }

    pub fn query_exact(&self, key: i64) -> QueryResult {
        match self.exact.get(&key) {
            None => QueryResult::Unknown { confidence: 0.0 },
            Some(vals) => {
                let mut distinct: Vec<i64> = vals.clone();
                distinct.sort_unstable();
                distinct.dedup();
                if distinct.len() > 1 {
                    QueryResult::Conflict { values: distinct }
                } else {
                    // 1.0 justified: deterministic HashMap lookup verified by construction
                    QueryResult::True { value: vals[0], confidence: 1.0 }
                }
            }
        }
    }

    /// Partial search: all keys with key % modulus == residue.
    pub fn query_partial(&self, residue: i64, modulus: i64) -> Vec<i64> {
        let mods = [97i64, 89, 83];
        let pos = mods.iter().position(|&m| m == modulus)
            .expect("modulus must be 97/89/83 for L2 index");
        let mut out: Vec<i64> = self.inv.get(&(pos, residue))
            .map(|s| s.iter().copied().collect()).unwrap_or_default();
        out.sort_unstable();
        out
    }

    /// Noisy associative retrieval with calibrated confidence.
    pub fn query_noisy(&self, qhv: &[f64]) -> QueryResult {
        if self.hvs.is_empty() {
            return QueryResult::Unknown { confidence: 0.0 };
        }
        let mut best_idx = 0;
        let mut best_sim = f64::NEG_INFINITY;
        for (i, hv) in self.hvs.iter().enumerate() {
            let sim: f64 = hv.iter().zip(qhv.iter()).map(|(a, b)| a * b).sum::<f64>() / self.d as f64;
            if sim > best_sim { best_sim = sim; best_idx = i; }
        }
        if best_sim < self.tau {
            return QueryResult::Unknown { confidence: best_sim };
        }
        QueryResult::True { value: self.vals[best_idx], confidence: (best_sim + 1.0) / 2.0 }
    }

    /// Calibrate tau on validation (known accept / unknown refuse).
    pub fn calibrate_tau(&mut self, known: &[(Vec<f64>, i64)], unknown: &[Vec<f64>]) -> (f64, f64) {
        let taus = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let mut best = (0.5f64, -1.0f64);
        for tau in taus {
            let k_ok = known.iter().filter(|(hv, v)| {
                matches!(self.query_noisy_tau(hv, tau), QueryResult::True { value, .. } if value == *v)
            }).count();
            let u_ok = unknown.iter().filter(|hv| {
                matches!(self.query_noisy_tau(hv, tau), QueryResult::Unknown { .. })
            }).count();
            let score = (k_ok as f64 / known.len().max(1) as f64
                + u_ok as f64 / unknown.len().max(1) as f64) / 2.0;
            if score > best.1 { best = (tau, score); }
        }
        self.tau = best.0;
        best
    }

    fn query_noisy_tau(&self, qhv: &[f64], tau: f64) -> QueryResult {
        if self.hvs.is_empty() { return QueryResult::Unknown { confidence: 0.0 }; }
        let mut best_idx = 0;
        let mut best_sim = f64::NEG_INFINITY;
        for (i, hv) in self.hvs.iter().enumerate() {
            let sim: f64 = hv.iter().zip(qhv.iter()).map(|(a, b)| a * b).sum::<f64>() / self.d as f64;
            if sim > best_sim { best_sim = sim; best_idx = i; }
        }
        if best_sim < tau { QueryResult::Unknown { confidence: best_sim } }
        else { QueryResult::True { value: self.vals[best_idx], confidence: (best_sim + 1.0) / 2.0 } }
    }

    /// Pure noisy-query kernel for cross-language equivalence tests:
    /// same inputs (hvs/vals/qhv/tau) => same result, independent of RNG.
    pub fn noisy_query(hvs: &[Vec<f64>], vals: &[i64], qhv: &[f64], tau: f64, d: usize) -> QueryResult {
        if hvs.is_empty() { return QueryResult::Unknown { confidence: 0.0 }; }
        let mut best_idx = 0;
        let mut best_sim = f64::NEG_INFINITY;
        for (i, hv) in hvs.iter().enumerate() {
            let sim: f64 = hv.iter().zip(qhv.iter()).map(|(a, b)| a * b).sum::<f64>() / d as f64;
            if sim > best_sim { best_sim = sim; best_idx = i; }
        }
        if best_sim < tau { QueryResult::Unknown { confidence: best_sim } }
        else { QueryResult::True { value: vals[best_idx], confidence: (best_sim + 1.0) / 2.0 } }
    }

    /// RNS compression (exact, raises if out of range — no silent wrap).
    pub fn compress_value(&mut self, x: u64) -> Result<Vec<u32>, crate::RasmError> {
        self.rns.encode_unsigned(x)
    }

    pub fn decompress_value(&mut self, residues: &[u32]) -> u64 {
        self.rns.decode_unsigned(residues)
    }

    /// Exact-restoration proof: all facts retrievable.
    pub fn retention_check(&self, kv: &[(i64, i64)]) -> bool {
        kv.iter().all(|(k, v)| matches!(self.query_exact(*k), QueryResult::True { value, .. } if value == *v))
    }

    pub fn learn_fact(&mut self, s: i64, r: i64, o: i64) { self.semantic.assert_true(s, r, o); }
    pub fn query_fact(&self, s: i64, r: i64, o: Option<i64>) -> String {
        self.semantic.query(s, r, o).to_string()
    }
    /// Procedural memory: store learned programs with train/val accuracy
    /// (port of memory_system.py::store_program).
    pub fn store_program(&mut self, program: Vec<(String, Option<i64>)>, train_acc: f64, val_acc: f64) {
        self.procedural.push((program, train_acc, val_acc));
    }
    pub fn procedural_len(&self) -> usize { self.procedural.len() }
    pub fn best_procedure(&self) -> Option<(Vec<(String, Option<i64>)>, f64, f64)> {
        self.procedural.iter().max_by(|a, b| a.2.partial_cmp(&b.2).unwrap()).cloned()
    }
    pub fn dedup_hits(&self) -> u64 { self.dedup_hits }
    /// Sorted list of all keys (for cross-language state comparison).
    pub fn all_keys(&self) -> Vec<i64> {
        let mut ks: Vec<i64> = self.exact.keys().copied().collect();
        ks.sort_unstable();
        ks
    }
    pub fn episodic_len(&self) -> usize { self.episodic.len() }
    pub fn tau(&self) -> f64 { self.tau }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedup_conflict_compression() {
        let mut m = MemorySystem::new(64, 0.5, 0);
        assert!(!m.write_episodic(1, 100));
        assert!(m.write_episodic(1, 100)); // duplicate
        m.write_episodic(1, 200);
        assert!(matches!(m.query_exact(1), QueryResult::Conflict { .. }));
        let enc = m.compress_value(42).unwrap();
        assert_eq!(m.decompress_value(&enc), 42);
    }

    #[test]
    fn partial_and_retention() {
        let mut m = MemorySystem::new(64, 0.5, 0);
        let kv: Vec<(i64, i64)> = (0..100).map(|i| (i, i * 2)).collect();
        for (k, v) in &kv { m.write_episodic(*k, *v); }
        assert!(m.query_partial(42 % 97, 97).contains(&42));
        assert!(m.retention_check(&kv));
    }
}
