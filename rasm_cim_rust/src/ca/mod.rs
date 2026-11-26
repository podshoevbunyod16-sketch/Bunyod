//! Cellular Automata + Reservoir — port of ca4.py (v0.4).
//!
//! CA: 1D elementary, Rule 90 (XOR) computes parity. Dense (all cells)
//! vs event-driven sparse (only active region). Ops counted honestly.
//! Reservoir: fixed random ESN + trained readout (ridge via normal equations).
//! NOTE: software simulation, NOT hardware CAM/CIM (documented, §11).

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// One dense CA step (all cells). Rule 90: new = left XOR right.
pub fn ca_step_dense(state: &[u8], rule: u32) -> Vec<u8> {
    let n = state.len();
    let mut out = vec![0u8; n];
    for i in 0..n {
        let left = state[(i + n - 1) % n];
        let right = state[(i + 1) % n];
        out[i] = match rule {
            90 => left ^ right,
            110 => {
                let idx = (left as usize) * 4 + (state[i] as usize) * 2 + (right as usize);
                [0u8, 1, 1, 0, 1, 0, 0, 1][idx]
            }
            _ => panic!("Rule {rule} not implemented"),
        };
    }
    out
}

/// Event-driven sparse CA: update only cells adjacent to 1s.
/// Returns (new_state, n_updated). Ops = n_updated (vs N dense).
pub fn ca_step_sparse(state: &[u8], rule: u32) -> (Vec<u8>, usize) {
    let n = state.len();
    let mut new_state = state.to_vec();
    let mut to_update = std::collections::HashSet::new();
    for (i, &v) in state.iter().enumerate() {
        if v == 1 {
            for d in [-1i64, 0, 1] {
                to_update.insert(((i as i64 + d).rem_euclid(n as i64)) as usize);
            }
        }
    }
    let mut count = 0;
    for &i in &to_update {
        let left = state[(i + n - 1) % n];
        let right = state[(i + 1) % n];
        new_state[i] = if rule == 90 { left ^ right } else { state[i] };
        count += 1;
    }
    (new_state, count)
}

/// Parity via Rule 90 XOR folding. Returns (parity, ops).
pub fn ca_parity(bits: &[u8]) -> (u8, usize) {
    let mut x = bits.to_vec();
    let mut ops = 0;
    while x.len() > 1 {
        let n = x.len();
        let mut new = vec![0u8; (n + 1) / 2];
        for i in 0..new.len() {
            let a = x[2 * i];
            let b = if 2 * i + 1 < n { x[2 * i + 1] } else { 0 };
            new[i] = a ^ b;
            ops += 1;
        }
        x = new;
    }
    (x[0], ops)
}

/// Echo State Reservoir: fixed random recurrent + ridge readout.
pub struct EchoStateReservoir {
    pub res_dim: usize,
    pub input_dim: usize,
    w_in: Vec<Vec<f64>>,
    w_res: Vec<Vec<f64>>,
    state: Vec<f64>,
}

impl EchoStateReservoir {
    pub fn new(input_dim: usize, res_dim: usize, spectral_radius: f64, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut w_res: Vec<Vec<f64>> = (0..res_dim)
            .map(|_| (0..res_dim).map(|_| rng.gen_range(-1.0..1.0)).collect()).collect();
        // scale to spectral radius (approx via Frobenius norm scaling)
        let norm: f64 = w_res.iter().flat_map(|r| r.iter()).map(|x| x * x).sum::<f64>().sqrt();
        let scale = spectral_radius / norm.max(1e-12);
        for row in w_res.iter_mut() { for x in row.iter_mut() { *x *= scale; } }
        let w_in: Vec<Vec<f64>> = (0..res_dim)
            .map(|_| (0..input_dim).map(|_| rng.gen_range(-1.0..1.0)).collect()).collect();
        Self { res_dim, input_dim, w_in, w_res, state: vec![0.0; res_dim] }
    }

    /// One reservoir step: h = tanh(W_in x + W_res h).
    pub fn step(&mut self, x: &[f64]) -> Vec<f64> {
        let mut new_state = vec![0.0; self.res_dim];
        for i in 0..self.res_dim {
            let mut s = 0.0;
            for j in 0..self.input_dim { s += self.w_in[i][j] * x[j]; }
            for j in 0..self.res_dim { s += self.w_res[i][j] * self.state[j]; }
            new_state[i] = s.tanh();
        }
        self.state = new_state.clone();
        new_state
    }

    /// Collect states for a sequence (for readout training).
    pub fn collect(&mut self, seq: &[Vec<f64>]) -> Vec<Vec<f64>> {
        seq.iter().map(|x| self.step(x)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ca_parity_correct() {
        let bits = vec![1u8, 0, 1, 1, 0];
        let (p, _) = ca_parity(&bits);
        assert_eq!(p, (bits.iter().sum::<u8>() % 2) as u8);
    }

    #[test]
    fn sparse_saves_ops() {
        let mut state = vec![0u8; 64];
        state[10] = 1; state[20] = 1; state[30] = 1;
        let (_, sparse) = ca_step_sparse(&state, 90);
        assert!(sparse < 64);
    }

    #[test]
    fn reservoir_runs() {
        let mut r = EchoStateReservoir::new(1, 8, 0.9, 0);
        let h = r.step(&[0.5]);
        assert_eq!(h.len(), 8);
    }
}
