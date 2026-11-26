//! Learning: faithful Rust port of v0.3 TanhRNNBPTT + v0.6 SelectiveSSM
//! (numpy BPTT + Adam). Hypothesis L1 (selective SSM beats RNN on parity)
//! was REFUTED in v0.6 (ssm 0.460 vs rnn 0.475, chance 0.50) — the Rust
//! port reproduces the protocol and preserves the negative result.
//!
//! Cross-language equivalence: numpy RNG init cannot be bit-reproduced in
//! Rust, so weights are NOT assumed equal. Instead:
//! - `set_weights`/`weights` inject EXACT weights from Python;
//! - with identical weights + data, forward/grads follow the Python
//!   formulas line-by-line (single code path: train_batch -> grads_only,
//!   which the finite-difference tests verify);
//! - with own init (splitmix/Box-Muller), only qualitative agreement
//!   (near chance on parity) is asserted.
//!
//! Mirrored sources: rasm_cim_v3/ssm3.py::TanhRNNBPTT,
//! rasm_cim_v6_v2/learning/selective_ssm.py::SelectiveSSM (input_dim=1).

/// Deterministic PRNG (splitmix64 + Box-Muller) for own-init experiments.
pub struct Rng(pub u64);
impl Rng {
    pub fn new(seed: u64) -> Self { Self(seed.wrapping_add(0x517c_c1b7_2722_0a95)) }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / (1u64 << 53) as f64
    }
    pub fn normal(&mut self, mu: f64, sigma: f64) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        mu + sigma * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[inline]
pub fn sigmoid(z: f64) -> f64 { 1.0 / (1.0 + (-z.clamp(-30.0, 30.0)).exp()) }

/// Adam exactly as in the Python ports:
/// m=.9m+.1g; v=.999v+.001g^2; p -= lr*mhat/(sqrt(vhat)+1e-8).
#[derive(Clone)]
pub struct Adam {
    m: Vec<f64>,
    v: Vec<f64>,
    t: usize,
}
impl Adam {
    pub fn new(n: usize) -> Self { Self { m: vec![0.0; n], v: vec![0.0; n], t: 0 } }
    pub fn step(&mut self, params: &mut [f64], grads: &[f64], lr: f64) {
        assert_eq!(params.len(), grads.len());
        if self.m.len() != params.len() {
            self.m = vec![0.0; params.len()];
            self.v = vec![0.0; params.len()];
        }
        self.t += 1;
        let b1f = 1.0 - 0.9f64.powi(self.t as i32);
        let b2f = 1.0 - 0.999f64.powi(self.t as i32);
        for i in 0..params.len() {
            let g = grads[i];
            self.m[i] = 0.9 * self.m[i] + 0.1 * g;
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g;
            let mhat = self.m[i] / b1f;
            let vhat = self.v[i] / b2f;
            params[i] -= lr * mhat / (vhat.sqrt() + 1e-8);
        }
    }
}

// ---------------------------------------------------------------------------
// TanhRNN + BPTT (ssm3.py::TanhRNNBPTT, input_dim=1)
// Flat param layout: [Wxh(H) | Whh(H*H row-major) | bh(H) | Why(H) | by(1)]
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct TanhRnnBptt {
    pub hidden: usize,
    pub w: Vec<f64>,
    adam: Adam,
}

impl TanhRnnBptt {
    pub fn new(hidden: usize, seed: u64) -> Self {
        let h = hidden;
        let mut rng = Rng::new(seed);
        let s = 1.0 / (h as f64).sqrt();
        let mut w = vec![0.0f64; h + h * h + h + h + 1];
        for i in 0..h { w[i] = rng.normal(0.0, s); }
        for i in h..h + h * h { w[i] = rng.normal(0.0, s) * 0.5; }
        // bh stays 0
        let oy = h + h * h + h;
        for i in 0..h { w[oy + i] = rng.normal(0.0, s); }
        Self { hidden: h, w, adam: Adam::new(h + h * h + h + h + 1) }
    }

    pub fn n_params(&self) -> usize { let h = self.hidden; h + h * h + h + h + 1 }
    pub fn set_weights(&mut self, w: Vec<f64>) {
        assert_eq!(w.len(), self.n_params());
        let n = w.len();
        self.w = w;
        self.adam = Adam::new(n);
    }
    pub fn weights(&self) -> &[f64] { &self.w }

    #[inline]
    fn off(&self) -> (usize, usize, usize, usize) {
        let h = self.hidden;
        (0, h, h + h * h, h + h * h + h) // Wxh, Whh, bh, Why
    }

    /// logits at final step + all hidden states h_0..h_T (h_0 = zeros).
    pub fn forward(&self, seq: &[f64]) -> (f64, Vec<f64>) {
        let h = self.hidden;
        let (ox, ohh, obh, oy) = self.off();
        let by = self.w.len() - 1;
        let mut hs = vec![0.0f64; h * (seq.len() + 1)];
        for (t, &x) in seq.iter().enumerate() {
            for i in 0..h {
                let mut s = self.w[ox + i] * x + self.w[obh + i];
                for j in 0..h { s += self.w[ohh + i * h + j] * hs[h * t + j]; }
                hs[h * (t + 1) + i] = s.tanh();
            }
        }
        let mut logit = self.w[by];
        for i in 0..h { logit += self.w[oy + i] * hs[h * seq.len() + i]; }
        (logit, hs)
    }

    pub fn predict_final(&self, seq: &[f64]) -> u8 {
        if sigmoid(self.forward(seq).0) > 0.5 { 1 } else { 0 }
    }

    pub fn mean_loss(&self, seqs: &[Vec<f64>], labels: &[u8]) -> f64 {
        let mut total = 0.0;
        for (seq, &y) in seqs.iter().zip(labels) {
            let p = sigmoid(self.forward(seq).0);
            let yf = y as f64;
            total += -(yf * (p + 1e-12).ln() + (1.0 - yf) * (1.0 - p + 1e-12).ln());
        }
        total / seqs.len() as f64
    }

    /// Analytic BPTT grads, mean over batch — the exact ssm3.py math,
    /// including its subtlety (no further propagation after t=0).
    pub fn grads_only(&self, seqs: &[Vec<f64>], labels: &[u8]) -> Vec<f64> {
        let h = self.hidden;
        let (ox, ohh, obh, oy) = self.off();
        let n = self.n_params();
        let mut g = vec![0.0f64; n];
        for (seq, &y) in seqs.iter().zip(labels) {
            let t_len = seq.len();
            let (logit, hs) = self.forward(seq);
            let p = sigmoid(logit);
            let dl = p - y as f64;
            for i in 0..h { g[oy + i] += dl * hs[h * t_len + i]; }
            g[n - 1] += dl; // by
            let mut dh = vec![0.0f64; h];
            for i in 0..h {
                dh[i] = dl * self.w[oy + i] * (1.0 - hs[h * t_len + i] * hs[h * t_len + i]);
            }
            for t in (0..t_len).rev() {
                let x = seq[t];
                for i in 0..h {
                    g[ox + i] += dh[i] * x;
                    g[obh + i] += dh[i];
                    for j in 0..h { g[ohh + i * h + j] += dh[i] * hs[h * t + j]; }
                }
                if t > 0 {
                    let mut ndh = vec![0.0f64; h];
                    for i in 0..h {
                        let mut s = 0.0;
                        for j in 0..h { s += self.w[ohh + j * h + i] * dh[j]; }
                        ndh[i] = s * (1.0 - hs[h * t + i] * hs[h * t + i]);
                    }
                    dh = ndh;
                }
            }
        }
        for v in g.iter_mut() { *v /= seqs.len() as f64; }
        g
    }

    /// One Adam step on batch (mean BCE on final logit). Single code path:
    /// grads come from grads_only() (finite-difference-verified).
    pub fn train_batch(&mut self, seqs: &[Vec<f64>], labels: &[u8], lr: f64) -> f64 {
        let g = self.grads_only(seqs, labels);
        let loss = self.mean_loss(seqs, labels);
        self.adam.step(&mut self.w, &g, lr);
        loss
    }

    pub fn accuracy_final(&self, seqs: &[Vec<f64>], labels: &[u8]) -> f64 {
        let ok = seqs.iter().zip(labels).filter(|(s, &y)| self.predict_final(s) == y).count();
        ok as f64 / seqs.len() as f64
    }
}

// ---------------------------------------------------------------------------
// Selective diagonal SSM (selective_ssm.py::SelectiveSSM)
// h_t = a_t*h_{t-1} + b_t*x_t ; logit = Wy·(c_T*h_T) + by
// a_t = sigmoid(Wa*x_t+ba), b_t = Wb*x_t+bb, c_t = Wc*x_t+bc
// Flat layout: [Wa | ba | Wb | bb | Wc | bc | Wy | by], each block D floats.
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Ssm {
    pub d: usize,
    pub w: Vec<f64>,
    adam: Adam,
}

impl Ssm {
    pub fn new(d: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let s = 1.0 / (d as f64).sqrt();
        let mut w = vec![0.0f64; 7 * d + 1];
        for i in 0..7 * d { w[i] = rng.normal(0.0, s); }
        Self { d, w, adam: Adam::new(7 * d + 1) }
    }
    pub fn n_params(&self) -> usize { 7 * self.d + 1 }
    pub fn set_weights(&mut self, w: Vec<f64>) {
        assert_eq!(w.len(), self.n_params());
        let n = w.len();
        self.w = w;
        self.adam = Adam::new(n);
    }
    pub fn weights(&self) -> &[f64] { &self.w }
    #[inline]
    fn blk(&self, k: usize) -> usize { k * self.d }

    pub fn forward(&self, seq: &[f64]) -> f64 {
        let d = self.d;
        let (wa, ba, wb, bb, wc, bc, wy) =
            (self.blk(0), self.blk(1), self.blk(2), self.blk(3), self.blk(4), self.blk(5), self.blk(6));
        let by = self.w.len() - 1;
        let mut h = vec![0.0f64; d];
        let mut logit = 0.0f64;
        for &x in seq {
            let mut new_h = vec![0.0f64; d];
            let mut c = vec![0.0f64; d];
            for i in 0..d {
                let a = sigmoid(self.w[wa + i] * x + self.w[ba + i]);
                let b = self.w[wb + i] * x + self.w[bb + i];
                let ci = self.w[wc + i] * x + self.w[bc + i];
                c[i] = ci;
                new_h[i] = a * h[i] + b * x;
            }
            logit = self.w[by];
            for i in 0..d { logit += self.w[wy + i] * (c[i] * new_h[i]); }
            h = new_h;
        }
        logit
    }

    pub fn predict_final(&self, seq: &[f64]) -> u8 {
        if sigmoid(self.forward(seq)) > 0.5 { 1 } else { 0 }
    }

    pub fn mean_loss(&self, seqs: &[Vec<f64>], labels: &[u8]) -> f64 {
        let mut total = 0.0;
        for (seq, &y) in seqs.iter().zip(labels) {
            let p = sigmoid(self.forward(seq));
            let yf = y as f64;
            total += -(yf * (p + 1e-12).ln() + (1.0 - yf) * (1.0 - p + 1e-12).ln());
        }
        total / seqs.len() as f64
    }

    /// Analytic grads (mean over batch) — the exact selective_ssm.py math:
    /// readout grad at t=T-1 only; per-step b-grads everywhere; no
    /// second-order terms through a_{t-1} (Python does `dh = dh * a`).
    pub fn grads_only(&self, seqs: &[Vec<f64>], labels: &[u8]) -> Vec<f64> {
        let d = self.d;
        let n = self.n_params();
        let (wa, ba, wb, bb, wc, bc, wy) =
            (self.blk(0), self.blk(1), self.blk(2), self.blk(3), self.blk(4), self.blk(5), self.blk(6));
        let mut g = vec![0.0f64; n];
        for (seq, &y) in seqs.iter().zip(labels) {
            let t_len = seq.len();
            // forward caches
            let mut hs: Vec<Vec<f64>> = Vec::with_capacity(t_len);
            let mut a_all: Vec<Vec<f64>> = Vec::with_capacity(t_len);
            let mut c_all: Vec<Vec<f64>> = Vec::with_capacity(t_len);
            let mut h = vec![0.0f64; d];
            let mut logit = 0.0f64;
            for &x in seq {
                let mut new_h = vec![0.0f64; d];
                let mut a = vec![0.0f64; d];
                let mut c = vec![0.0f64; d];
                for i in 0..d {
                    let ai = sigmoid(self.w[wa + i] * x + self.w[ba + i]);
                    let bi = self.w[wb + i] * x + self.w[bb + i];
                    let ci = self.w[wc + i] * x + self.w[bc + i];
                    a[i] = ai; c[i] = ci;
                    new_h[i] = ai * h[i] + bi * x;
                }
                logit = self.w[n - 1];
                for i in 0..d { logit += self.w[wy + i] * (c[i] * new_h[i]); }
                hs.push(std::mem::take(&mut h)); a_all.push(a); c_all.push(c);
                h = new_h;
            }
            let p = sigmoid(logit);
            let dl = p - y as f64;
            let t_last = t_len - 1;
            let cf = &c_all[t_last];
            // Python: hT = c_all[-1] * hs[-1]; grads Wy += outer(dl, hT);
            //        dh = Wy^T dl * c_T; dc = Wy^T dl * hs[-1]  (hs[-1] = h_T final)
            for i in 0..d { g[wy + i] += dl * cf[i] * h[i]; }
            g[n - 1] += dl;
            let mut dh = vec![0.0f64; d];
            let mut dc = vec![0.0f64; d];
            for i in 0..d {
                dh[i] = dl * self.w[wy + i] * cf[i];
                dc[i] = dl * self.w[wy + i] * h[i];
            }
            for t in (0..=t_last).rev() {
                let x = seq[t];
                let h_prev: Vec<f64> = hs[t].clone();
                for i in 0..d {
                    let a = a_all[t][i];
                    g[wb + i] += dh[i] * x * x;
                    g[bb + i] += dh[i] * x;
                    if t == t_last {
                        g[wc + i] += dc[i] * x;
                        g[bc + i] += dc[i];
                    }
                    let da = dh[i] * h_prev[i] * a * (1.0 - a);
                    g[wa + i] += da * x;
                    g[ba + i] += da;
                    dh[i] = dh[i] * a;
                }
            }
        }
        for v in g.iter_mut() { *v /= seqs.len() as f64; }
        g
    }

    pub fn train_batch(&mut self, seqs: &[Vec<f64>], labels: &[u8], lr: f64) -> f64 {
        let g = self.grads_only(seqs, labels);
        let loss = self.mean_loss(seqs, labels);
        self.adam.step(&mut self.w, &g, lr);
        loss
    }

    pub fn accuracy_final(&self, seqs: &[Vec<f64>], labels: &[u8]) -> f64 {
        let ok = seqs.iter().zip(labels).filter(|(s, &y)| self.predict_final(s) == y).count();
        ok as f64 / seqs.len() as f64
    }
}

/// Parity dataset generator (own stream; for exact cross-language tests the
/// harness feeds the same Python-generated data through bindings).
pub fn make_parity(n: usize, length: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<u8>) {
    let mut rng = Rng::new(seed);
    let mut seqs = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    for _ in 0..n {
        let s: Vec<f64> = (0..length).map(|_| if rng.next_u64() & 1 == 1 { 1.0 } else { 0.0 }).collect();
        let l = (s.iter().map(|&v| v as u8).sum::<u8>()) % 2;
        seqs.push(s);
        labels.push(l);
    }
    (seqs, labels)
}

/// v0.6 run_learning() protocol with Rust-side init (qualitative only).
pub fn run_parity_experiment(epochs: usize, lr: f64) -> (f64, f64) {
    let (train_s, train_y) = make_parity(200, 8, 0);
    let (test_s, test_y) = make_parity(200, 8, 1);
    let mut ssm = Ssm::new(8, 0);
    for _ in 0..epochs { ssm.train_batch(&train_s, &train_y, lr); }
    let ssm_acc = ssm.accuracy_final(&test_s, &test_y);
    let mut rnn = TanhRnnBptt::new(16, 0);
    for _ in 0..epochs { rnn.train_batch(&train_s, &train_y, lr); }
    let rnn_acc = rnn.accuracy_final(&test_s, &test_y);
    (ssm_acc, rnn_acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_deterministic() {
        let m1 = TanhRnnBptt::new(8, 42);
        let m2 = TanhRnnBptt::new(8, 42);
        let seq = vec![1.0, 0.0, 1.0, 1.0];
        assert_eq!(m1.forward(&seq).0.to_bits(), m2.forward(&seq).0.to_bits());
        assert_eq!(m1.predict_final(&seq), m2.predict_final(&seq));
    }

    #[test]
    fn gradcheck_rnn_finite_diff() {
        let (seqs, targets) = make_parity(16, 3, 11);
        let m = TanhRnnBptt::new(4, 11);
        let g = m.grads_only(&seqs, &targets);
        let eps = 1e-6;
        let mut max_rel = 0.0f64;
        for idx in [0usize, 1, 5, 20, m.n_params() - 2, m.n_params() - 1] {
            let mut wp = m.w.clone();
            wp[idx] += eps;
            let mut wm = m.w.clone();
            wm[idx] -= eps;
            let mut mp = TanhRnnBptt::new(4, 11);
            mp.set_weights(wp);
            let mut mm = TanhRnnBptt::new(4, 11);
            mm.set_weights(wm);
            let num = (mp.mean_loss(&seqs, &targets) - mm.mean_loss(&seqs, &targets)) / (2.0 * eps);
            let rel = (num - g[idx]).abs() / (num.abs() + 1e-9);
            max_rel = max_rel.max(rel);
        }
        assert!(max_rel < 1e-4, "rnn gradcheck rel={max_rel}");
    }

    #[test]
    fn gradcheck_ssm_finite_diff() {
        let (seqs, targets) = make_parity(16, 3, 11);
        let m = Ssm::new(4, 11);
        let g = m.grads_only(&seqs, &targets);
        let eps = 1e-6;
        let mut max_rel = 0.0f64;
        for idx in [0usize, 1, 3, 4, 6 * m.d, 7 * m.d] {
            let mut wp = m.w.clone();
            wp[idx] += eps;
            let mut wm = m.w.clone();
            wm[idx] -= eps;
            let mut mp = Ssm::new(4, 11);
            mp.set_weights(wp);
            let mut mm = Ssm::new(4, 11);
            mm.set_weights(wm);
            let num = (mp.mean_loss(&seqs, &targets) - mm.mean_loss(&seqs, &targets)) / (2.0 * eps);
            let rel = (num - g[idx]).abs() / (num.abs() + 1e-9);
            max_rel = max_rel.max(rel);
        }
        assert!(max_rel < 1e-3, "ssm gradcheck rel={max_rel}");
    }

    #[test]
    fn ssm_train_loss_matches_eval_and_moves() {
        let (seqs, targets) = make_parity(16, 3, 11);
        let m = Ssm::new(4, 11);
        let l0 = m.mean_loss(&seqs, &targets);
        let mut m2 = Ssm::new(4, 11);
        let loss_reported = m2.train_batch(&seqs, &targets, 0.05);
        assert!((loss_reported - l0).abs() < 1e-12, "train vs eval loss mismatch");
        let l2 = m2.mean_loss(&seqs, &targets);
        assert!((l2 - l0).abs() > 1e-12, "no learning happened");
    }

    #[test]
    fn parity_both_near_chance_negative_result_preserved() {
        // v0.6 L1 REFUTED: ssm 0.460 vs rnn 0.475 at chance 0.50.
        // Rust port must show the same qualitative picture — we do NOT
        // tune until the hypothesis "passes".
        let (ssm_acc, rnn_acc) = run_parity_experiment(30, 0.02);
        assert!((ssm_acc - 0.5).abs() < 0.15, "ssm {ssm_acc} far from chance — protocol drift?");
        assert!((rnn_acc - 0.5).abs() < 0.15, "rnn {rnn_acc} far from chance — protocol drift?");
        assert!(ssm_acc <= rnn_acc + 0.1, "L1 unexpectedly CONFIRMED at Rust init — recheck protocol");
    }
}
