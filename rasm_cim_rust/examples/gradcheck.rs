//! Finite-difference gradient check for learning models (uses the exact
//! train-path grads_only). Run: cargo run --release --example gradcheck
use rasm_cim_rust::learning::{Ssm, TanhRnnBptt, make_parity};

fn check<G: Fn(f64) -> f64>(_name: &str, loss: G, grads: &Vec<f64>, idxs: &[usize], eps: f64) -> f64 {
    let mut worst = 0.0f64;
    for &i in idxs {
        let num = (loss(eps_at(grads, i, eps)) - 0.0); // placeholder, real loop below
        let _ = num;
    }
    worst
}
fn eps_at(_g: &Vec<f64>, _i: usize, _e: f64) -> f64 { 0.0 }

fn main() {
    let (seqs, targets) = make_parity(16, 3, 11);
    // --- RNN: all params ---
    let m = TanhRnnBptt::new(4, 11);
    let g = m.grads_only(&seqs, &targets);
    let n = m.n_params();
    let mut worst = 0.0f64;
    for idx in 0..n {
        let mut wp = m.w.clone();
        wp[idx] += 1e-6;
        let mut wm = m.w.clone();
        wm[idx] -= 1e-6;
        let mut mp = m.clone(); mp.set_weights(wp);
        let mut mm = m.clone(); mm.set_weights(wm);
        let num = (mp.mean_loss(&seqs, &targets) - mm.mean_loss(&seqs, &targets)) / 2e-6;
        let rel = (num - g[idx]).abs() / (num.abs() + 1e-9);
        if rel > worst { worst = rel; }
    }
    println!("TanhRnnBptt: max_rel_err over ALL {n} params = {worst:.2e}");
    assert!(worst < 1e-4);
    // --- SSM: all params ---
    let s = Ssm::new(4, 11);
    let g2 = s.grads_only(&seqs, &targets);
    let n2 = s.n_params();
    let mut worst2 = 0.0f64;
    for idx in 0..n2 {
        let mut wp = s.w.clone();
        wp[idx] += 1e-6;
        let mut wm = s.w.clone();
        wm[idx] -= 1e-6;
        let mut mp = s.clone(); mp.set_weights(wp);
        let mut mm = s.clone(); mm.set_weights(wm);
        let num = (mp.mean_loss(&seqs, &targets) - mm.mean_loss(&seqs, &targets)) / 2e-6;
        let rel = (num - g2[idx]).abs() / (num.abs() + 1e-9);
        if rel > worst2 { worst2 = rel; }
    }
    println!("Ssm: max_rel_err over ALL {n2} params = {worst2:.2e}");
    assert!(worst2 < 1e-3);
    println!("GRADCHECK OK — full parameter sweep for both models");
}
