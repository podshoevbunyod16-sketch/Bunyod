//! v0.6 run_learning() protocol in Rust: parity len-8, 200/200 split,
//! 30 epochs, lr=0.02. Prints accuracies (negative result preserved).
use rasm_cim_rust::learning::run_parity_experiment;
fn main() {
    let (ssm, rnn) = run_parity_experiment(30, 0.02);
    println!("selective SSM={ssm:.3} RNN={rnn:.3} (chance=0.5) — v0.6 Python: 0.460 / 0.475");
}
