//! BFS program synthesis — port of program3.py / program4.py (v0.3/v0.4).
//!
//! Synthesize on TRAIN only, select on VAL, report on TEST (no peeking).
//! BFS is SEARCH, not "discovery" (documented). Counts candidates/ops/time.
//! Safe interpreter (no eval). Ops: add/sub/mul/max/min (arg), neg/square/abs.

use std::collections::{HashSet, VecDeque};

pub const OPS_ARG: [&str; 5] = ["add", "sub", "mul", "max", "min"];
pub const OPS_NOARG: [&str; 3] = ["neg", "square", "abs"];
pub const ARG_VALUES: [i64; 8] = [-3, -2, -1, 0, 1, 2, 3, 5];

#[derive(Debug, Clone, PartialEq)]
pub enum Op { Arg(&'static str, i64), NoArg(&'static str) }

pub type Program = Vec<Op>;

impl Op {
    /// Lookup by name (safe interpreter boundary for Python/serialized progs).
    pub fn arg_from(name: &str, arg: i64) -> Option<Op> {
        if OPS_ARG.contains(&name) {
            OPS_ARG.iter().find(|&&n| n == name).map(|&n| Op::Arg(n, arg))
        } else { None }
    }
    pub fn noarg_from(name: &str) -> Option<Op> {
        if OPS_NOARG.contains(&name) {
            OPS_NOARG.iter().find(|&&n| n == name).map(|&n| Op::NoArg(n))
        } else { None }
    }
}

pub fn apply_op(op: &Op, x: i64) -> i64 {
    match op {
        Op::Arg("add", a) => x + a,
        Op::Arg("sub", a) => x - a,
        Op::Arg("mul", a) => x * a,
        Op::Arg("max", a) => x.max(*a),
        Op::Arg("min", a) => x.min(*a),
        Op::NoArg("neg") => -x,
        Op::NoArg("square") => x * x,
        Op::NoArg("abs") => x.abs(),
        _ => panic!("bad op"),
    }
}

pub fn execute(program: &Program, x: i64) -> i64 {
    program.iter().fold(x, |v, op| apply_op(op, v))
}

pub struct SynthesisResult {
    pub program: Option<Program>,
    pub train_acc: f64,
    pub val_acc: f64,
    pub test_acc: f64,
    pub candidates: usize,
    pub time_ms: f64,
}

fn accuracy(program: &Option<Program>, examples: &[(i64, i64)]) -> f64 {
    match program {
        None => 0.0,
        Some(p) => {
            let ok = examples.iter().filter(|(x, y)| execute(p, *x) == *y).count();
            ok as f64 / examples.len().max(1) as f64
        }
    }
}

/// BFS synthesis with train/val/test split. max_depth limits program length.
pub fn synthesize_split(
    train: &[(i64, i64)],
    val: &[(i64, i64)],
    test: &[(i64, i64)],
    max_depth: usize,
    max_nodes: usize,
) -> SynthesisResult {
    let t0 = std::time::Instant::now();
    let start: Vec<i64> = train.iter().map(|(x, _)| *x).collect();
    let target: Vec<i64> = train.iter().map(|(_, y)| *y).collect();
    let mut queue: VecDeque<(Program, Vec<i64>)> = VecDeque::new();
    queue.push_back((Vec::new(), start.clone()));
    let mut visited: HashSet<Vec<i64>> = HashSet::new();
    visited.insert(start);
    let mut candidates = 0usize;
    let mut found: Option<Program> = None;

    'outer: while let Some((prog, state)) = queue.pop_front() {
        if prog.len() >= max_depth { continue; }
        // arg ops
        for op_name in OPS_ARG {
            for arg in ARG_VALUES {
                candidates += 1;
                if candidates > max_nodes { break 'outer; }
                let new_state: Vec<i64> = state.iter().map(|v| apply_op(&Op::Arg(op_name, arg), *v)).collect();
                let mut new_prog = prog.clone();
                new_prog.push(Op::Arg(op_name, arg));
                if new_state == target { found = Some(new_prog); break 'outer; }
                if visited.insert(new_state.clone()) {
                    queue.push_back((new_prog, new_state));
                }
            }
        }
        // noarg ops
        for op_name in OPS_NOARG {
            candidates += 1;
            if candidates > max_nodes { break 'outer; }
            let new_state: Vec<i64> = state.iter().map(|v| apply_op(&Op::NoArg(op_name), *v)).collect();
            let mut new_prog = prog.clone();
            new_prog.push(Op::NoArg(op_name));
            if new_state == target { found = Some(new_prog); break 'outer; }
            if visited.insert(new_state.clone()) {
                queue.push_back((new_prog, new_state));
            }
        }
    }
    let time_ms = t0.elapsed().as_secs_f64() * 1000.0;
    SynthesisResult {
        train_acc: accuracy(&found, train),
        val_acc: accuracy(&found, val),
        test_acc: accuracy(&found, test),
        program: found,
        candidates,
        time_ms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthesize_add_mul() {
        let train = [(0, 6), (1, 9)];
        let val = [(2, 12)];
        let test = [(5, 21)];
        let res = synthesize_split(&train, &val, &test, 3, 50000);
        assert_eq!(res.test_acc, 1.0);
        assert!(res.program.is_some());
    }

    #[test]
    fn execute_safe() {
        let p = vec![Op::Arg("add", 2), Op::Arg("mul", 3)];
        assert_eq!(execute(&p, 5), 21);
    }
}
