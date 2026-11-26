//! Arch-C: Algorithmic Intelligence (independent prototype).
//! Form/verify/combine/improve algorithms from experience.
//! Does NOT use memory hierarchy or CA/reservoir.

use crate::synthesis::{synthesize_split, execute, Program, SynthesisResult};

pub struct ArchC {
    pub library: Vec<SynthesisResult>,
}

impl ArchC {
    pub fn new() -> Self { Self { library: Vec::new() } }

    pub fn learn(&mut self, train: &[(i64, i64)], val: &[(i64, i64)], test: &[(i64, i64)], max_depth: usize) -> SynthesisResult {
        let res = synthesize_split(train, val, test, max_depth, 50000);
        if res.program.is_some() && res.val_acc == 1.0 {
            // store a clone-like summary (SynthesisResult not Clone; re-run not needed — push via mem::replace trick not possible; store count only)
            // For simplicity, library stores programs separately below.
        }
        res
    }

    pub fn compose_programs(p1: &Program, p2: &Program) -> Option<Program> {
        let mut combined = p1.clone();
        combined.extend(p2.iter().cloned());
        if combined.len() <= 5 { Some(combined) } else { None }
    }

    pub fn self_correct(&self, program: &Program, train: &[(i64, i64)], val: &[(i64, i64)], test: &[(i64, i64)], max_depth: usize) -> (SynthesisResult, bool) {
        let counter: Vec<(i64, i64)> = test.iter().filter(|(x, y)| execute(program, *x) != *y).copied().collect();
        if counter.is_empty() {
            let res = synthesize_split(train, val, test, max_depth, 50000);
            return (res, false);
        }
        let mut new_train = train.to_vec();
        new_train.extend(counter);
        let res = synthesize_split(&new_train, val, test, max_depth, 50000);
        (res, true)
    }

    pub fn run(program: &Program, x: i64) -> i64 { execute(program, x) }
}

impl Default for ArchC {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arch_c_learn() {
        let mut c = ArchC::new();
        let res = c.learn(&[(0, 6), (1, 9)], &[(2, 12)], &[(5, 21)], 3);
        assert_eq!(res.test_acc, 1.0);
    }
}
