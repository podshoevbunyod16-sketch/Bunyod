//! Arch-A: Memory-Centric Intelligence (independent prototype).
//! Associative retrieval, hierarchy, links, update, contradiction resolution.
//! Does NOT use CA/reservoir/synthesis.

use crate::memory::{MemorySystem, QueryResult};

pub struct ArchA {
    pub mem: MemorySystem,
}

impl ArchA {
    pub fn new(d: usize, seed: u64) -> Self {
        Self { mem: MemorySystem::new(d, 0.5, seed) }
    }
    pub fn write(&mut self, key: i64, value: i64) -> bool { self.mem.write_episodic(key, value) }
    pub fn query_exact(&self, key: i64) -> QueryResult { self.mem.query_exact(key) }
    pub fn query_partial(&self, residue: i64, modulus: i64) -> Vec<i64> {
        self.mem.query_partial(residue, modulus)
    }
    pub fn query_noisy(&self, qhv: &[f64]) -> QueryResult { self.mem.query_noisy(qhv) }
    pub fn learn_fact(&mut self, s: i64, r: i64, o: i64) { self.mem.learn_fact(s, r, o); }
    pub fn query_fact(&self, s: i64, r: i64, o: Option<i64>) -> String {
        self.mem.query_fact(s, r, o)
    }
    pub fn retention(&self, kv: &[(i64, i64)]) -> bool { self.mem.retention_check(kv) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arch_a_basic() {
        let mut a = ArchA::new(64, 0);
        a.write(1, 100);
        assert!(matches!(a.query_exact(1), QueryResult::True { value: 100, .. }));
        a.learn_fact(1, 0, 2);
        assert_eq!(a.query_fact(1, 0, Some(2)), "TRUE");
    }
}
