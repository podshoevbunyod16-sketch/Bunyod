//! Arch-B: Dynamic Computational Intelligence (independent prototype).
//! Dynamics / local interactions / graphs instead of sequential tokens.
//! Does NOT use memory hierarchy or program synthesis.

use std::collections::{HashMap, HashSet, VecDeque};
use crate::ca::{ca_step_dense, ca_step_sparse, ca_parity, EchoStateReservoir};

pub struct ArchB {
    pub n_cells: usize,
    pub state: Vec<u8>,
    pub reservoir: EchoStateReservoir,
    pub adj: HashMap<i64, Vec<i64>>,
}

impl ArchB {
    pub fn new(n_cells: usize, seed: u64) -> Self {
        Self {
            n_cells,
            state: vec![0u8; n_cells],
            reservoir: EchoStateReservoir::new(1, 64, 0.9, seed),
            adj: HashMap::new(),
        }
    }

    pub fn set_state(&mut self, bits: &[u8]) { self.state = bits.to_vec(); }

    /// Returns (new_state, ops). Sparse ops = updated cells only.
    pub fn ca_step(&mut self, rule: u32, sparse: bool) -> (Vec<u8>, usize) {
        if sparse {
            let (new, n) = ca_step_sparse(&self.state, rule);
            self.state = new.clone();
            (new, n)
        } else {
            let new = ca_step_dense(&self.state, rule);
            let n = self.n_cells;
            self.state = new.clone();
            (new, n)
        }
    }

    pub fn ca_parity(&self, bits: &[u8]) -> u8 { ca_parity(bits).0 }

    pub fn add_edge(&mut self, a: i64, b: i64) {
        self.adj.entry(a).or_default().push(b);
    }

    /// Graph propagation (local message passing, BFS).
    pub fn propagate(&self, start: i64, steps: usize) -> HashSet<i64> {
        let mut seen = HashSet::new();
        seen.insert(start);
        let mut frontier: VecDeque<(i64, usize)> = VecDeque::new();
        frontier.push_back((start, 0));
        while let Some((node, d)) = frontier.pop_front() {
            if d >= steps { continue; }
            if let Some(neighbors) = self.adj.get(&node) {
                for &nb in neighbors {
                    if seen.insert(nb) {
                        frontier.push_back((nb, d + 1));
                    }
                }
            }
        }
        seen
    }

    pub fn reservoir_step(&mut self, x: f64) -> Vec<f64> {
        self.reservoir.step(&[x])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arch_b_ca_and_graph() {
        let mut b = ArchB::new(64, 0);
        let mut bits = vec![0u8; 64];
        bits[10] = 1;
        b.set_state(&bits);
        let (_, dense) = b.ca_step(90, false);
        b.set_state(&bits);
        let (_, sparse) = b.ca_step(90, true);
        assert!(sparse < dense);
        b.add_edge(0, 1); b.add_edge(1, 2);
        assert!(b.propagate(0, 2).contains(&2));
    }
}
