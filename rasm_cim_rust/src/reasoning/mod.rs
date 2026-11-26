//! Reasoning facade over TruthGraph (§3 semantics): facts, composition.
//! Kept separate from memory: memory = storage, reasoning = derivation.

use crate::graph::TruthGraph;

pub struct Reasoner<'a> {
    pub graph: &'a mut TruthGraph,
}

impl<'a> Reasoner<'a> {
    pub fn new(graph: &'a mut TruthGraph) -> Self { Self { graph } }

    /// Compose (s1,r1,o1) & (o1,r2,o2) -> (s1, compose(r1,r2), o2) —
    /// mirrors v0.4 GraphMemory.compose with relation composition = (r1*100+r2).
    pub fn compose(&mut self, s1: i64, r1: i64, o1: i64, r2: i64, o2: i64) -> String {
        self.graph.assert_true(s1, r1, o1);
        self.graph.assert_true(o1, r2, o2);
        let rc = r1 * 100 + r2;
        self.graph.assert_true(s1, rc, o2);
        self.graph.query(s1, rc, Some(o2)).to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::TruthGraph;

    #[test]
    fn composition() {
        let mut g = TruthGraph::new(vec![102]);
        let mut r = Reasoner::new(&mut g);
        assert_eq!(r.compose(1, 1, 2, 2, 3), "TRUE");
        assert_eq!(g.query(1, 102, Some(3)).to_string(), "TRUE");
        assert_eq!(g.query(1, 999, None).to_string(), "UNKNOWN");
    }
}
