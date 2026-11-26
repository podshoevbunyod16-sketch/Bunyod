//! TruthGraph — port of graph4.py (v0.4).
//!
//! TRUE/FALSE/UNKNOWN/CONFLICT + history + retract.
//! Fixes v0.3 GraphMemory overwrite (no FALSE, no history).

use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TruthStatus { True, False, Unknown, Conflict }

impl std::fmt::Display for TruthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            TruthStatus::True => write!(f, "TRUE"),
            TruthStatus::False => write!(f, "FALSE"),
            TruthStatus::Unknown => write!(f, "UNKNOWN"),
            TruthStatus::Conflict => write!(f, "CONFLICT"),
        }
    }
}

pub struct TruthGraph {
    true_edges: HashMap<(i64, i64), HashSet<i64>>,
    false_edges: HashMap<(i64, i64), HashSet<i64>>,
    functional: HashSet<i64>,
    history: Vec<(u64, String, i64, i64, i64)>,
    step: u64,
}

impl TruthGraph {
    pub fn new(functional_relations: Vec<i64>) -> Self {
        Self {
            true_edges: HashMap::new(),
            false_edges: HashMap::new(),
            functional: functional_relations.into_iter().collect(),
            history: Vec::new(),
            step: 0,
        }
    }

    fn log(&mut self, op: &str, s: i64, r: i64, o: i64) {
        self.step += 1;
        self.history.push((self.step, op.to_string(), s, r, o));
    }

    pub fn assert_true(&mut self, s: i64, r: i64, o: i64) {
        self.log("assert_true", s, r, o);
        self.true_edges.entry((s, r)).or_default().insert(o);
    }

    pub fn assert_false(&mut self, s: i64, r: i64, o: i64) {
        self.log("assert_false", s, r, o);
        self.false_edges.entry((s, r)).or_default().insert(o);
    }

    pub fn retract(&mut self, s: i64, r: i64, o: i64) {
        self.log("retract", s, r, o);
        if let Some(set) = self.true_edges.get_mut(&(s, r)) { set.remove(&o); }
    }

    pub fn query(&self, s: i64, r: i64, o: Option<i64>) -> TruthStatus {
        let trues = self.true_edges.get(&(s, r));
        let falses = self.false_edges.get(&(s, r));
        if let Some(o) = o {
            let in_true = trues.map(|s| s.contains(&o)).unwrap_or(false);
            let in_false = falses.map(|s| s.contains(&o)).unwrap_or(false);
            return match (in_true, in_false) {
                (true, true) => TruthStatus::Conflict,
                (true, false) => TruthStatus::True,
                (false, true) => TruthStatus::False,
                (false, false) => TruthStatus::Unknown,
            };
        }
        let n_true = trues.map(|s| s.len()).unwrap_or(0);
        if n_true > 1 && self.functional.contains(&r) {
            return TruthStatus::Conflict;
        }
        let has_both = match (trues, falses) {
            (Some(t), Some(f)) => !t.is_disjoint(f),
            _ => false,
        };
        if has_both { return TruthStatus::Conflict; }
        if n_true == 1 { return TruthStatus::True; }
        if falses.map(|s| !s.is_empty()).unwrap_or(false) { return TruthStatus::False; }
        TruthStatus::Unknown
    }

    pub fn history_len(&self) -> usize { self.history.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truth_semantics() {
        let mut g = TruthGraph::new(vec![0]);
        g.assert_true(1, 0, 2);
        assert_eq!(g.query(1, 0, Some(2)), TruthStatus::True);
        assert_eq!(g.query(1, 0, Some(3)), TruthStatus::Unknown);
        g.assert_false(1, 0, 2);
        assert_eq!(g.query(1, 0, Some(2)), TruthStatus::Conflict);
        g.assert_true(1, 0, 3); // functional with two trues
        assert_eq!(g.query(1, 0, None), TruthStatus::Conflict);
        assert_eq!(g.history_len(), 3);
    }
}
