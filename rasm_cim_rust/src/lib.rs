//! RASM-CIM v0.7 Rust compute core.
//!
//! Port of the v0.1–v0.6 Python core to Rust: RNS arithmetic, hierarchical
//! memory (exact/partial/noisy/semantic/episodic/procedural), cellular
//! automata, reservoir computing, graph propagation, BFS program synthesis,
//! packed storage. Python remains the research interface (see bindings/).
//!
//! `#![forbid(unsafe_code)]` — compiler-enforced, not just a convention.
//! Deps: thiserror, rayon, rand (dev: criterion). No serde/tokio: removed as
//! unused (see docs/ERRORS_FOUND_v0.7.md #4).

#![forbid(unsafe_code)]

pub mod rns;
pub mod memory;
pub mod graph;
pub mod ca;
pub mod synthesis;
pub mod compute;
pub mod storage;
pub mod architectures;
pub mod learning;
pub mod reasoning;
pub mod profiles;
pub mod core;

use thiserror::Error;

/// Typed errors for RASM-CIM Rust core (replaces Python exceptions).
#[derive(Debug, Error, PartialEq)]
pub enum RasmError {
    #[error("value {0} outside RNS range [0, {1})")]
    OutOfRange(i64, i64),
    #[error("moduli must be pairwise coprime")]
    NonCoprimeModuli,
    #[error("empty moduli list")]
    EmptyModuli,
    #[error("unknown query key {0}")]
    UnknownKey(i64),
    #[error("conflicting values for key {0}")]
    Conflict(i64),
    #[error("storage format version mismatch: expected {expected}, got {got}")]
    FormatVersion { expected: u32, got: u32 },
    #[error("storage checksum mismatch")]
    ChecksumMismatch,
    #[error("invalid storage data: {0}")]
    InvalidData(String),
    #[error("synthesis failed: no program found within limits")]
    SynthesisFailed,
}

