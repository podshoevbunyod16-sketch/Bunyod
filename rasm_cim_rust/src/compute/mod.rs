//! Compute core: PackedStore + parallel ops — port of compute_core.py (v0.6 §8).
//!
//! PackedStore: contiguous (key i64, value i64) records, 16 B/record.
//! Python dict overhead measured 5.8× larger (v0.6). Rust HashMap index
//! + packed buffer. Parallel batch ops via rayon where beneficial.

use std::collections::HashMap;
use rayon::prelude::*;

/// Packed binary storage for int facts.
pub struct PackedStore {
    buf: Vec<u8>,               // contiguous LE bytes
    index: HashMap<i64, usize>, // key -> byte offset
}

impl PackedStore {
    pub fn new() -> Self {
        Self { buf: Vec::new(), index: HashMap::new() }
    }

    pub fn put(&mut self, key: i64, value: i64) {
        let off = self.buf.len();
        self.buf.extend_from_slice(&key.to_le_bytes());
        self.buf.extend_from_slice(&value.to_le_bytes());
        self.index.insert(key, off);
    }

    pub fn get(&self, key: i64) -> Option<i64> {
        let off = *self.index.get(&key)?;
        let mut kb = [0u8; 8];
        let mut vb = [0u8; 8];
        kb.copy_from_slice(&self.buf[off..off + 8]);
        vb.copy_from_slice(&self.buf[off + 8..off + 16]);
        Some(i64::from_le_bytes(vb))
    }

    pub fn bytes_used(&self) -> usize { self.buf.len() }

    /// Raw buffer: little-endian (key,value) records — bit-identical to
    /// Python `struct.pack("<qq", k, v)` concatenation (v0.6 PackedStore).
    pub fn to_bytes(&self) -> Vec<u8> { self.buf.clone() }

    /// Rebuild store from raw record bytes. Insert order = file order,
    /// so duplicate keys keep FIRST-wins? No: mirrors put() semantics —
    /// later record overwrites index (same as Python rebuild loop).
    pub fn from_bytes(data: &[u8]) -> Result<Self, crate::RasmError> {
        if data.len() % 16 != 0 {
            return Err(crate::RasmError::InvalidData("len % 16 != 0".into()));
        }
        let mut store = Self::new();
        for (i, chunk) in data.chunks_exact(16).enumerate() {
            let mut k = [0u8; 8]; let mut v = [0u8; 8];
            k.copy_from_slice(&chunk[0..8]);
            v.copy_from_slice(&chunk[8..16]);
            let key = i64::from_le_bytes(k);
            store.buf.extend_from_slice(chunk);
            store.index.insert(key, i * 16);
            let _ = v;
        }
        Ok(store)
    }
    pub fn len(&self) -> usize { self.index.len() }
    pub fn is_empty(&self) -> bool { self.index.is_empty() }

    /// Parallel batch get (rayon). Order preserved.
    pub fn batch_get_parallel(&self, keys: &[i64]) -> Vec<Option<i64>> {
        keys.par_iter().map(|k| self.get(*k)).collect()
    }

    /// Parallel batch put (rayon): build records in parallel, then append.
    pub fn batch_put_parallel(&mut self, pairs: &[(i64, i64)]) {
        let records: Vec<(i64, usize, [u8; 16])> = pairs.par_iter().enumerate()
            .map(|(i, (k, v))| {
                let mut rec = [0u8; 16];
                rec[..8].copy_from_slice(&k.to_le_bytes());
                rec[8..].copy_from_slice(&v.to_le_bytes());
                (*k, i, rec)
            }).collect();
        let base = self.buf.len();
        // append in order (deterministic)
        let mut sorted = records;
        sorted.sort_by_key(|(_, i, _)| *i);
        for (k, _, rec) in sorted {
            let off = self.buf.len();
            self.buf.extend_from_slice(&rec);
            self.index.insert(k, off);
        }
        let _ = base;
    }
}

impl Default for PackedStore {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_roundtrip() {
        let mut ps = PackedStore::new();
        for i in 0..100 { ps.put(i, i * 2); }
        assert_eq!(ps.get(42), Some(84));
        assert_eq!(ps.bytes_used(), 1600);
    }

    #[test]
    fn parallel_batch() {
        let mut ps = PackedStore::new();
        let pairs: Vec<(i64, i64)> = (0..1000).map(|i| (i, i * 3)).collect();
        ps.batch_put_parallel(&pairs);
        let keys: Vec<i64> = (0..1000).collect();
        let vals = ps.batch_get_parallel(&keys);
        assert!(vals.iter().enumerate().all(|(i, v)| *v == Some(i as i64 * 3)));
    }
}
