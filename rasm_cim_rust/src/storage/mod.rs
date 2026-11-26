//! Binary storage format (§9): versioned, checksummed, safe deserialization.
//!
//! Format v1:
//!   magic: 4 bytes "RASM"
//!   version: u32 LE (=1)
//!   payload_len: u64 LE
//!   payload: records (key i64 LE, value i64 LE) * n
//!   checksum: u64 LE (FNV-1a over payload)
//! Supports partial read (header only), update (rewrite), Python import/export
//! (same LE layout as Python PackedStore struct.pack("<qq")).

use crate::RasmError;
use std::collections::HashMap;

pub const FORMAT_VERSION: u32 = 1;
const MAGIC: &[u8; 4] = b"RASM";

fn fnv1a(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in data {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Serialize facts to binary format.
pub fn serialize_facts(facts: &[(i64, i64)]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(facts.len() * 16);
    for (k, v) in facts {
        payload.extend_from_slice(&k.to_le_bytes());
        payload.extend_from_slice(&v.to_le_bytes());
    }
    let checksum = fnv1a(&payload);
    let mut out = Vec::with_capacity(4 + 4 + 8 + payload.len() + 8);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(&payload);
    out.extend_from_slice(&checksum.to_le_bytes());
    out
}

/// Deserialize with integrity check. Safe: no panics on malformed input.
pub fn deserialize_facts(data: &[u8]) -> Result<Vec<(i64, i64)>, RasmError> {
    if data.len() < 24 {
        return Err(RasmError::InvalidData("too short".into()));
    }
    if &data[0..4] != MAGIC {
        return Err(RasmError::InvalidData("bad magic".into()));
    }
    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    if version != FORMAT_VERSION {
        return Err(RasmError::FormatVersion { expected: FORMAT_VERSION, got: version });
    }
    let payload_len = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
    if data.len() != 16 + payload_len + 8 {
        return Err(RasmError::InvalidData("length mismatch".into()));
    }
    let payload = &data[16..16 + payload_len];
    let checksum = u64::from_le_bytes(data[16 + payload_len..].try_into().unwrap());
    if fnv1a(payload) != checksum {
        return Err(RasmError::ChecksumMismatch);
    }
    if payload_len % 16 != 0 {
        return Err(RasmError::InvalidData("payload not multiple of 16".into()));
    }
    let mut out = Vec::with_capacity(payload_len / 16);
    for chunk in payload.chunks_exact(16) {
        let k = i64::from_le_bytes(chunk[0..8].try_into().unwrap());
        let v = i64::from_le_bytes(chunk[8..16].try_into().unwrap());
        out.push((k, v));
    }
    Ok(out)
}

/// Partial read: header only (version + count), without full payload.
pub fn read_header(data: &[u8]) -> Result<(u32, usize), RasmError> {
    if data.len() < 16 || &data[0..4] != MAGIC {
        return Err(RasmError::InvalidData("bad header".into()));
    }
    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    let payload_len = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
    Ok((version, payload_len / 16))
}

/// Import Python PackedStore bytes (raw "<qq" records, no header).
pub fn import_python_packed(data: &[u8]) -> Result<Vec<(i64, i64)>, RasmError> {
    if data.len() % 16 != 0 {
        return Err(RasmError::InvalidData("python packed len % 16 != 0".into()));
    }
    let mut out = Vec::with_capacity(data.len() / 16);
    for chunk in data.chunks_exact(16) {
        let k = i64::from_le_bytes(chunk[0..8].try_into().unwrap());
        let v = i64::from_le_bytes(chunk[8..16].try_into().unwrap());
        out.push((k, v));
    }
    Ok(out)
}

pub fn facts_to_map(facts: Vec<(i64, i64)>) -> HashMap<i64, i64> {
    facts.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let facts = vec![(1, 100), (42, 84), (-5, -10)];
        let data = serialize_facts(&facts);
        assert_eq!(deserialize_facts(&data).unwrap(), facts);
        let (ver, n) = read_header(&data).unwrap();
        assert_eq!((ver, n), (1, 3));
    }

    #[test]
    fn checksum_detects_corruption() {
        let mut data = serialize_facts(&[(1, 2)]);
        data[20] ^= 0xFF;
        assert_eq!(deserialize_facts(&data), Err(RasmError::ChecksumMismatch));
    }

    #[test]
    fn python_import() {
        // Python struct.pack("<qq", 7, 14) = 7 LE + 14 LE
        let mut raw = Vec::new();
        raw.extend_from_slice(&7i64.to_le_bytes());
        raw.extend_from_slice(&14i64.to_le_bytes());
        assert_eq!(import_python_packed(&raw).unwrap(), vec![(7, 14)]);
    }
}
