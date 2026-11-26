//! Residue Number System (RNS) — port of rns.py / rns2.py / rns4.py.
//!
//! Pairwise-coprime moduli, CRT decode, range checks (no silent wrap —
//! fixes v0.5 audit issue #3), signed support via offset.
//! All arithmetic is exact integer ops; no floating point.

use crate::RasmError;

/// Range-aware RNS with explicit overflow detection.
#[derive(Debug, Clone)]
pub struct Rns {
    moduli: Vec<u32>,
    m: u64,        // product of moduli
    offset: u64,   // M/2 for signed range
    ops: u64,      // operation counter (honest counting)
}

impl Rns {
    /// Create RNS from pairwise-coprime moduli.
    pub fn new(moduli: Vec<u32>) -> Result<Self, RasmError> {
        if moduli.is_empty() {
            return Err(RasmError::EmptyModuli);
        }
        for i in 0..moduli.len() {
            for j in (i + 1)..moduli.len() {
                if gcd(moduli[i] as u64, moduli[j] as u64) != 1 {
                    return Err(RasmError::NonCoprimeModuli);
                }
            }
        }
        let m: u64 = moduli.iter().map(|&x| x as u64).product();
        Ok(Self { offset: m / 2, m, moduli, ops: 0 })
    }

    pub fn moduli(&self) -> &[u32] { &self.moduli }
    pub fn m(&self) -> u64 { self.m }
    pub fn ops(&self) -> u64 { self.ops }
    pub fn reset_ops(&mut self) { self.ops = 0 }

    /// Encode unsigned x in [0, M). Errors if out of range (no silent wrap).
    pub fn encode_unsigned(&mut self, x: u64) -> Result<Vec<u32>, RasmError> {
        if x >= self.m {
            return Err(RasmError::OutOfRange(x as i64, self.m as i64));
        }
        self.ops += self.moduli.len() as u64;
        Ok(self.moduli.iter().map(|&m| (x % m as u64) as u32).collect())
    }

    /// Encode signed x in [-M/2, M/2).
    pub fn encode_signed(&mut self, x: i64) -> Result<Vec<u32>, RasmError> {
        let lo = -(self.offset as i64);
        let hi = self.m as i64 - self.offset as i64;
        if x < lo || x >= hi {
            return Err(RasmError::OutOfRange(x, self.m as i64));
        }
        self.encode_unsigned((x + self.offset as i64) as u64)
    }

    /// CRT decode residues -> unsigned x in [0, M).
    pub fn decode_unsigned(&mut self, residues: &[u32]) -> u64 {
        assert_eq!(residues.len(), self.moduli.len());
        self.ops += self.moduli.len() as u64;
        let mut x = residues[0] as u64;
        let mut mp = self.moduli[0] as u64;
        for i in 1..self.moduli.len() {
            let mi = self.moduli[i] as u64;
            let inv = modinv(mp % mi, mi);
            // x must be reduced mod mi before subtracting (Python % semantics).
            let diff = (residues[i] as u64 + mi - (x % mi)) % mi;
            let t = (diff * inv) % mi;
            x += mp * t;
            mp *= mi;
        }
        x % self.m
    }

    /// Decode to signed range.
    pub fn decode_signed(&mut self, residues: &[u32]) -> i64 {
        self.decode_unsigned(residues) as i64 - self.offset as i64
    }

    /// Componentwise RNS subtraction (borrow-free). Ops counted.
    pub fn sub_residues(&mut self, a: &[u32], b: &[u32]) -> Vec<u32> {
        self.ops += self.moduli.len() as u64;
        a.iter().zip(b.iter()).zip(self.moduli.iter())
            .map(|((&x, &y), &m)| ((x as u64 + m as u64 - y as u64) % m as u64) as u32).collect()
    }

    /// Componentwise RNS addition (carry-free). Ops counted.
    pub fn add_residues(&mut self, a: &[u32], b: &[u32]) -> Vec<u32> {
        self.ops += self.moduli.len() as u64;
        a.iter().zip(b.iter()).zip(self.moduli.iter())
            .map(|((&x, &y), &m)| ((x + y) % m) as u32).collect()
    }

    /// Componentwise RNS multiplication (carry-free). Ops counted.
    pub fn mul_residues(&mut self, a: &[u32], b: &[u32]) -> Vec<u32> {
        self.ops += self.moduli.len() as u64;
        a.iter().zip(b.iter()).zip(self.moduli.iter())
            .map(|((&x, &y), &m)| ((x as u64 * y as u64) % m as u64) as u32).collect()
    }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 { let t = b; b = a % b; a = t; }
    a
}

/// Modular inverse via extended Euclid (a, m coprime).
fn modinv(a: u64, m: u64) -> u64 {
    let (mut t, mut new_t) = (0i64, 1i64);
    let (mut r, mut new_r) = (m as i64, a as i64);
    while new_r != 0 {
        let q = r / new_r;
        let tmp_t = t - q * new_t; t = new_t; new_t = tmp_t;
        let tmp_r = r - q * new_r; r = new_r; new_r = tmp_r;
    }
    if t < 0 { t += m as i64; }
    t as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_unsigned() {
        let mut r = Rns::new(vec![3, 5, 7, 11, 13]).unwrap();
        for x in [0u64, 42, 1000, 15014] {
            let enc = r.encode_unsigned(x).unwrap();
            assert_eq!(r.decode_unsigned(&enc), x);
        }
    }

    #[test]
    fn overflow_raises() {
        let mut r = Rns::new(vec![3, 5, 7, 11, 13]).unwrap();
        assert!(r.encode_unsigned(15015).is_err()); // M = 15015
    }

    #[test]
    fn signed_roundtrip() {
        let mut r = Rns::new(vec![3, 5, 7, 11, 13]).unwrap();
        for x in [-100i64, -1, 0, 42, 7500] {
            let enc = r.encode_signed(x).unwrap();
            assert_eq!(r.decode_signed(&enc), x);
        }
    }

    #[test]
    fn add_mul_residues() {
        let mut r = Rns::new(vec![3, 5, 7, 11, 13]).unwrap();
        let a = r.encode_unsigned(100).unwrap();
        let b = r.encode_unsigned(23).unwrap();
        let sum = r.add_residues(&a, &b);
        assert_eq!(r.decode_unsigned(&sum), 123);
        let prod = r.mul_residues(&a, &b);
        assert_eq!(r.decode_unsigned(&prod), 2300);
    }

    #[test]
    fn non_coprime_rejected() {
        assert!(Rns::new(vec![4, 6]).is_err());
    }
}
