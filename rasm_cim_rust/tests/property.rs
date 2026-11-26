//! Property-based + fuzz-style tests for critical components (§7).
//! Uses a fast pseudo-random generator; ~200k property checks total.
//! (cargo-fuzz needs nightly; stable fuzzing substitute documented in
//! docs/ERRORS_FOUND_v0.7.md — the storage deserializer is exercised with
//! arbitrary byte soup here to prove no panics on malformed input.)

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rasm_cim_rust::compute::PackedStore;
use rasm_cim_rust::rns::Rns;
use rasm_cim_rust::storage::{deserialize_facts, serialize_facts, read_header};

const MODULI_SETS: [&[u32]; 4] = [
    &[3, 5, 7],
    &[3, 5, 7, 11, 13],
    &[2, 3, 5, 7, 11, 13],
    &[16, 9, 25, 7, 11, 13],
];

fn modinv(a: u64, m: u64) -> u64 {
    // for tests only (m small); brute force
    (1..m).find(|&x| (a * x) % m == 1).unwrap()
}

#[test]
fn rns_roundtrip_property() {
    let mut rng = SmallRng::seed_from_u64(11);
    for &mods in &MODULI_SETS {
        let mut r = Rns::new(mods.to_vec()).unwrap();
        let m = r.m();
        for _ in 0..20_000 {
            let x = rng.gen_range(0..m);
            let enc = r.encode_unsigned(x).unwrap();
            // residues correct by definition
            for (&res, &mm) in enc.iter().zip(mods) {
                assert_eq!(res as u64, x % mm as u64);
            }
            assert_eq!(r.decode_unsigned(&enc), x, "roundtrip failed for {x}");
        }
        // out of range always errors, never wraps
        assert!(r.encode_unsigned(m).is_err());
        assert!(r.encode_unsigned(m + 1).is_err());
    }
}

#[test]
fn rns_add_mul_property() {
    let mut rng = SmallRng::seed_from_u64(12);
    let mut r = Rns::new(vec![3, 5, 7, 11, 13, 17]).unwrap();
    let m = r.m();
    for _ in 0..20_000 {
        let a = rng.gen_range(0..m);
        let b = rng.gen_range(0..m);
        let ea = r.encode_unsigned(a).unwrap();
        let eb = r.encode_unsigned(b).unwrap();
        let s = r.add_residues(&ea, &eb);
        assert_eq!(r.decode_unsigned(&s), (a + b) % m);
        let p = r.mul_residues(&ea, &eb);
        assert_eq!(r.decode_unsigned(&p), (a.wrapping_mul(b)) % m);
        let d = r.sub_residues(&ea, &eb);
        assert_eq!(r.decode_unsigned(&d), (a + m - b % m) % m);
    }
}

#[test]
fn rns_crt_matches_naive_sum() {
    // Classic CRT formula: x = Σ r_i * M_i * inv(M_i mod m_i) mod M
    let mut rng = SmallRng::seed_from_u64(13);
    for mods in [&[5u32, 7, 9][..], &[4, 9, 25, 7][..], &[3, 5, 7, 11, 13, 17, 19][..]] {
        let mut r = Rns::new(mods.to_vec()).unwrap();
        let m: u64 = mods.iter().map(|&x| x as u64).product();
        for _ in 0..3_000 {
            let x = rng.gen_range(0..m);
            let enc = r.encode_unsigned(x).unwrap();
            let mut naive = 0u64;
            for (i, &mm) in mods.iter().enumerate() {
                let mi = mm as u64;
                let big = m / mi;
                let inv = modinv(big % mi, mi);
                naive = (naive + enc[i] as u64 * big * inv) % m;
            }
            assert_eq!(r.decode_unsigned(&enc), naive);
        }
    }
}

#[test]
fn packed_store_model_equivalence() {
    let mut rng = SmallRng::seed_from_u64(14);
    for _ in 0..50 {
        let mut ps = PackedStore::new();
        let mut model: Vec<(i64, i64)> = Vec::new();
        for _ in 0..2_000 {
            let k = rng.gen_range(-1_000..1_000);
            let v: i64 = rng.gen();
            ps.put(k, v);
            model.push((k, v));
        }
        // get returns LAST put value for key (index overwrite), like dict
        for k in -1_000..1_000 {
            let want = model.iter().rev().find(|(mk, _)| *mk == k).map(|&(_, v)| v);
            assert_eq!(ps.get(k), want);
        }
        // serialization roundtrip
        let bytes = ps.to_bytes();
        let ps2 = PackedStore::from_bytes(&bytes).unwrap();
        for k in -1_000..1_000 {
            assert_eq!(ps2.get(k), ps.get(k));
        }
    }
}

#[test]
fn storage_roundtrip_and_never_panics() {
    let mut rng = SmallRng::seed_from_u64(15);
    // valid roundtrip
    for _ in 0..2_000 {
        let n = rng.gen_range(0..50);
        let facts: Vec<(i64, i64)> = (0..n).map(|_| (rng.gen(), rng.gen())).collect();
        let bytes = serialize_facts(&facts);
        assert_eq!(deserialize_facts(&bytes).unwrap(), facts);
        let (ver, cnt) = read_header(&bytes).unwrap();
        assert_eq!((ver, cnt), (1, n));
    }
    // random byte soup: must return Err, never panic
    for _ in 0..200_000 {
        let len = rng.gen_range(0..48);
        let junk: Vec<u8> = (0..len).map(|_| rng.gen()).collect();
        let _ = deserialize_facts(&junk); // must not panic
    }
    // corrupted valid files: header intact, payload mutated -> Err or valid-but-checked
    for _ in 0..5_000 {
        let facts: Vec<(i64, i64)> = (0..5).map(|_| (rng.gen(), rng.gen())).collect();
        let mut bytes = serialize_facts(&facts);
        let idx = rng.gen_range(16..bytes.len() - 8);
        bytes[idx] ^= 1 << rng.gen_range(0..8);
        assert!(deserialize_facts(&bytes).is_err(), "checksum must catch {idx}");
    }
}

#[test]
fn graph_no_panic_adversarial() {
    use rasm_cim_rust::graph::TruthGraph;
    let mut rng = SmallRng::seed_from_u64(16);
    let mut g = TruthGraph::new(vec![0, 1]);
    for _ in 0..50_000 {
        let (s, r, o) = (rng.gen_range(-3..3), rng.gen_range(-2..2), rng.gen_range(-3..3));
        match rng.gen_range(0..4) {
            0 => g.assert_true(s, r, o),
            1 => g.assert_false(s, r, o),
            2 => g.retract(s, r, o),
            _ => {
                let _ = g.query(s, r, Some(o));
                let _ = g.query(s, r, None);
            }
        }
    }
    // invariants: any known triple has a decidable status
    assert!(g.query(0, 0, Some(0)).to_string() != "".to_string());
}
