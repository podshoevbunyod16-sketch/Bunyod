//! Criterion micro-benchmarks of the Rust core (stable measurement of
//! per-operation cost; the Python-vs-Rust comparison lives in
//! benchmarks/run_bench.py so both sides see identical data).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rasm_cim_rust::ca::{ca_parity, ca_step_dense, ca_step_sparse};
use rasm_cim_rust::compute::PackedStore;
use rasm_cim_rust::memory::{MemorySystem};
use rasm_cim_rust::rns::Rns;
use rasm_cim_rust::storage::{deserialize_facts, serialize_facts};
use rasm_cim_rust::synthesis::synthesize_split;

fn rns_bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("rns");
    let vals: Vec<u64> = (0..2000).map(|i| (i as u64 * 7919) % 15015).collect();
    g.throughput(Throughput::Elements(vals.len() as u64));
    g.bench_function("encode_2000", |b| {
        b.iter(|| {
            let mut r = Rns::new(vec![3, 5, 7, 11, 13]).unwrap();
            vals.iter().map(|&x| r.encode_unsigned(x).unwrap()).collect::<Vec<_>>()
        })
    });
    g.bench_function("add_mul_2000", |b| {
        b.iter(|| {
            let mut r = Rns::new(vec![3, 5, 7, 11, 13]).unwrap();
            let a = r.encode_unsigned(100).unwrap();
            let acc: Vec<Vec<u32>> = vals.iter()
                .map(|&x| { let e = r.encode_unsigned(x).unwrap(); let s = r.add_residues(&a, &e); r.mul_residues(&s, &a) })
                .collect();
            acc
        })
    });
    g.finish();
}

fn memory_bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("memory");
    for n in [1_000usize, 10_000, 100_000] {
        let vals: Vec<i64> = (0..n).map(|i| (i as i64 * 7919) % 15015).collect();
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::new("build_exact", n), &n, |b, _| {
            b.iter(|| {
                let mut m = MemorySystem::new(64, 0.5, 0);
                for (i, v) in vals.iter().enumerate() { m.write_episodic(i as i64 + 1, *v); }
                m.episodic_len()
            })
        });
        g.bench_with_input(BenchmarkId::new("partial_mod97", n), &n, |b, _| {
            let mut m = MemorySystem::new(64, 0.5, 0);
            for (i, v) in vals.iter().enumerate() { m.write_episodic(i as i64 + 1, *v); }
            b.iter(|| (0..97i64).map(|r| m.query_partial(r, 97).len()).sum::<usize>())
        });
    }
    g.finish();
}

fn ca_bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("ca");
    let mut state = vec![0u8; 1024];
    for i in (0..1024).step_by(2) { state[i] = ((i as u32).wrapping_mul(2_654_435_761) >> 13) as u8 & 1; }
    g.throughput(Throughput::Elements(1024u64));
    g.bench_function("dense_step1024", |b| b.iter(|| ca_step_dense(&state, 90)));
    let mut sparse = vec![0u8; 1024];
    for i in (0..1024).step_by(64) { sparse[i] = 1; }
    g.bench_function("sparse_step1024", |b| b.iter(|| ca_step_sparse(&sparse, 90)));
    g.bench_function("parity_1024", |b| b.iter(|| ca_parity(&state)));
    g.finish();
}

fn storage_bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("storage");
    let facts: Vec<(i64, i64)> = (0..50_000).map(|i| (i, i * 2)).collect();
    let bytes = serialize_facts(&facts);
    g.throughput(Throughput::Bytes(bytes.len() as u64));
    g.bench_function("serialize_50k", |b| b.iter(|| serialize_facts(&facts)));
    g.bench_function("deserialize_50k", |b| b.iter(|| deserialize_facts(&bytes).unwrap().len()));
    g.bench_function("packed_put_50k", |b| b.iter(|| {
        let mut ps = PackedStore::new();
        for (k, v) in &facts { ps.put(*k, *v); }
        ps.bytes_used()
    }));
    g.finish();
}

fn synth_bench(c: &mut Criterion) {
    c.bench_function("synth_len3", |b| {
        b.iter(|| {
            let r = synthesize_split(&[(0, 6), (1, 9), (-2, 0)], &[(2, 12)], &[(5, 21)], 3, 50000);
            (r.program.is_some(), r.candidates)
        })
    });
}

criterion_group!(benches, rns_bench, memory_bench, ca_bench, storage_bench, synth_bench);
criterion_main!(benches);
