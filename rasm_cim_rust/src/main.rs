//! RASM-CIM v0.7 CLI: run core operations, print JSON results.
//! Usage: rasm_cim <command> [args]
//! Commands: rns-roundtrip, memory-demo, ca-demo, synth-demo, packed-demo, storage-demo

use rasm_cim_rust::rns::Rns;
use rasm_cim_rust::memory::{MemorySystem, QueryResult};
use rasm_cim_rust::ca::{ca_step_dense, ca_step_sparse, ca_parity};
use rasm_cim_rust::synthesis::synthesize_split;
use rasm_cim_rust::compute::PackedStore;
use rasm_cim_rust::storage::{serialize_facts, deserialize_facts};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("help");
    match cmd {
        "rns-roundtrip" => {
            let mut r = Rns::new(vec![3, 5, 7, 11, 13]).unwrap();
            let mut ok = true;
            for x in [0u64, 42, 1000, 15014] {
                let enc = r.encode_unsigned(x).unwrap();
                if r.decode_unsigned(&enc) != x { ok = false; }
            }
            println!("{{\"ok\": {}, \"ops\": {}}}", ok, r.ops());
        }
        "memory-demo" => {
            let mut m = MemorySystem::new(128, 0.5, 0);
            m.write_episodic(1, 100);
            let dup = m.write_episodic(1, 100);
            m.write_episodic(1, 200);
            let q = m.query_exact(1);
            let conflict = matches!(q, QueryResult::Conflict { .. });
            println!("{{\"dup\": {}, \"conflict\": {}}}", dup, conflict);
        }
        "ca-demo" => {
            let mut state = vec![0u8; 64];
            state[10] = 1; state[20] = 1; state[30] = 1;
            let dense = ca_step_dense(&state, 90);
            let (_, sparse_ops) = ca_step_sparse(&state, 90);
            let (parity, _) = ca_parity(&[1, 0, 1, 1, 0]);
            println!("{{\"dense_len\": {}, \"sparse_ops\": {}, \"parity\": {}}}", dense.len(), sparse_ops, parity);
        }
        "synth-demo" => {
            let res = synthesize_split(&[(0, 6), (1, 9)], &[(2, 12)], &[(5, 21)], 3, 50000);
            println!("{{\"test_acc\": {}, \"candidates\": {}}}", res.test_acc, res.candidates);
        }
        "packed-demo" => {
            let mut ps = PackedStore::new();
            for i in 0..1000 { ps.put(i, i * 2); }
            println!("{{\"bytes\": {}, \"get42\": {}}}", ps.bytes_used(), ps.get(42).unwrap());
        }
        "storage-demo" => {
            let facts = vec![(1, 100), (42, 84)];
            let data = serialize_facts(&facts);
            let back = deserialize_facts(&data).unwrap();
            println!("{{\"roundtrip\": {}, \"bytes\": {}}}", back == facts, data.len());
        }
        _ => {
            println!("commands: rns-roundtrip, memory-demo, ca-demo, synth-demo, packed-demo, storage-demo");
        }
    }
}
