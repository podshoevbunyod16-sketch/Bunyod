# RASM-CIM v0.7 — Android / Termux build & run instructions

**Status disclaimer**: these instructions were validated by *compiling the
crate for `aarch64-linux-android` on a Linux x86_64 host* (rlib artifact
verified `ELF 64-bit LSB relocatable, ARM aarch64`) and by a clean
Linux-native build+test here. **No on-device Android run has been performed
yet** — Android support is therefore NOT declared complete (v0.7 rule §8).
The first real Termux run replaces the results table below with measured
device numbers (v0.8 plan item M1).

## A. Termux (on-device, no NDK needed)

```bash
pkg update
pkg install rust python binutils clang make
pip install pytest numpy

# get the crate (unzip the delivery archive or git clone)
cd ~/ && unzip RASM-CIM_v0.7_RUST_*.zip && cd rasm_cim_rust

# 1) CLI (no Python) — proves the pure-Rust core runs on ARM64 Android
cargo build --release --bin rasm_cim
./target/release/rasm_cim rns-roundtrip    # -> {"ok": true, ...}
./target/release/rasm_cim memory-demo
./target/release/rasm_cim synth-demo       # -> candidates: 238 on this task
cargo test --release                       # all unit + property tests

# 2) Python bindings (research interface)
cd bindings/python
cargo build --release                      # links against Termux libpython
cp target/release/librasm_cim_ext.so "$(python -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')/rasm_cim_ext.so"
cd ../..
python -c "import rasm_cim_ext as rx; print(rx.__version__)"

# 3) cross-language tests + benchmarks on device
python -m pytest tests/test_cross_language.py -q
mkdir -p results_v7; python benchmarks/run_bench.py --drive --out results_v7/bench_termux.json
```

Mobile profile rules already implemented in the core (`src/profiles`):
- Mobile = **1 rayon thread, parallelism OFF**, RAM budget 256 MB soft cap —
  on battery/thermal-limited ARM cores the measured desktop result (parallel
  put 0.87× *slower*) says sequential is the right default.
- Streaming/paged read of the storage file is **not implemented yet**
  (v0.8 M2); until then keep files under budget and use
  `storage::read_header` to inspect without loading payloads.

## B. Cross-compile from a Linux host (NDK route, for CI)

```bash
# Android NDK r27+ provides clang + sysroot
export CC_aarch64_linux_android=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang
cargo build --target aarch64-linux-android --release          # rlib/CLI need linking → CC above
cargo build --target aarch64-linux-android --release --lib    # rlib alone: verified HERE without NDK
```

## C. Build matrix as measured in this sandbox (2026-11-25)

| Target | Artifact | Status |
|---|---|---|
| x86_64-unknown-linux-gnu | lib+bin+cdylib(py)+tests | ✅ full: 32+6 rust tests, 28 cross-language tests pass |
| aarch64-linux-android | rlib (release) | ✅ compiles; objects verified `ELF ARM aarch64` (951 KB rlib) |
| aarch64-linux-android | bin/cdylib link | ⚠️ needs NDK linker — not available in sandbox (documented, not faked) |
| aarch64-apple-darwin | cargo check | ✅ compiles (no macOS runner — untested at runtime) |
| x86_64-pc-windows-msvc | cargo check | ✅ compiles (no Windows runner — untested at runtime) |

`Windows/macOS: compile-checked only. Runtime numbers on those platforms are
NOT claimed.`
