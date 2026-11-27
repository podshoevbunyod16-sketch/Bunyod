# ANDROID / TERMUX PROFILE — RASM-CIM v0.15.0
**Status: NOT TESTED. No Android device or emulator was available in the verification
environment. Every number below that is marked "measured" is a host (Linux x86-64) number and
is NOT transferable to ARM64/Android. Nothing in this file is a device measurement.**

## Target profile (what to measure on a real device)
- CPU: ARM64 (aarch64), 1–2 cores, throttled; RAM: ≤ 3 GB total, soft budget 256 MB for the process (mobile profile already implemented in `rasm_cim_rust/src/profiles`: 1 rayon thread, parallelism OFF).
- Storage: flash, fsync latency unknown — the v0.7/v0.10 flash-fsync verdict remains **INCONCLUSIVE** (needs on-device run).
- Battery/thermal: long runs may down-clock; report both cold and warm runs.

## How to run (device side)
See `rasm_cim_rust/mobile/TERMUX.md` for the Rust/Termux build matrix. For the Python stack:

```bash
pkg update && pkg install python rust Binutils clang make
pip install numpy scikit-learn pytest hypothesis   # atheris optional (fuzzing)
git clone https://github.com/podshoevbunyod16-sketch/Bunyod && cd Bunyod
git checkout 56d275c   # or later verification commit
python -m pytest tests_v11 tests_v12 tests_v13 tests_v14 tests_v15 tests_phase_c -q
python -m pytest rasm_cim_rust/tests/test_cross_language.py -q   # after cargo build, see TERMUX.md
python -m rasm_cim --demo evaluate --split test
/ system/bin/time -v python -m rasm_cim --demo ask "Какая столица у Франции?"
```

## What the device run must report
- pytest pass/fail counts on-device; cross-language tests after `cargo build --release`.
- ask p50/p95; evaluate p50 and unsupported rate; peak RSS; crash/restore demo (`python -m rasm_cim_v9.cli crash`).
- KB replay time at 1k/5k events; export/import roundtrip on-device.
- Thermal status if available (`/sys/class/thermal` may be absent — record `null`, not a failure).

## Open items carried from v0.7/v0.9 (do not claim fixed)
- flash fsync p50/p95 on device — OPEN (v0.9 paste gap, r3 harness written, never device-run).
- Device RSS per arm (r2 cross-contaminated) — OPEN, r3 harness ready, never device-run.
- aarch64-android cdylib LINK — requires NDK/Termux cargo build on device (not host-verifiable).

**Required action: a user with a Termux ARM64 device runs the above and attaches JSON/log
output; until then every device gate stays NOT TESTED / OPEN.**
