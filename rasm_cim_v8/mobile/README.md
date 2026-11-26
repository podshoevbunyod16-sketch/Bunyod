# RASM-CIM v0.8 — Android / Termux package (M1)

**Status: package ready + host-validated; NO real Android device was available
in this workspace, so M1 is NOT declared complete and all device numbers below
are what the USER will obtain by running it — this file contains zero invented
device measurements.**

## What's included (all pure-Python, stdlib-only core)
- `rasm_cim_v8/storage/format.py` — binary formats v1 (frozen-spec compatible) & v2 (paged, crc32)
- `rasm_cim_v8/storage/streaming.py` — LazyStore: paged pread/mmap/buffered reads, LRU page cache, corrupt-page policy
- `rasm_cim_v8/memory/stream_memory.py` — MobileStreamMemory (stream/residue/dict tiers, optional packed-HV cache)
- `rasm_cim_v8/hv/bitpack.py` — packed bipolar HV, `int.bit_count` similarity (no numpy needed)
- `rasm_cim_v8/cli.py` — JSON CLI (`selftest/build/get/scan/noisy/m5-demo`)
- `mobile_bench.py` — the measurement harness below
- `termux_install.sh`, `termux_run.sh`

## Install (Termux)
1. Copy the **whole workspace root** (folders `rasm_cim_v8/` … the parent of `mobile/`) to the device, e.g. `~/rasm/`.
2. `cd ~/rasm/rasm_cim_v8/mobile && bash termux_install.sh`
3. `bash termux_run.sh 20000`

Optional acceleration (not required): `pip install numpy` — the stdlib path is what is measured by default.

## What gets measured (M1 spec)
| metric | source |
|---|---|
| interpreter import/start time | `import_s` |
| runtime (build, open, 2000×get, scan×4, HV open, noisy) | section fields |
| peak RSS | VmHWM from /proc/self/status + ru_maxrss |
| CPU time | getrusage user+sys |
| serial vs parallel | scans_serial_s vs scans_threads4_s vs scans_proc2_s |
| temperature / throttling | every /sys/class/thermal & hwmon node if readable — reported `"available": false` when the device hides them (never faked) |

## Memory budget
Target <40 MiB peak RSS at 100k records (v0.8 M2 measured **26.2 MiB** on host
desktop; the device number must come from an actual run). Spec budget 3 GB is
not approached: file is 16 B/record + 16 B/record index; 1M records ≈ 32 MB on
disk, streaming RSS stayed 26–84 MiB in host runs.

## M2-refactor HV A/B (what to look for in the JSON)
`mobile_bench.py` runs legacy `open_hv` and the new streaming backend in two
ISOLATED subprocesses and reports `hv_ab_legacy`, `hv_ab_stream`,
`hv_ab_rss_delta_mib`. Pass criterion (from the refactor spec): streaming
peak RSS < 40 MiB at n=100000 on the device (host achieved 32.1 MiB at the
same point; the legacy device numbers 38.7/56.4 MiB are the user-reported
baseline this refactor targets). `auto` picks the numpy path if the device
has numpy, otherwise plain exact regeneration (slower, zero cache).

## Reporting a device result
Attach `mobile_bench_result.json`. Results from an emulator or a Linux host
(this repo ran it on one) are labelled `platform: linux` by the harness itself —
they must not be presented as Android numbers.
