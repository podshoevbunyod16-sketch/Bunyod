# v0.9 M4 — Android/Termux A/B

**STATUS 2026-11-26: r2 run received** (20k x2 + 100k x2, mismatches 0/0,
mem9 best p95 — see results_v9/device/DEVICE_RESULTS_PASTED.json). TWO steps left:

1) **Extract the arms your printed summary filtered out** (it kept only dicts with
`picks`; the flash-fsync update numbers decide the 5 ms gate and ARE in the JSONs):

```bash
python3 - <<'EXTRACT'
import json, glob, os
for f in sorted(glob.glob(os.path.expanduser('~/rasm_v9_device_results/v09_ab_*.json'))):
    d = json.load(open(f))
    for k, v in d.items():
        if isinstance(v, dict) and "picks" not in v:
            print(f.split('/')[-1], "|", k, ":", {a: b for a, b in v.items() if a != "picks"})
EXTRACT
```

2) **Re-run with the r3 harness** (`mobile_bench9.py` in this archive is now r3: each
arm runs in an isolated subprocess — fixes VmRSS/VmHWM cross-contamination seen in r2
where all arms @100k reported the same 54.8/68.8 MiB).

Nothing in this folder has device numbers yet; the sandbox has no Android.
Host smoke (2026-11-25, Linux 6.1.189+, n=3000) proves only that the harness
RUNS end-to-end: all 5 arms populated, mismatch stream_vs_legacy=0,
mem9_vs_legacy=0, update p50=0.514 ms with fsync on the sandbox FS.
These are **host** numbers and must never be quoted as device numbers.

## Steps on the phone
1. `pkg install python` (3.10+), copy+unpack `RASM-CIM_v0.1-v0.9_PYTHON_FULL_*.zip` to `~/rasm_cim_v9_pkg`.
2. `bash rasm_cim_v9_pkg/rasm_cim_v9/mobile/termux_setup_v09.sh`
3. `bash rasm_cim_v9_pkg/rasm_cim_v9/mobile/termux_run_v09.sh`  (n=20k first, 100k if RSS stays < ~300 MB free)
4. Send back `~/rasm_v9_device_results/*.json`. Device JSONs are stored under
   `results_v9/device/` separately from host results — this separation is a
   spec gate, not organization detail.

## What it measures (identical facts/keys/seed/dim=256/bits=7 for every arm)
- v08 legacy HV `open_hv()` vs v08 streaming HV vs v0.9 `MobileMemory9` (v3+journal+auto)
- q p50/p95, init time, VmRSS at query + VmHWM peak, update p50/p95 **with fsync
  on flash** (the host tmpfs number 0.03 ms is NOT representative — see M2 note),
- HV pick mismatch counts vs the legacy reference (0 required),
- battery before/after; temp/cpu-freq fields null-tolerant (missing sysfs ≠ failure).

## Gate for v1.0 (from docs_v9/PLAN_v1.0.md)
M4 closes when: 20k AND 100k JSONs cover ALL FIVE arms (incl. v09_v3_journal_updates
verdict from DEVICE flash fsync — host 0.032 ms tmpfs number may not substitute),
mismatches=0 (MET in r2 data), and an r3 run provides uncontaminated per-arm
RSS/peak. Device outputs stay in ~/rasm_v9_device_results and are archived under
results_v9/device/ only.
