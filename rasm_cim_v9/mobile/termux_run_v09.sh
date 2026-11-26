#!/data/data/com.termux/files/usr/bin/bash
# v0.9 M4 — A/B measurement run on the DEVICE. Results: ~/rasm_v9_device_results/
# No telemetry is sent anywhere; temperature/freq fields may be null (kernel
# restrictions) — that is NOT a run failure.
set -eu
export RASM_PKG="${RASM_PKG:-$HOME/rasm_cim_v9_pkg}"
export PYTHONPATH="$RASM_PKG:${PYTHONPATH:-}"
cd "$RASM_PKG"
# 1) correctness on-device first (hypothesis props are slow on phones: -x subset)
python3 -m pytest tests_v9/test_m1_m2.py tests_v9/test_m5_regressions.py \
  tests_v9/test_m6_graph.py -q
# 2) the A/B itself, sizes recommended by v0.8 device experience:
termux-battery-status > ~/rasm_v9_device_results/battery_before.json 2>/dev/null || true
python3 rasm_cim_v9/mobile/mobile_bench9.py 20000 100000
termux-battery-status > ~/rasm_v9_device_results/battery_after.json  2>/dev/null || true
echo "DONE — send ~/rasm_v9_device_results/*.json back for the report."
