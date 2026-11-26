#!/data/data/com.termux/files/usr/bin/bash
# RASM-CIM v0.8 — Termux bench + smoke. Usage: ./termux_run.sh 20000
set -e
N="${1:-20000}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PKG_PARENT="$(dirname "$(dirname "$HERE")")"
export PYTHONPATH="$PKG_PARENT"
cd "$HERE"
echo "== CLI smoke =="
python -m rasm_cim_v8.cli build "$N" "$HERE/bench.rasm"
python -m rasm_cim_v8.cli scan "$HERE/bench.rasm"
python -m rasm_cim_v8.cli get "$HERE/bench.rasm" 12345
python -m rasm_cim_v8.cli noisy "$HERE/bench.rasm" 777
echo "== full mobile bench (start time / RSS / CPU / serial-vs-parallel / thermal) =="
python mobile_bench.py --n "$N"
echo "Result: $HERE/mobile_bench_result.json  (attach this file when reporting a device run)"
