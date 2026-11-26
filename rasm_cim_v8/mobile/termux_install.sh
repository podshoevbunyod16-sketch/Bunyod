#!/data/data/com.termux/files/usr/bin/bash
# RASM-CIM v0.8 — Termux install (Android). Core is STDLIB-ONLY; numpy is
# optional (faster dense variants in M5) and NOT required for the mobile path.
set -e
pkg update -y
pkg install -y python
python --version
# no pip packages are required for build/stream/exact/noisy paths
HERE="$(cd "$(dirname "$0")" && pwd)"
PKG_PARENT="$(dirname "$(dirname "$HERE")")"
echo "Package root expected at: $PKG_PARENT  (must contain rasm_cim_v8/)"
PYTHONPATH="$PKG_PARENT" python -m rasm_cim_v8.cli selftest && echo "SELFTEST OK"
echo "Install complete. Run:  ./termux_run.sh [n]"
