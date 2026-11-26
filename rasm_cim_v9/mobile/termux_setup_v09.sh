#!/data/data/com.termux/files/usr/bin/bash
# v0.9 M4 — one-time Termux setup. Run AFTER unpacking the v0.1–v0.9 archive
# into ~/rasm_cim_v9_pkg (keeps v0.8 for the A/B side by side — required).
set -eu
command -v python3 >/dev/null || pkg install -y python
python3 - << 'PY'
import sys
assert sys.version_info >= (3, 10), f"need 3.10+, got {sys.version.split()[0]}"
print("python", sys.version.split()[0], "OK  (numpy NOT required; optional speedup)")
PY
echo "now run:  RASM_PKG=$HOME/rasm_cim_v9_pkg bash $HOME/rasm_cim_v9_pkg/rasm_cim_v9/mobile/termux_run_v09.sh"
