#!/usr/bin/env python3
"""Build & smoke-test standalone `modelforge-backend` executable via PyInstaller for Desktop bundles."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    out_dir = REPO_ROOT / "release_artifacts" / "backend_bundle"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_sep = ";" if os.name == "nt" else ":"
    plugins_arg = f"{REPO_ROOT / 'plugins'}{data_sep}plugins"

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--name",
        "modelforge-backend",
        "--distpath",
        str(out_dir),
        "--add-data",
        plugins_arg,
        "--hidden-import",
        "modelforge.integrations.builtin_mcp_server",
        "--hidden-import",
        "uvicorn.logging",
        "--hidden-import",
        "uvicorn.loops.auto",
        "--hidden-import",
        "uvicorn.protocols.http.auto",
        "--hidden-import",
        "uvicorn.protocols.websockets.auto",
        "--hidden-import",
        "uvicorn.lifespan.on",
        str(REPO_ROOT / "modelforge" / "__main__.py"),
    ]
    print("+ Running PyInstaller:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)

    exe_name = "modelforge-backend.exe" if os.name == "nt" else "modelforge-backend"
    exe_path = out_dir / "modelforge-backend" / exe_name
    if not exe_path.exists():
        raise FileNotFoundError(f"Expected bundled backend binary not found: {exe_path}")

    # Smoke-test the compiled binary on an isolated temp home directory
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_home:
        proc = subprocess.run(
            [str(exe_path), "--home", tmp_home, "--json", "doctor"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=True,
        )
        doc = json.loads(proc.stdout)
        assert doc["healthy"] is True, doc
        print(f"Verified standalone backend binary: {exe_path} (healthy={doc['healthy']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
