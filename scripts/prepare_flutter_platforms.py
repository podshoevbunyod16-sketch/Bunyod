#!/usr/bin/env python3
"""Cross-platform Flutter scaffold & AndroidManifest configurator (Linux / Windows / macOS)."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FLUTTER_DIR = REPO_ROOT / "apps" / "flutter_app"


def run_cmd(cmd: list[str], cwd: Path) -> None:
    if os.name == "nt":
        exe = shutil.which(f"{cmd[0]}.bat") or shutil.which(f"{cmd[0]}.cmd") or shutil.which(cmd[0]) or cmd[0]
    else:
        exe = shutil.which(cmd[0]) or cmd[0]
    full_cmd = [exe, *cmd[1:]]
    print(f"+ {' '.join(full_cmd)} (cwd={cwd})", flush=True)
    subprocess.run(full_cmd, cwd=str(cwd), check=True)


def patch_android_manifest(manifest_path: Path) -> None:
    if not manifest_path.exists():
        return
    txt = manifest_path.read_text(encoding="utf-8")
    if "android.permission.INTERNET" not in txt:
        txt = txt.replace(
            "<application",
            '<uses-permission android:name="android.permission.INTERNET"/>\n    <application',
            1,
        )
    if "usesCleartextTraffic" not in txt:
        txt = txt.replace(
            "<application",
            '<application\n        android:usesCleartextTraffic="true"',
            1,
        )
    manifest_path.write_text(txt, encoding="utf-8")
    print(f"Patched AndroidManifest.xml: {manifest_path}", flush=True)


def patch_windows_cmake(cmake_path: Path) -> None:
    if not cmake_path.exists():
        return
    txt = cmake_path.read_text(encoding="utf-8")
    if "/WX" in txt:
        txt = txt.replace("/WX", "")
        cmake_path.write_text(txt, encoding="utf-8")
        print(f"Patched Windows CMakeLists.txt (removed /WX): {cmake_path}", flush=True)


def _force_remove_readonly(func, path, _exc_info) -> None:
    try:
        os.chmod(path, 0o700)
        func(path)
    except Exception:
        pass


def main() -> int:
    required_platforms = ("windows",) if os.name == "nt" else ("android", "windows", "linux", "macos")
    need_create = not all((FLUTTER_DIR / p).exists() for p in required_platforms)

    if need_create:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            backup = Path(tmp) / "backup"
            backup.mkdir(parents=True, exist_ok=True)
            for item in ("pubspec.yaml", "analysis_options.yaml", "lib", "test"):
                src = FLUTTER_DIR / item
                if src.exists():
                    dst = backup / item
                    if src.is_dir():
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)

            platforms_args = (
                ["--platforms", "windows"]
                if os.name == "nt"
                else ["--platforms=android,windows,linux,macos"]
            )
            run_cmd(
                [
                    "flutter",
                    "create",
                    "--org",
                    "ai.modelforge",
                    "--project-name",
                    "modelforge_app",
                    *platforms_args,
                    ".",
                ],
                cwd=FLUTTER_DIR,
            )

            # Restore ModelForge custom Dart sources & tests over template defaults
            for item in ("pubspec.yaml", "analysis_options.yaml", "lib", "test"):
                bsrc = backup / item
                tdst = FLUTTER_DIR / item
                if bsrc.exists():
                    if tdst.exists():
                        if tdst.is_dir():
                            shutil.rmtree(tdst, onerror=_force_remove_readonly)
                        else:
                            os.chmod(tdst, 0o700)
                            tdst.unlink()
                    if bsrc.is_dir():
                        shutil.copytree(bsrc, tdst)
                    else:
                        shutil.copy2(bsrc, tdst)

    patch_android_manifest(
        FLUTTER_DIR / "android" / "app" / "src" / "main" / "AndroidManifest.xml"
    )
    patch_windows_cmake(FLUTTER_DIR / "windows" / "CMakeLists.txt")
    run_cmd(["flutter", "pub", "get"], cwd=FLUTTER_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
