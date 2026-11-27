"""Discovery for the rasm_cim_ext PyO3 extension (cross-language tests).

The frozen v0.7 test module (test_cross_language.py, freeze-locked) does a
plain `import rasm_cim_ext` with no discovery logic. Before this conftest
existed, the only way to run it was a manual `PYTHONPATH=<dir-with-rasm_cim_ext.so>`.

This conftest is ADDITIVE (freeze-safe): it
  1. leaves rasm_cim_ext alone if it is already importable;
  2. otherwise honors RASM_CIM_EXT (absolute path to a rasm_cim_ext.so)
     — this is the CI/build hook after `cargo build --release`;
  3. otherwise loads the shipped PREBUILT release extension
     (rasm_cim_rust/bindings/python/prebuilt/rasm_cim_ext-linux-x86_64.so),
     but only on linux-x86_64 (the only artifact shipped), copying it to a
     temp dir under the importable name rasm_cim_ext.so;
  4. otherwise marks the cross-language tests as SKIPPED with an explicit,
     visible reason (never silently passes, never hides a failure).

RASM_CIM_EXT_DISABLE=1 forces the skip path (used to test this hook itself).
The debug .so under benchmarks/dbg/ is deliberately NOT auto-used: Rust
debug builds panic on integer overflow and are not behavior-equivalent for
these equivalence tests.
"""
import importlib
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PREBUILT = os.path.join(ROOT, "rasm_cim_rust", "bindings", "python", "prebuilt",
                        "rasm_cim_ext-linux-x86_64.so")

STATUS, REASON = "ok", None


def _load_from_path(so_path, tmp_root):
    if not os.path.isfile(so_path):
        return False, f"rasm_cim_ext not found at {so_path}"
    tmp = tempfile.mkdtemp(prefix="rasm_cim_ext_", dir=tmp_root)
    dst = os.path.join(tmp, "rasm_cim_ext.so")
    shutil.copy2(so_path, dst)
    if tmp not in sys.path:
        sys.path.insert(0, tmp)
    try:
        importlib.import_module("rasm_cim_ext")
        return True, None
    except Exception as e:  # noqa: BLE001 — report load failure honestly
        return False, f"rasm_cim_ext failed to load from {so_path}: {e}"


def _ensure():
    if os.environ.get("RASM_CIM_EXT_DISABLE") == "1":
        return False, "RASM_CIM_EXT_DISABLE=1 (skip path forced)"
    try:
        importlib.import_module("rasm_cim_ext")
        return True, None
    except Exception:
        pass
    override = os.environ.get("RASM_CIM_EXT")
    if override:
        return _load_from_path(override, tempfile.gettempdir())
    if sys.platform.startswith("linux") and os.uname().machine == "x86_64":
        return _load_from_path(PREBUILT, tempfile.gettempdir())
    return False, (f"no rasm_cim_ext build for platform {sys.platform}/"
                   f"{os.uname().machine}; build with cargo or set RASM_CIM_EXT")


ok, reason = _ensure()
STATUS, REASON = ("ok", None) if ok else ("unavailable", reason)
if STATUS != "ok":
    # The frozen test module does `import rasm_cim_ext` at import time, i.e.
    # during collection — skipping in pytest_collection_modifyitems would be
    # too late. Instead, register a stub module whose attribute access skips
    # the calling test with an explicit reason (visible as SKIPPED, never
    # silently green).
    print(f"[rasm_cim_rust/tests/conftest] cross-language tests will SKIP: {REASON}",
          file=sys.stderr)
    import types

    class _MissingExtension(types.ModuleType):
        __version__ = "unavailable"

        def __getattr__(self, name):
            import pytest
            pytest.skip(f"rasm_cim_ext unavailable: {REASON}")

    sys.modules["rasm_cim_ext"] = _MissingExtension("rasm_cim_ext")
