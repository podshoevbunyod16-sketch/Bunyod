"""Spec §13 — portability: stdlib only, no hardcoded paths, ARM64/Termux
friendly, no network, Python >= 3.10 syntax only.

These are static checks that can actually be run in CI.  The device run
itself (a real phone under Termux) is reported separately in
V016_TEST_REPORT.md and is NOT claimed by anything here.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys

import pytest

PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "rasm_cim_v16")

#: modules the runtime may import.  Anything else is a portability bug.
STDLIB_OK = {
    "argparse", "ast", "collections", "dataclasses", "gc", "hashlib",
    "itertools", "json", "os", "re", "resource", "struct", "sys", "tempfile",
    "time", "typing", "unicodedata", "bisect", "array", "math", "random",
    "subprocess", "__future__",
}
#: first-party packages inside this repo
LOCAL_OK = {"rasm_cim_v16", "rasm_cim_v9", "rasm_cim_v8"}


def _modules():
    out = []
    for root, _dirs, files in os.walk(PKG):
        for f in files:
            if f.endswith(".py"):
                out.append(os.path.join(root, f))
    return sorted(out)


def test_package_is_found():
    assert _modules(), "no modules found under rasm_cim_v16"


@pytest.mark.parametrize("path", _modules(), ids=lambda p: os.path.basename(p))
def test_only_stdlib_and_local_imports(path):
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:                     # relative import -> local
                continue
            names = [(node.module or "").split(".")[0]]
        else:
            continue
        for n in names:
            if n and n not in STDLIB_OK and n not in LOCAL_OK:
                bad.append(n)
    assert not bad, f"{os.path.basename(path)} imports third-party {sorted(set(bad))}"


@pytest.mark.parametrize("path", _modules(), ids=lambda p: os.path.basename(p))
def test_no_hardcoded_absolute_paths(path):
    src = open(path, encoding="utf-8").read()
    hits = re.findall(r'"/(?:home|Users|tmp|rasm)[^"]*"', src)
    hits = [h for h in hits if not h.startswith('"/tmp-"')]
    assert not hits, f"{os.path.basename(path)} hardcodes {hits}"


@pytest.mark.parametrize("path", _modules(), ids=lambda p: os.path.basename(p))
def test_no_network_usage(path):
    src = open(path, encoding="utf-8").read()
    for mod in ("urllib", "requests", "socket", "http.client", "ftplib",
                "smtplib", "openai", "anthropic", "torch", "transformers"):
        assert mod not in src, f"{os.path.basename(path)} references {mod}"


@pytest.mark.parametrize("path", _modules(), ids=lambda p: os.path.basename(p))
def test_python_310_syntax(path):
    """Compile with the oldest supported feature set we can check here.

    A full 3.10/3.14 run is a separate job (see V016_TEST_REPORT.md); this at
    least proves there is no newer-than-3.10 syntax and no removed API in the
    AST-level surface we use.
    """
    src = open(path, encoding="utf-8").read()
    ast.parse(src, filename=path, feature_version=(3, 10))
    compile(src, path, "exec")


def test_no_removed_or_deprecated_apis():
    banned = ("distutils", "imp.", "asyncio.coroutine", "collections.Mapping",
              "collections.MutableMapping", "@asyncio.coroutine",
              "time.clock(", "locale.getdefaultlocale")
    for path in _modules():
        src = open(path, encoding="utf-8").read()
        for b in banned:
            assert b not in src, f"{os.path.basename(path)} uses removed API {b}"


def test_runtime_imports_cleanly_in_a_subprocess():
    code = ("import sys; sys.path.insert(0, %r);"
            "import rasm_cim_v16, rasm_cim_v16.cli;"
            "print(rasm_cim_v16.__version__)") % os.path.dirname(PKG)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().startswith("0.16")


def test_cli_help_runs():
    r = subprocess.run([sys.executable, "-m", "rasm_cim_v16", "--help"],
                       capture_output=True, text=True, timeout=60,
                       cwd=os.path.dirname(PKG))
    assert r.returncode == 0, r.stderr
    for cmd in ("chat", "ask", "teach", "say", "migrate", "rollback", "demo"):
        assert cmd in r.stdout


def test_cli_demo_runs_end_to_end(tmp_path):
    r = subprocess.run([sys.executable, "-m", "rasm_cim_v16", "demo",
                        "--home", str(tmp_path / ".rasm-cim"),
                        "--answer", "normal"],
                       capture_output=True, text=True, timeout=300,
                       cwd=os.path.dirname(PKG))
    assert r.returncode == 0, r.stderr[-2000:]
    assert "Записал:" in r.stdout
    assert "CONTRADICTED" in r.stdout or "Противоречие" in r.stdout


def test_cli_single_question(tmp_path):
    home = str(tmp_path / ".rasm-cim")
    env = dict(os.environ, RASM_CIM_HOME=home)
    base = [sys.executable, "-m", "rasm_cim_v16"]
    cwd = os.path.dirname(PKG)
    r1 = subprocess.run(base + ["teach", "Исаак Ньютон — это физик"],
                        capture_output=True, text=True, env=env, cwd=cwd,
                        timeout=120)
    assert r1.returncode == 0, r1.stderr
    r2 = subprocess.run(base + ["ask", "Исаак Ньютон — физик?"],
                        capture_output=True, text=True, env=env, cwd=cwd,
                        timeout=120)
    assert r2.returncode == 0, r2.stderr
    assert r2.stdout.startswith("Да"), r2.stdout


def test_cli_rejects_an_unknown_answer_mode(tmp_path):
    r = subprocess.run([sys.executable, "-m", "rasm_cim_v16", "--answer",
                        "banana", "ask", "привет"],
                       capture_output=True, text=True, timeout=60,
                       cwd=os.path.dirname(PKG))
    assert r.returncode == 2
    assert "Неизвестный режим" in r.stderr


def _imported_roots(path):
    """Top-level package names a module imports (AST, not substring matching:
    the string "rasm_cim" is a prefix of "rasm_cim_v16", so grepping the source
    would report v0.16 importing itself)."""
    import ast
    tree = ast.parse(open(path, encoding="utf-8").read())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_frozen_versions_are_untouched():
    """v0.16 must be additive: it imports nothing from the frozen trees."""
    frozen = {"rasm_cim", "rasm_cim_v1_snapshot", "rasm_cim_v2", "rasm_cim_v3",
              "rasm_cim_v4", "rasm_cim_v5", "rasm_cim_v6", "rasm_cim_v6_v2",
              "rasm_cim_rust"}
    offenders = {}
    for path in _modules():
        hit = _imported_roots(path) & frozen
        if hit:
            offenders[os.path.basename(path)] = sorted(hit)
    assert not offenders, offenders


def test_v16_imports_only_v9_and_itself():
    """The one sanctioned dependency is the live v0.9 substrate; nothing else
    from the historical trees may be pulled in."""
    allowed = {"rasm_cim_v9", "rasm_cim_v16"}
    external = set()
    for path in _modules():
        for root in _imported_roots(path):
            if root.startswith("rasm_cim") and root not in allowed:
                external.add((os.path.basename(path), root))
    assert not external, external


def test_cli_inspect_subcommands_are_distinct(tmp_path):
    """Every `:cmd` must be reachable from the CLI and produce its own output.

    Regression: the subparser pinned the handler name into the namespace, so
    `inspect :entities` (and `graph <name>`) silently printed `:inspect`.
    """
    import os
    import subprocess
    import sys
    home = str(tmp_path)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    py = sys.executable
    for t in ["Исаак Ньютон — это физик", "Ньютон — это Исаак Ньютон",
              "Исаак Ньютон открыл закон всемирного тяготения"]:
        subprocess.run([py, "-m", "rasm_cim_v16", "--home", home, "teach", t],
                       check=True, capture_output=True, text=True, cwd=root)

    def run(*argv):
        r = subprocess.run([py, "-m", "rasm_cim_v16", "--home", home, *argv],
                           capture_output=True, text=True, cwd=root, timeout=120)
        assert r.returncode == 0, r.stderr
        return r.stdout

    outs = {}
    for cmd in ("inspect", "entities", "graph", "memory", "rules", "context",
                "conflicts", "evidence", "explain", "help", "stats"):
        outs[cmd] = run(cmd)
    assert len(set(outs.values())) == len(outs), "commands must differ"
    assert "вид=PERSON" in outs["entities"]
    # `inspect :entities` and `entities` are the same command
    assert outs["entities"] == run("inspect", ":entities")
    # a command argument reaches the handler
    assert "st-000001" in run("graph", "Исаак Ньютон")
    assert "st-000001" not in run("graph", "Ньютон")
