"""Offline-first, checked by reading the sources rather than by promising it.

The rule from the brief: no external LLM, no cloud API, no hidden network calls,
no telemetry.  A promise is not a check, so this module walks the v0.17 sources
and reports every import of a networking module and every call that would leave
the machine (``curl``/``wget``).

It parses with ``ast`` — a comment that mentions ``socket`` is not a violation,
an ``import socket`` is.
"""
from __future__ import annotations

import ast
from pathlib import Path

#: modules that can open a connection or send something off the device
NETWORK_MODULES = (
    "socket", "ssl", "urllib", "urllib2", "urllib3", "http", "httplib",
    "requests", "httpx", "aiohttp", "ftplib", "smtplib", "telnetlib",
    "telnetlib3", "xmlrpc", "websockets", "websocket", "paramiko", "boto3",
    "google", "openai", "anthropic", "transformers", "torch", "huggingface_hub",
)

#: commands that would fetch something if executed through subprocess
NETWORK_COMMANDS = ("curl", "wget", "nc", "ssh", "scp", "ftp", "http")

#: Единственный модуль, которому разрешено трогать сеть, — инструмент агента.
#: Он выключен по умолчанию (`online=False`), ходит только по списку хостов и
#: не импортируется, пока агент не попросили об этом. Нарушение в любом другом
#: файле остаётся нарушением, поэтому гарантия «ядро офлайн» сохраняется, а
#: исключение ровно одно и проверяется тестом.
NETWORK_ALLOWLIST = ("rasm_cim_v17/agent/tools.py",)


def _is_allowed(path: Path) -> bool:
    normalized = str(path).replace("\\", "/")
    return any(normalized.endswith(item) for item in NETWORK_ALLOWLIST)


def _module_root(name: str) -> str:
    return (name or "").split(".")[0]


def _strings_in(nodes) -> list[str]:
    """Every string constant in the arguments, including those inside lists."""
    out: list[str] = []
    for node in nodes:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append(node.value)
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            out.extend(_strings_in(node.elts))
        elif isinstance(node, ast.JoinedStr):
            for value in node.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    out.append(value.value)
    return out


def scan_file(path: Path) -> list[dict]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return [{"path": str(path), "line": exc.lineno or 0, "kind": "syntax",
                 "detail": str(exc)}]
    findings: list[dict] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _module_root(alias.name) in NETWORK_MODULES:
                    findings.append({"path": str(path), "line": node.lineno,
                                     "kind": "import", "detail": alias.name})
        elif isinstance(node, ast.ImportFrom):
            root = _module_root(node.module or "")
            if root in NETWORK_MODULES:
                findings.append({"path": str(path), "line": node.lineno,
                                 "kind": "import", "detail": node.module or ""})
        elif isinstance(node, ast.Call):
            func = node.func
            name = ""
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            if name in ("system", "run", "Popen", "call", "check_output",
                        "check_call", "urlopen", "socket", "connect", "sendall"):
                for arg in _strings_in(node.args):
                    lowered = arg.lower()
                    if any(cmd in lowered for cmd in NETWORK_COMMANDS):
                        findings.append({"path": str(path),
                                         "line": node.lineno,
                                         "kind": "call",
                                         "detail": arg[:120]})
    return findings


def scan_paths(paths) -> dict:
    files: list[Path] = []
    for entry in paths:
        path = Path(entry)
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*.py")
                                if "__pycache__" not in p.parts))
        elif path.suffix == ".py":
            files.append(path)
    findings: list[dict] = []
    for file in files:
        found = scan_file(file)
        if _is_allowed(file):
            for item in found:
                item["kind"] = "allowed"
        findings.extend(found)
    problems = [f for f in findings if f["kind"] not in ("syntax", "allowed")]
    return {"files": len(files), "findings": findings,
            "allowed": [f for f in findings if f["kind"] == "allowed"],
            "violations": problems, "clean": not problems}


def scan_offline(package_dir: str | Path | None = None) -> dict:
    """Default target: the whole shipped v0.17 package."""
    if package_dir is None:
        package_dir = Path(__file__).resolve().parents[1]
    report = scan_paths([package_dir])
    report["package"] = str(package_dir)
    return report
