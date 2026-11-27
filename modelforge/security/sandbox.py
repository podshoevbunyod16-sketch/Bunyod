"""Workspace confinement, untrusted-input prompt-injection sanitizer & restricted runner."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


class WorkspaceSandboxError(PermissionError):
    """Raised when an operation attempts to escape the permitted project workspace."""


def resolve_within_workspace(workspace_root: Path, candidate_path: str | Path) -> Path:
    """Resolve `candidate_path` and guarantee it stays strictly inside `workspace_root`."""
    root_resolved = workspace_root.resolve()
    cand = Path(candidate_path)
    if not cand.is_absolute():
        cand = root_resolved / cand
    resolved = cand.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise WorkspaceSandboxError(
            f"Path '{candidate_path}' resolves outside allowed workspace '{root_resolved}'."
        ) from exc
    return resolved


_INJECTION_PATTERNS = [
    re.compile(r"(?i)\bignore\s+(?:all\s+)?(?:previous|prior|system)\s+instructions\b"),
    re.compile(r"(?i)\b(?:execute|run)\s+(?:shell|bash|cmd|powershell|rm\s+-rf|curl\s+|wget\s+)\b"),
    re.compile(r"(?i)<\s*(?:system|tool_call|admin_override)\s*>"),
]


def sanitize_untrusted_content(source: str, content: str, max_chars: int = 16_000) -> Dict[str, Any]:
    """
    Wrap external web/document/MCP text as untrusted data so agents never treat it as system commands.
    Specification Section 16: Never execute instructions from websites, documents, or MCP responses
    as trusted system commands.
    """
    clipped = content[:max_chars]
    warnings: List[str] = []
    for pat in _INJECTION_PATTERNS:
        if pat.search(clipped):
            warnings.append(f"Detected suspicious instruction-like pattern: {pat.pattern}")
    return {
        "trust_level": "untrusted_external_data",
        "executable": False,
        "source": source,
        "warnings": warnings,
        "content": clipped,
    }


@dataclass
class RestrictedExecutionResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_sec: float
    timed_out: bool
    isolation_level: str  # Explicitly "restricted_subprocess" (not claimed as full kernel VM sandbox)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _posix_limit_setter(max_cpu_seconds: int, max_memory_mb: int):
    def _apply() -> None:
        try:
            import resource

            cpu_lim = max(1, int(max_cpu_seconds))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_lim, cpu_lim + 1))
            mem_bytes = max(64, int(max_memory_mb)) * 1024 * 1024
            # RLIMIT_AS caps virtual address space; guard against platforms where Python's mmap needs headroom
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except Exception:
            pass

    return _apply


def run_restricted_python(
    script_or_module_args: List[str],
    workspace_dir: Path,
    timeout_sec: int = 30,
    max_memory_mb: int = 1024,
    allow_network: bool = False,
) -> RestrictedExecutionResult:
    """
    Execute Python code inside an isolated working directory with stripped environment,
    wall-clock timeout, and POSIX rlimits where supported.
    Per Specification Section 16: honestly tagged as `restricted_subprocess`, never misrepresented
    as a full kernel/VM container.
    """
    ws = workspace_dir.resolve()
    ws.mkdir(parents=True, exist_ok=True)

    clean_env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": str(ws),
        "TMPDIR": str(ws),
        "MODELFORGE_OFFLINE": "0" if allow_network else "1",
    }
    if os.name == "nt":
        for k in ("SYSTEMROOT", "TEMP", "TMP", "USERPROFILE"):
            if k in os.environ:
                clean_env[k] = os.environ[k]

    preexec = _posix_limit_setter(timeout_sec, max_memory_mb) if os.name == "posix" else None
    cmd = [sys.executable, "-I" if False else "-B", *script_or_module_args]

    t0 = time.monotonic()
    timed_out = False
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(ws),
            env=clean_env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            preexec_fn=preexec,
        )
        exit_code = completed.returncode
        stdout = completed.stdout[-16_000:]
        stderr = completed.stderr[-16_000:]
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = -9
        stdout = (exc.stdout or "")[-8_000:] if isinstance(exc.stdout, str) else ""
        stderr = f"Process exceeded wall-clock timeout ({timeout_sec}s)."

    return RestrictedExecutionResult(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_sec=round(time.monotonic() - t0, 4),
        timed_out=timed_out,
        isolation_level="restricted_subprocess",
    )
