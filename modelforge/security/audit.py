"""Tamper-evident hash-chained security & action audit log."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from modelforge.security.secrets import redact_secrets


class AuditLogger:
    """Append-only JSONL audit log with SHA-256 hash chaining and automatic secret redaction."""

    def __init__(self, log_path: Path) -> None:
        self._log_path = log_path
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last_hash = self._read_last_hash()

    def _read_last_hash(self) -> str:
        if not self._log_path.exists():
            return "0" * 64
        try:
            lines = [ln for ln in self._log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if not lines:
                return "0" * 64
            last = json.loads(lines[-1])
            return str(last.get("entry_hash", "0" * 64))
        except Exception:
            return "0" * 64

    def record(
        self,
        actor: str,
        action: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
        extra_secrets: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            safe_details_raw = json.dumps(details or {}, ensure_ascii=False, sort_keys=True)
            safe_details_str = redact_secrets(safe_details_raw, extra_secrets)
            safe_details = json.loads(safe_details_str)
            ts = round(time.time(), 6)
            payload = {
                "timestamp": ts,
                "actor": actor,
                "action": action,
                "status": status,
                "details": safe_details,
                "prev_hash": self._last_hash,
            }
            canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            entry_hash = hashlib.sha256(canonical).hexdigest()
            payload["entry_hash"] = entry_hash
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._last_hash = entry_hash
            return payload

    def list_entries(self, limit: int = 100) -> List[Dict[str, Any]]:
        if not self._log_path.exists():
            return []
        lines = [ln for ln in self._log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        selected = lines[-limit:] if limit > 0 else lines
        return [json.loads(ln) for ln in selected]

    def verify_integrity(self) -> tuple[bool, int, Optional[str]]:
        """Verify SHA-256 chain across all recorded audit entries."""
        if not self._log_path.exists():
            return True, 0, None
        prev = "0" * 64
        count = 0
        for idx, line in enumerate(self._log_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            entry = json.loads(line)
            recorded_hash = entry.pop("entry_hash", None)
            if entry.get("prev_hash") != prev:
                return False, count, f"Broken prev_hash link at line {idx}"
            canonical = json.dumps(entry, ensure_ascii=False, sort_keys=True).encode("utf-8")
            expected = hashlib.sha256(canonical).hexdigest()
            if recorded_hash != expected:
                return False, count, f"Tampered entry hash at line {idx}"
            prev = recorded_hash
            count += 1
        return True, count, None
