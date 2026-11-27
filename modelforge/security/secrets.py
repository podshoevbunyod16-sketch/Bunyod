"""Secret vault (OS keyring when available + 0600 local vault fallback) & log redaction."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Dict, Iterable, List, Optional

_SECRET_PATTERNS = [
    re.compile(r"(sk-[A-Za-z0-9_\-]{16,})"),
    re.compile(r"(ghp_[A-Za-z0-9]{20,})"),
    re.compile(r"(github_pat_[A-Za-z0-9_]{20,})"),
    re.compile(r"(Bearer\s+[A-Za-z0-9\-._~+/]+=*)", re.IGNORECASE),
    re.compile(r'("?(?:api_key|apikey|secret|token|password)"?\s*[:=]\s*"?)([^"\s,}]{6,})', re.IGNORECASE),
]


def redact_secrets(text: str, extra_secrets: Optional[Iterable[str]] = None) -> str:
    """Scrub known secret values and API-key patterns from logs, errors, and configs."""
    if not text:
        return text
    result = text
    if extra_secrets:
        for val in extra_secrets:
            if val and len(val) >= 5:
                result = result.replace(val, "***REDACTED***")
    for idx, pat in enumerate(_SECRET_PATTERNS):
        if idx == 4:
            result = pat.sub(r"\1***REDACTED***", result)
        else:
            result = pat.sub("***REDACTED***", result)
    return result


class SecretManager:
    """Stores provider keys out of plain configs & Git; uses OS keyring if available, else 0600 vault."""

    SERVICE_NAME = "modelforge-ai"

    def __init__(self, home_dir: Path) -> None:
        self._home_dir = home_dir
        self._vault_path = home_dir / "security" / "secrets.vault.json"
        self._vault_path.parent.mkdir(parents=True, exist_ok=True)
        self._keyring = self._probe_keyring()
        self._known_values: set[str] = set()

    @staticmethod
    def _probe_keyring():
        try:
            import keyring  # type: ignore
            from keyring.errors import KeyringError  # type: ignore

            # Verify backend is actually usable (headless Linux often has null/fail keyring)
            backend = keyring.get_keyring()
            if "fail" in type(backend).__name__.lower() or "null" in type(backend).__name__.lower():
                return None
            return keyring
        except Exception:
            return None

    def _machine_mask(self) -> bytes:
        seed = f"{os.getuid() if hasattr(os, 'getuid') else 0}:{self._home_dir.resolve()}".encode("utf-8")
        return hashlib.sha256(seed).digest()

    def _obfuscate(self, value: str) -> str:
        raw = value.encode("utf-8")
        mask = self._machine_mask()
        xored = bytes(b ^ mask[i % len(mask)] for i, b in enumerate(raw))
        return base64.urlsafe_b64encode(xored).decode("ascii")

    def _deobfuscate(self, encoded: str) -> str:
        xored = base64.urlsafe_b64decode(encoded.encode("ascii"))
        mask = self._machine_mask()
        raw = bytes(b ^ mask[i % len(mask)] for i, b in enumerate(xored))
        return raw.decode("utf-8")

    def _load_vault(self) -> Dict[str, str]:
        if not self._vault_path.exists():
            return {}
        try:
            return json.loads(self._vault_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_vault(self, data: Dict[str, str]) -> None:
        self._vault_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        try:
            os.chmod(self._vault_path, stat.S_IRUSR | stat.S_IWUSR)
        except Exception:
            pass

    @property
    def storage_backend(self) -> str:
        return "os_keyring" if self._keyring is not None else "local_0600_vault"

    def set_secret(self, name: str, value: str) -> None:
        clean_name = name.strip()
        if not clean_name or not value:
            raise ValueError("Secret name and value must be non-empty.")
        self._known_values.add(value)
        if self._keyring is not None:
            try:
                self._keyring.set_password(self.SERVICE_NAME, clean_name, value)
                vault = self._load_vault()
                vault[clean_name] = "__KEYRING__"
                self._save_vault(vault)
                return
            except Exception:
                pass
        vault = self._load_vault()
        vault[clean_name] = self._obfuscate(value)
        self._save_vault(vault)

    def get_secret(self, name: str) -> Optional[str]:
        clean_name = name.strip()
        env_val = os.environ.get(clean_name) or os.environ.get(f"MODELFORGE_{clean_name.upper()}")
        if env_val:
            self._known_values.add(env_val)
            return env_val
        vault = self._load_vault()
        stored = vault.get(clean_name)
        if stored == "__KEYRING__" and self._keyring is not None:
            try:
                val = self._keyring.get_password(self.SERVICE_NAME, clean_name)
                if val:
                    self._known_values.add(val)
                return val
            except Exception:
                return None
        if stored and stored != "__KEYRING__":
            try:
                val = self._deobfuscate(stored)
                self._known_values.add(val)
                return val
            except Exception:
                return None
        return None

    def delete_secret(self, name: str) -> bool:
        clean_name = name.strip()
        vault = self._load_vault()
        existed = clean_name in vault
        if existed:
            if vault[clean_name] == "__KEYRING__" and self._keyring is not None:
                try:
                    self._keyring.delete_password(self.SERVICE_NAME, clean_name)
                except Exception:
                    pass
            del vault[clean_name]
            self._save_vault(vault)
        return existed

    def list_secret_names(self) -> List[str]:
        return sorted(self._load_vault().keys())

    def redact(self, text: str) -> str:
        return redact_secrets(text, self._known_values)
