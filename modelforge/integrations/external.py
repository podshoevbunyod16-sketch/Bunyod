"""GitHub repository & Composio external tool adapters with explicit permissions & Offline guard."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict

from modelforge.security.permissions import PermissionDeniedError
from modelforge.security.secrets import SecretManager


class GitHubIntegration:
    """Inspects local Git & GitHub status; destructive/publish actions require explicit user confirmation."""

    def __init__(self, repo_root: Path, offline_getter: Callable[[], bool] = lambda: False) -> None:
        self.repo_root = repo_root
        self._offline_getter = offline_getter

    def status(self) -> Dict[str, Any]:
        git_bin = shutil.which("git")
        gh_bin = shutil.which("gh")
        branch = "unknown"
        commit = "unknown"
        if git_bin and (self.repo_root / ".git").exists():
            try:
                branch = subprocess.check_output(
                    [git_bin, "rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=str(self.repo_root),
                    text=True,
                ).strip()
                commit = subprocess.check_output(
                    [git_bin, "rev-parse", "--short", "HEAD"],
                    cwd=str(self.repo_root),
                    text=True,
                ).strip()
            except Exception:
                pass
        return {
            "git_available": bool(git_bin),
            "gh_cli_available": bool(gh_bin),
            "branch": branch,
            "head_commit": commit,
            "offline_mode": self._offline_getter(),
        }


class ComposioAdapter:
    """
    Optional Composio external-integrations adapter.
    Honestly reports availability and requires explicit API key + network permission.
    """

    def __init__(self, secrets_mgr: SecretManager, offline_getter: Callable[[], bool] = lambda: False) -> None:
        self._secrets = secrets_mgr
        self._offline_getter = offline_getter

    def status(self) -> Dict[str, Any]:
        if self._offline_getter():
            return {"available": False, "reason": "Blocked in Offline Mode."}
        has_pkg = False
        try:
            import composio  # type: ignore  # noqa: F401

            has_pkg = True
        except ImportError:
            pass
        has_key = bool(self._secrets.get_secret("COMPOSIO_API_KEY"))
        return {
            "package_installed": has_pkg,
            "api_key_configured": has_key,
            "status": "ready" if (has_pkg and has_key) else "optional_not_configured",
            "note": "External SaaS actions via Composio require COMPOSIO_API_KEY and explicit agent permission.",
        }

    def execute_action(self, action_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if self._offline_getter():
            raise PermissionDeniedError("Composio actions are disabled in Offline Mode.")
        st = self.status()
        if st["status"] != "ready":
            raise RuntimeError(
                "NOT CONFIGURED: Install `composio-core` and store `COMPOSIO_API_KEY` in ModelForge Secret Vault."
            )
        import composio  # type: ignore  # noqa: F401

        raise RuntimeError(f"Action '{action_name}' requires explicit user OAuth connection.")
