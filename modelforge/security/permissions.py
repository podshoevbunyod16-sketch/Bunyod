"""Capabilities, explicit permissions & confirmation gates for high-impact actions."""

from __future__ import annotations

import secrets
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class PermissionScope(str, Enum):
    WORKSPACE_READ = "workspace:read"
    WORKSPACE_WRITE = "workspace:write"
    WORKSPACE_DELETE = "workspace:delete"
    CODE_EXEC_RESTRICTED = "code:exec_restricted"
    MODEL_TRAIN = "model:train"
    MODEL_INFER = "model:infer"
    MEMORY_READ = "memory:read"
    MEMORY_WRITE = "memory:write"
    NETWORK_READ = "network:read"
    MCP_CALL = "mcp:call"
    PLUGIN_RUN = "plugin:run"
    PACKAGE_INSTALL = "system:package_install"
    PAID_COMPUTE = "compute:paid"
    PUBLISH_EXTERNAL = "external:publish"


# High-impact actions that ALWAYS require explicit user confirmation unless confirmed via token.
HIGH_IMPACT_SCOPES: Set[PermissionScope] = {
    PermissionScope.WORKSPACE_DELETE,
    PermissionScope.PACKAGE_INSTALL,
    PermissionScope.PAID_COMPUTE,
    PermissionScope.PUBLISH_EXTERNAL,
}


class PermissionDeniedError(PermissionError):
    """Raised when an agent, plugin, or tool lacks a required permission."""


class ConfirmationRequiredError(PermissionError):
    """Raised when a dangerous, external, destructive, or paid operation awaits user confirmation."""

    def __init__(self, message: str, request_id: str, action: str, details: Dict[str, Any]):
        super().__init__(message)
        self.request_id = request_id
        self.action = action
        self.details = details


@dataclass
class PermissionPolicy:
    subject_id: str
    subject_type: str  # agent | plugin | mcp_server | user_task
    allowed_scopes: Set[str] = field(
        default_factory=lambda: {
            PermissionScope.WORKSPACE_READ.value,
            PermissionScope.MEMORY_READ.value,
        }
    )
    allowed_workspace_subdir: Optional[str] = None
    allow_network: bool = False
    require_confirmation_for: Set[str] = field(
        default_factory=lambda: {s.value for s in HIGH_IMPACT_SCOPES}
    )

    def check(self, scope: PermissionScope | str, offline_mode: bool = False) -> None:
        scope_val = scope.value if isinstance(scope, PermissionScope) else str(scope)
        if offline_mode and scope_val in (
            PermissionScope.NETWORK_READ.value,
            PermissionScope.PAID_COMPUTE.value,
            PermissionScope.PUBLISH_EXTERNAL.value,
        ):
            raise PermissionDeniedError(
                f"Action '{scope_val}' is blocked because ModelForge is in Offline Mode."
            )
        if scope_val == PermissionScope.NETWORK_READ.value and not self.allow_network:
            raise PermissionDeniedError(
                f"Subject '{self.subject_id}' ({self.subject_type}) does not have network permission."
            )
        if scope_val not in self.allowed_scopes:
            raise PermissionDeniedError(
                f"Subject '{self.subject_id}' ({self.subject_type}) lacks permission '{scope_val}'."
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "subject_type": self.subject_type,
            "allowed_scopes": sorted(self.allowed_scopes),
            "allowed_workspace_subdir": self.allowed_workspace_subdir,
            "allow_network": self.allow_network,
            "require_confirmation_for": sorted(self.require_confirmation_for),
        }


@dataclass
class PendingConfirmation:
    request_id: str
    action: str
    scope: str
    subject_id: str
    summary: str
    details: Dict[str, Any]
    created_at: float
    expires_at: float
    status: str = "pending"  # pending | approved | rejected | consumed
    approval_token: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("approval_token", None)
        return d


class ConfirmationManager:
    """Manages explicit user confirmations for destructive, paid, or external actions."""

    def __init__(self, ttl_seconds: int = 900) -> None:
        self._ttl = ttl_seconds
        self._requests: Dict[str, PendingConfirmation] = {}

    def require_or_verify(
        self,
        action: str,
        scope: PermissionScope | str,
        subject_id: str,
        summary: str,
        details: Optional[Dict[str, Any]] = None,
        confirmation_token: Optional[str] = None,
    ) -> None:
        scope_val = scope.value if isinstance(scope, PermissionScope) else str(scope)
        now = time.time()

        if confirmation_token:
            for req in self._requests.values():
                if (
                    req.status == "approved"
                    and req.approval_token == confirmation_token
                    and req.action == action
                    and req.expires_at >= now
                ):
                    req.status = "consumed"
                    return
            raise PermissionDeniedError("Invalid, expired, or mismatched confirmation token.")

        req_id = f"confirm-{secrets.token_hex(6)}"
        pending = PendingConfirmation(
            request_id=req_id,
            action=action,
            scope=scope_val,
            subject_id=subject_id,
            summary=summary,
            details=details or {},
            created_at=now,
            expires_at=now + self._ttl,
        )
        self._requests[req_id] = pending
        raise ConfirmationRequiredError(
            f"Confirmation required for '{action}': {summary} (request_id={req_id})",
            request_id=req_id,
            action=action,
            details=pending.to_dict(),
        )

    def approve(self, request_id: str) -> str:
        req = self._requests.get(request_id)
        if not req:
            raise KeyError(f"Unknown confirmation request: {request_id}")
        if time.time() > req.expires_at:
            req.status = "expired"
            raise PermissionDeniedError(f"Confirmation request {request_id} has expired.")
        token = f"tok-{secrets.token_urlsafe(18)}"
        req.status = "approved"
        req.approval_token = token
        return token

    def reject(self, request_id: str) -> None:
        req = self._requests.get(request_id)
        if not req:
            raise KeyError(f"Unknown confirmation request: {request_id}")
        req.status = "rejected"

    def list_pending(self) -> List[Dict[str, Any]]:
        now = time.time()
        return [
            r.to_dict()
            for r in self._requests.values()
            if r.status == "pending" and r.expires_at >= now
        ]
