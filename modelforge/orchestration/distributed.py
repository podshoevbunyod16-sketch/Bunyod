"""Remote & Hybrid Mode client + extensible distributed worker-node registry (Sections 1 & 18)."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from modelforge.security.permissions import PermissionDeniedError


@dataclass
class WorkerNode:
    id: str
    name: str
    endpoint_url: str
    architecture: str = "x86_64"
    ram_mb: int = 8192
    has_gpu: bool = False
    is_paid_cloud: bool = False
    status: str = "registered"
    last_checked_at: float = 0.0
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RemoteExecutorClient:
    """Connects Local/Hybrid GUI or CLI to a remote ModelForge Core server with clean offline/unreachable handling."""

    def __init__(self, base_url: str, timeout_sec: float = 8.0, offline_mode: bool = False) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.offline_mode = offline_mode

    def check_health(self) -> Dict[str, Any]:
        if self.offline_mode:
            raise PermissionDeniedError("Remote server connection is disabled in Offline Mode.")
        if not self.base_url:
            return {"reachable": False, "error": "No remote_server_url configured."}
        try:
            import httpx

            t0 = time.monotonic()
            with httpx.Client(timeout=self.timeout_sec) as client:
                resp = client.get(f"{self.base_url}/api/v1/health")
                latency_ms = round((time.monotonic() - t0) * 1000.0, 2)
                if resp.status_code == 200:
                    return {
                        "reachable": True,
                        "latency_ms": latency_ms,
                        "remote_info": resp.json(),
                    }
                return {
                    "reachable": False,
                    "status_code": resp.status_code,
                    "error": f"Remote returned HTTP {resp.status_code}",
                }
        except Exception as exc:
            return {
                "reachable": False,
                "error": f"{type(exc).__name__}: {exc}",
            }


class WorkerRegistry:
    """Extensible registry for user PC / server worker nodes (Hybrid & future Distributed Compute)."""

    def __init__(self) -> None:
        self._workers: Dict[str, WorkerNode] = {}

    def register(self, worker: WorkerNode) -> WorkerNode:
        self._workers[worker.id] = worker
        return worker

    def list_workers(self) -> List[Dict[str, Any]]:
        return [w.to_dict() for w in self._workers.values()]

    def probe_worker(self, worker_id: str, offline_mode: bool = False) -> Dict[str, Any]:
        w = self._workers.get(worker_id)
        if not w:
            raise KeyError(f"Unknown worker node: {worker_id}")
        client = RemoteExecutorClient(w.endpoint_url, timeout_sec=5.0, offline_mode=offline_mode)
        res = client.check_health()
        w.last_checked_at = time.time()
        w.status = "online" if res.get("reachable") else "unreachable"
        return {"worker": w.to_dict(), "probe": res}
