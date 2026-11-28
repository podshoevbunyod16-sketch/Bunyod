"""Unified single-source-of-truth runtime state manager."""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional


class SystemStateManager:
    """Thread-safe single source of runtime state across components."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._started_at = time.time()
        self._active_project_id: Optional[str] = None
        self._counters: Dict[str, int] = {
            "models_trained": 0,
            "inferences_run": 0,
            "agent_runs": 0,
            "prompt_edits": 0,
            "experiments_run": 0,
        }
        self._component_status: Dict[str, str] = {
            "core": "ready",
            "storage": "ready",
            "task_manager": "ready",
            "model_runtime": "ready",
            "agent_runtime": "ready",
            "research_engine": "ready",
            "mcp_manager": "ready",
        }

    def set_active_project(self, project_id: Optional[str]) -> None:
        with self._lock:
            self._active_project_id = project_id

    @property
    def active_project_id(self) -> Optional[str]:
        with self._lock:
            return self._active_project_id

    def increment(self, key: str, delta: int = 1) -> None:
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + delta

    def set_component_status(self, component: str, status: str) -> None:
        with self._lock:
            self._component_status[component] = status

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "uptime_sec": round(time.time() - self._started_at, 2),
                "active_project_id": self._active_project_id,
                "counters": dict(self._counters),
                "components": dict(self._component_status),
            }
