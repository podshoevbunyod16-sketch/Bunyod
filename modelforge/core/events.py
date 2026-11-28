"""Real-time event bus supporting both thread callbacks and async WebSocket subscribers."""

from __future__ import annotations

import asyncio
import secrets
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple


@dataclass
class Event:
    id: str
    topic: str
    payload: Dict[str, Any]
    project_id: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EventBus:
    """Thread-safe event hub for Core, CLI, and FastAPI WebSocket streaming."""

    def __init__(self, max_history: int = 500) -> None:
        self._lock = threading.Lock()
        self._history: Deque[Event] = deque(maxlen=max_history)
        self._sync_subscribers: List[Callable[[Event], None]] = []
        self._async_queues: List[Tuple[asyncio.AbstractEventLoop, asyncio.Queue[Dict[str, Any]]]] = []

    def publish(self, topic: str, payload: Dict[str, Any], project_id: str = "") -> Event:
        evt = Event(
            id=f"evt-{secrets.token_hex(5)}",
            topic=topic,
            payload=payload,
            project_id=project_id,
        )
        with self._lock:
            self._history.append(evt)
            sync_subs = list(self._sync_subscribers)
            async_subs = list(self._async_queues)

        for cb in sync_subs:
            try:
                cb(evt)
            except Exception:
                pass

        evt_dict = evt.to_dict()
        for loop, q in async_subs:
            try:
                loop.call_soon_threadsafe(self._push_nonblocking, q, evt_dict)
            except Exception:
                pass
        return evt

    @staticmethod
    def _push_nonblocking(q: "asyncio.Queue[Dict[str, Any]]", item: Dict[str, Any]) -> None:
        if q.full():
            try:
                q.get_nowait()
            except Exception:
                pass
        try:
            q.put_nowait(item)
        except Exception:
            pass

    def subscribe_sync(self, callback: Callable[[Event], None]) -> Callable[[], None]:
        with self._lock:
            self._sync_subscribers.append(callback)

        def _unsub() -> None:
            with self._lock:
                if callback in self._sync_subscribers:
                    self._sync_subscribers.remove(callback)

        return _unsub

    def subscribe_async(
        self, loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> tuple["asyncio.Queue[Dict[str, Any]]", Callable[[], None]]:
        active_loop = loop or asyncio.get_running_loop()
        q: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=256)
        entry = (active_loop, q)
        with self._lock:
            self._async_queues.append(entry)

        def _unsub() -> None:
            with self._lock:
                if entry in self._async_queues:
                    self._async_queues.remove(entry)

        return q, _unsub

    def recent_events(self, limit: int = 50, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._history)
        if project_id:
            items = [e for e in items if not e.project_id or e.project_id == project_id]
        return [e.to_dict() for e in items[-limit:]]
