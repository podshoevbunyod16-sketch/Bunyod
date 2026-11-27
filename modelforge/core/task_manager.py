"""Background & synchronous task manager with progress, logs, retries, cancellation & recovery."""

from __future__ import annotations

import json
import secrets
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from modelforge.core.events import EventBus
from modelforge.security.secrets import redact_secrets
from modelforge.storage.database import StorageDatabase


@dataclass
class TaskContext:
    task_id: str
    cancel_event: threading.Event
    _update_fn: Callable[[str, float, Optional[str]], None]

    @property
    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise InterruptedError(f"Task {self.task_id} cancelled by user.")

    def report(self, stage: str, progress: float, log_line: Optional[str] = None) -> None:
        self._update_fn(stage, max(0.0, min(1.0, float(progress))), log_line)


@dataclass
class TaskRecord:
    id: str
    project_id: str
    task_type: str
    title: str
    status: str  # queued | running | completed | failed | cancelled | interrupted
    stage: str = "queued"
    progress: float = 0.0
    payload: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    error: str = ""
    retries: int = 0
    max_retries: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TaskManager:
    """Executes long-running tasks without blocking GUI/API; persists progress & logs to SQLite."""

    def __init__(self, db: StorageDatabase, events: EventBus, max_workers: int = 3) -> None:
        self._db = db
        self._events = events
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="modelforge-worker")
        self._lock = threading.Lock()
        self._cancel_flags: Dict[str, threading.Event] = {}
        self._futures: Dict[str, Future[Any]] = {}

    def _persist(self, rec: TaskRecord) -> None:
        self._db.upsert_row(
            "tasks",
            {
                "id": rec.id,
                "project_id": rec.project_id,
                "task_type": rec.task_type,
                "title": rec.title,
                "status": rec.status,
                "stage": rec.stage,
                "progress": rec.progress,
                "payload_json": json.dumps(rec.payload, ensure_ascii=False),
                "result_json": json.dumps(rec.result, ensure_ascii=False),
                "logs_json": json.dumps(rec.logs[-200:], ensure_ascii=False),
                "error": rec.error,
                "retries": rec.retries,
                "max_retries": rec.max_retries,
                "created_at": rec.created_at,
                "updated_at": rec.updated_at,
            },
        )

    @staticmethod
    def _from_row(row: Dict[str, Any]) -> TaskRecord:
        return TaskRecord(
            id=row["id"],
            project_id=row["project_id"],
            task_type=row["task_type"],
            title=row["title"],
            status=row["status"],
            stage=row["stage"],
            progress=float(row["progress"]),
            payload=json.loads(row["payload_json"] or "{}"),
            result=json.loads(row["result_json"] or "{}"),
            logs=json.loads(row["logs_json"] or "[]"),
            error=row["error"] or "",
            retries=int(row["retries"]),
            max_retries=int(row["max_retries"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    def submit(
        self,
        task_type: str,
        title: str,
        fn: Callable[[TaskContext], Dict[str, Any]],
        project_id: str = "",
        payload: Optional[Dict[str, Any]] = None,
        max_retries: int = 0,
        wait: bool = False,
    ) -> TaskRecord:
        task_id = f"task-{secrets.token_hex(5)}"
        now = time.time()
        rec = TaskRecord(
            id=task_id,
            project_id=project_id,
            task_type=task_type,
            title=title,
            status="queued",
            stage="queued",
            progress=0.0,
            payload=payload or {},
            max_retries=max_retries,
            created_at=now,
            updated_at=now,
        )
        cancel_ev = threading.Event()
        with self._lock:
            self._cancel_flags[task_id] = cancel_ev
        self._persist(rec)
        self._events.publish("task.queued", rec.to_dict(), project_id=project_id)

        def _update(stage: str, progress: float, log_line: Optional[str]) -> None:
            rec.stage = stage
            rec.progress = round(progress, 4)
            rec.updated_at = time.time()
            if log_line:
                clean = redact_secrets(log_line)
                rec.logs.append(f"[{time.strftime('%H:%M:%S')}] {clean}")
            self._persist(rec)
            self._events.publish(
                "task.progress",
                {
                    "task_id": rec.id,
                    "task_type": rec.task_type,
                    "stage": rec.stage,
                    "progress": rec.progress,
                    "log": rec.logs[-1] if log_line and rec.logs else "",
                },
                project_id=project_id,
            )

        ctx = TaskContext(task_id=task_id, cancel_event=cancel_ev, _update_fn=_update)

        def _runner() -> TaskRecord:
            attempt = 0
            while True:
                if cancel_ev.is_set():
                    rec.status = "cancelled"
                    rec.stage = "cancelled"
                    rec.updated_at = time.time()
                    self._persist(rec)
                    self._events.publish("task.cancelled", rec.to_dict(), project_id=project_id)
                    return rec

                rec.status = "running"
                rec.stage = "starting"
                rec.updated_at = time.time()
                self._persist(rec)
                self._events.publish("task.started", rec.to_dict(), project_id=project_id)
                try:
                    res = fn(ctx)
                    rec.result = res or {}
                    rec.status = "completed"
                    rec.stage = "completed"
                    rec.progress = 1.0
                    rec.updated_at = time.time()
                    self._persist(rec)
                    self._events.publish("task.completed", rec.to_dict(), project_id=project_id)
                    return rec
                except InterruptedError as exc:
                    rec.status = "cancelled"
                    rec.stage = "cancelled"
                    rec.error = str(exc)
                    rec.updated_at = time.time()
                    self._persist(rec)
                    self._events.publish("task.cancelled", rec.to_dict(), project_id=project_id)
                    return rec
                except Exception as exc:
                    err_msg = redact_secrets(f"{type(exc).__name__}: {exc}")
                    rec.logs.append(f"[ERROR] {err_msg}")
                    if attempt < rec.max_retries and not cancel_ev.is_set():
                        attempt += 1
                        rec.retries = attempt
                        rec.stage = f"retrying ({attempt}/{rec.max_retries})"
                        rec.updated_at = time.time()
                        self._persist(rec)
                        time.sleep(0.2)
                        continue
                    rec.status = "failed"
                    rec.stage = "failed"
                    rec.error = err_msg
                    rec.updated_at = time.time()
                    rec.logs.append(redact_secrets(traceback.format_exc(limit=4)))
                    self._persist(rec)
                    self._events.publish("task.failed", rec.to_dict(), project_id=project_id)
                    return rec

        fut = self._pool.submit(_runner)
        with self._lock:
            self._futures[task_id] = fut
        if wait:
            return fut.result()
        return rec

    def cancel(self, task_id: str) -> TaskRecord:
        with self._lock:
            ev = self._cancel_flags.get(task_id)
        if ev:
            ev.set()
        rec = self.get_task(task_id)
        if rec.status in ("queued", "running"):
            rec.status = "cancelled"
            rec.stage = "cancelling"
            rec.updated_at = time.time()
            self._persist(rec)
        return rec

    def get_task(self, task_id: str) -> TaskRecord:
        row = self._db.query_one("SELECT * FROM tasks WHERE id = ?", (task_id,))
        if not row:
            raise KeyError(f"Task not found: {task_id}")
        return self._from_row(row)

    def list_tasks(self, project_id: str = "", limit: int = 50) -> List[TaskRecord]:
        if project_id:
            rows = self._db.query_all(
                "SELECT * FROM tasks WHERE project_id = ? ORDER BY created_at DESC LIMIT ?",
                (project_id, limit),
            )
        else:
            rows = self._db.query_all(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        return [self._from_row(r) for r in rows]

    def wait_for(self, task_id: str, timeout: Optional[float] = 60.0) -> TaskRecord:
        with self._lock:
            fut = self._futures.get(task_id)
        if fut is not None:
            fut.result(timeout=timeout)
        return self.get_task(task_id)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)
