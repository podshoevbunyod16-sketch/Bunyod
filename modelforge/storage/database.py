"""SQLite WAL metadata repository with schema migration, backup, restore & integrity verification."""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1

_DDL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    workspace_path TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'local',
    tags_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    name TEXT NOT NULL,
    architecture TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'created',
    spec_json TEXT NOT NULL,
    tokenizer_path TEXT NOT NULL DEFAULT '',
    latest_checkpoint TEXT NOT NULL DEFAULT '',
    metrics_json TEXT NOT NULL DEFAULT '{}',
    provenance_json TEXT NOT NULL DEFAULT '{}',
    tags_json TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS model_versions (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    spec_json TEXT NOT NULL,
    checkpoint_path TEXT NOT NULL DEFAULT '',
    checkpoint_sha256 TEXT NOT NULL DEFAULT '',
    metrics_json TEXT NOT NULL DEFAULT '{}',
    notes TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    name TEXT NOT NULL,
    format TEXT NOT NULL,
    source_path TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    license TEXT NOT NULL DEFAULT 'unspecified',
    record_count INTEGER NOT NULL DEFAULT 0,
    validation_json TEXT NOT NULL DEFAULT '{}',
    splits_json TEXT NOT NULL DEFAULT '{}',
    tags_json TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    name TEXT NOT NULL,
    agent_type TEXT NOT NULL,
    goal TEXT NOT NULL,
    system_prompt TEXT NOT NULL,
    model_ref TEXT NOT NULL DEFAULT 'local-default',
    tools_json TEXT NOT NULL DEFAULT '[]',
    permissions_json TEXT NOT NULL DEFAULT '{}',
    limits_json TEXT NOT NULL DEFAULT '{}',
    output_format TEXT NOT NULL DEFAULT 'markdown',
    completion_criteria TEXT NOT NULL DEFAULT '',
    workspace_subdir TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'ready',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL DEFAULT '',
    task_type TEXT NOT NULL,
    title TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL DEFAULT 'queued',
    progress REAL NOT NULL DEFAULT 0.0,
    payload_json TEXT NOT NULL DEFAULT '{}',
    result_json TEXT NOT NULL DEFAULT '{}',
    logs_json TEXT NOT NULL DEFAULT '[]',
    error TEXT NOT NULL DEFAULT '',
    retries INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    goal TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    sources_json TEXT NOT NULL DEFAULT '[]',
    code_version TEXT NOT NULL DEFAULT '',
    architecture TEXT NOT NULL DEFAULT '',
    config_json TEXT NOT NULL DEFAULT '{}',
    dataset_id TEXT NOT NULL DEFAULT '',
    dataset_hash TEXT NOT NULL DEFAULT '',
    seed INTEGER NOT NULL DEFAULT 42,
    environment_json TEXT NOT NULL DEFAULT '{}',
    hardware_json TEXT NOT NULL DEFAULT '{}',
    duration_sec REAL NOT NULL DEFAULT 0.0,
    resources_json TEXT NOT NULL DEFAULT '{}',
    metrics_json TEXT NOT NULL DEFAULT '{}',
    checkpoint_path TEXT NOT NULL DEFAULT '',
    logs_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'planned',
    is_negative_result INTEGER NOT NULL DEFAULT 0,
    conclusion TEXT NOT NULL DEFAULT '',
    next_experiment_suggestion TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS prompt_edits (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    prompt TEXT NOT NULL,
    plan_json TEXT NOT NULL DEFAULT '[]',
    affected_files_json TEXT NOT NULL DEFAULT '[]',
    snapshot_dir TEXT NOT NULL,
    diff_text TEXT NOT NULL DEFAULT '',
    test_report_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL,
    rolled_back INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS mcp_servers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    transport TEXT NOT NULL,
    endpoint_or_cmd TEXT NOT NULL,
    args_json TEXT NOT NULL DEFAULT '[]',
    enabled INTEGER NOT NULL DEFAULT 1,
    permissions_json TEXT NOT NULL DEFAULT '{}',
    assigned_agents_json TEXT NOT NULL DEFAULT '[]',
    discovered_tools_json TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS mcp_call_logs (
    id TEXT PRIMARY KEY,
    server_id TEXT NOT NULL,
    agent_id TEXT NOT NULL DEFAULT '',
    tool_name TEXT NOT NULL,
    status TEXT NOT NULL,
    duration_ms REAL NOT NULL,
    request_summary TEXT NOT NULL DEFAULT '',
    response_summary TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_providers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    provider_type TEXT NOT NULL,
    base_url TEXT NOT NULL DEFAULT '',
    default_model TEXT NOT NULL DEFAULT '',
    secret_key_name TEXT NOT NULL DEFAULT '',
    is_paid INTEGER NOT NULL DEFAULT 0,
    is_external INTEGER NOT NULL DEFAULT 0,
    enabled INTEGER NOT NULL DEFAULT 1,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'note',
    content TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS rag_chunks (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    source_path TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    content TEXT NOT NULL,
    tokens_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


class StorageDatabase:
    """Thread-safe SQLite WAL database backing all ModelForge registries & state."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(_DDL)
                conn.execute(
                    "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
                # Recover any tasks interrupted by an unclean process exit
                conn.execute(
                    "UPDATE tasks SET status = 'interrupted', stage = 'interrupted_on_restart', "
                    "error = 'Recovered after process restart; safe to retry or resume.', updated_at = ? "
                    "WHERE status IN ('running', 'queued')",
                    (time.time(),),
                )
                conn.commit()
            finally:
                conn.close()

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(sql, params)
                conn.commit()
            finally:
                conn.close()

    def query_all(self, sql: str, params: tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def query_one(self, sql: str, params: tuple[Any, ...] = ()) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(sql, params).fetchone()
                return dict(row) if row is not None else None
            finally:
                conn.close()

    def upsert_row(self, table: str, data: Dict[str, Any]) -> None:
        cols = list(data.keys())
        placeholders = ", ".join("?" for _ in cols)
        col_list = ", ".join(cols)
        vals = tuple(
            json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
            for v in data.values()
        )
        sql = f"INSERT OR REPLACE INTO {table} ({col_list}) VALUES ({placeholders})"
        self.execute(sql, vals)

    def check_integrity(self) -> tuple[bool, str]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("PRAGMA integrity_check;").fetchone()
                msg = str(row[0]) if row else "unknown"
                return msg.lower() == "ok", msg
            finally:
                conn.close()

    def create_backup(self, backup_path: Path) -> Path:
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            src = self._connect()
            dst = sqlite3.connect(str(backup_path))
            try:
                src.backup(dst)
            finally:
                dst.close()
                src.close()
        return backup_path

    def restore_backup(self, backup_path: Path) -> None:
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup database not found: {backup_path}")
        with self._lock:
            shutil.copy2(backup_path, self.db_path)
        self._init_db()
