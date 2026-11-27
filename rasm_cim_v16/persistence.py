"""v0.16 — persistence: ~/.rasm-cim/kb.json and ~/.rasm-cim/context.json.

Layout (both documents are versioned; see ``migrate.py`` for the v1 -> v2
upgrade):

    ~/.rasm-cim/
        kb.json          semantic memory: entities, typed statements,
                         custom relations, evidence, conflicts, procedural cfg
        context.json     episodic + working memory: topics, active entities,
                         focus, pending clarifications
        kb.json.bak-<ts> written by migrate.py BEFORE any destructive step

Writes are crash-safe the same way the v3 store is: serialise to a temporary
file, ``flush`` + ``os.fsync``, then ``os.replace`` (atomic on POSIX).  A
process killed mid-save leaves either the old file or the new one, never a
half-written document.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Optional

from . import LEGACY_SCHEMA_VERSION, SCHEMA_VERSION

KB_FILENAME = "kb.json"
CONTEXT_FILENAME = "context.json"
DEFAULT_HOME = "~/.rasm-cim"


def default_home() -> str:
    """``$RASM_CIM_HOME`` if set (tests, Termux with a custom prefix), else
    ``~/.rasm-cim``.  Never a hardcoded absolute path."""
    return os.environ.get("RASM_CIM_HOME", DEFAULT_HOME)


def atomic_write_json(path: str, obj) -> int:
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1, sort_keys=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return os.path.getsize(path)


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class Store:
    """Owns the two JSON documents and the DialogueManager's (de)serialisation."""

    def __init__(self, home: str | None = None, clock=None, autosave: bool = False):
        self.home = os.path.expanduser(home or default_home())
        self.kb_path = os.path.join(self.home, KB_FILENAME)
        self.context_path = os.path.join(self.home, CONTEXT_FILENAME)
        self._clock = clock or (lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
        self.autosave = autosave
        self.last_save: dict = {}

    # -- queries -----------------------------------------------------------
    def exists(self) -> bool:
        return os.path.exists(self.kb_path)

    def context_exists(self) -> bool:
        return os.path.exists(self.context_path)

    def sizes(self) -> dict:
        return {"kb": os.path.getsize(self.kb_path) if self.exists() else 0,
                "context": os.path.getsize(self.context_path)
                if self.context_exists() else 0}

    def backups(self) -> list[str]:
        if not os.path.isdir(self.home):
            return []
        return sorted(f for f in os.listdir(self.home)
                      if f.startswith(KB_FILENAME + ".bak-"))

    # -- save --------------------------------------------------------------
    def serialise(self, dm) -> tuple[dict, dict]:
        kb = {
            "schema": SCHEMA_VERSION,
            "generator": "rasm-cim v0.16",
            "saved_at": self._clock(),
            "graph": dm.kg.to_dict(),
            "evidence": dm.evidence.to_dict(),
            "conflicts": dm.detector.to_dict(),
            "memory": dm.memory.to_dict(),
            "procedural": dm.memory.procedural(),
        }
        ctx = dm.context.to_dict()
        ctx["saved_at"] = self._clock()
        return kb, ctx

    def save(self, dm=None) -> dict:
        if dm is None:
            dm = self._dm
        kb, ctx = self.serialise(dm)
        kb_bytes = atomic_write_json(self.kb_path, kb)
        ctx_bytes = atomic_write_json(self.context_path, ctx)
        self.last_save = {"kb": self.kb_path, "kb_bytes": kb_bytes,
                          "context": self.context_path,
                          "context_bytes": ctx_bytes,
                          "saved_at": kb["saved_at"]}
        return dict(self.last_save)

    # -- load --------------------------------------------------------------
    def load(self, dm) -> dict:
        """Load both documents into ``dm``.  Missing files are not an error —
        that is simply a fresh system."""
        self._dm = dm
        notes = {"kb": [], "context": []}
        if self.exists():
            kb = read_json(self.kb_path)
            notes["kb"] = self._upgrade_kb(kb)
            dm.kg.load_dict(kb.get("graph", {}))
            # the graph loader may have adopted custom relations; keep the
            # inference engine and parser pointing at the same schema object
            dm.inference.schema = dm.kg.schema
            dm.parser.schema = dm.kg.schema
            dm.evidence.load_dict(kb.get("evidence", {}))
            dm.detector.load_dict(kb.get("conflicts", {}))
            dm.memory.load_dict(kb.get("memory", {}))
            dm.memory.kg = dm.kg
        if self.context_exists():
            ctx = read_json(self.context_path)
            notes["context"] = dm.context.load_dict(ctx)
            dm.context.resolver = dm.kg.resolver
        dm.kg._truth = None            # rebuild the polarity oracle lazily
        self.last_save = {"loaded_from": self.home}
        return notes

    def attach(self, dm) -> None:
        self._dm = dm
        dm.attach_store(self)
        dm.autosave = dm.autosave or self.autosave

    # -- backups / rollback ------------------------------------------------
    def backup(self, path: str | None = None, tag: str = "") -> str | None:
        src = path or self.kb_path
        if not os.path.exists(src):
            return None
        stamp = time.strftime("%Y%m%d-%H%M%S")
        dst = f"{src}.bak-{stamp}{('-' + tag) if tag else ''}"
        with open(src, "rb") as f:
            data = f.read()
        atomic_write_json(dst, json.loads(data.decode("utf-8")))
        return dst

    def rollback(self, backup_path: str) -> int:
        if not os.path.exists(backup_path):
            raise FileNotFoundError(backup_path)
        data = read_json(backup_path)
        return atomic_write_json(self.kb_path, data)

    # -- schema upgrade ----------------------------------------------------
    @staticmethod
    def _upgrade_kb(kb: dict) -> list[str]:
        """Delegate to migrate.py so the rules live in exactly one place."""
        from .migrate import upgrade_kb_dict
        return upgrade_kb_dict(kb)

    def needs_migration(self) -> bool:
        if not self.exists():
            return False
        kb = read_json(self.kb_path)
        return int(kb.get("schema", LEGACY_SCHEMA_VERSION)) < SCHEMA_VERSION
