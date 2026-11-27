"""File artifact & project workspace manager with SHA-256 integrity checks."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Dict


class ArtifactStore:
    """Manages project directories, checkpoints, datasets, and snapshots."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir.resolve()
        self.projects_dir = self.root_dir / "projects"
        self.checkpoints_dir = self.root_dir / "checkpoints"
        self.datasets_dir = self.root_dir / "datasets"
        self.snapshots_dir = self.root_dir / "snapshots"
        self.exports_dir = self.root_dir / "exports"
        self.backups_dir = self.root_dir / "backups"
        for d in (
            self.projects_dir,
            self.checkpoints_dir,
            self.datasets_dir,
            self.snapshots_dir,
            self.exports_dir,
            self.backups_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def sha256_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def ensure_project_workspace(self, project_id: str) -> Path:
        ws = (self.projects_dir / project_id).resolve()
        ws.mkdir(parents=True, exist_ok=True)
        for sub in ("src", "models", "datasets", "agents", "tests", "docs"):
            (ws / sub).mkdir(exist_ok=True)
        return ws

    def create_workspace_snapshot(self, project_id: str, snapshot_id: str) -> Path:
        src = self.ensure_project_workspace(project_id)
        dst = (self.snapshots_dir / project_id / snapshot_id).resolve()
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
        return dst

    def restore_workspace_snapshot(self, project_id: str, snapshot_dir: Path) -> None:
        if not snapshot_dir.exists():
            raise FileNotFoundError(f"Snapshot directory not found: {snapshot_dir}")
        dst = (self.projects_dir / project_id).resolve()
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(snapshot_dir, dst)

    def disk_usage_summary(self) -> Dict[str, int]:
        summary: Dict[str, int] = {}
        for label, d in [
            ("projects_bytes", self.projects_dir),
            ("checkpoints_bytes", self.checkpoints_dir),
            ("datasets_bytes", self.datasets_dir),
            ("snapshots_bytes", self.snapshots_dir),
            ("exports_bytes", self.exports_dir),
        ]:
            total = 0
            if d.exists():
                for p in d.rglob("*"):
                    if p.is_file():
                        total += p.stat().st_size
            summary[label] = total
        return summary
