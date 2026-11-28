"""Project lifecycle management with isolated workspaces & confirmation-gated deletion."""

from __future__ import annotations

import json
import re
import secrets
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from modelforge.core.events import EventBus
from modelforge.security.permissions import ConfirmationManager, PermissionScope
from modelforge.security.sandbox import resolve_within_workspace
from modelforge.storage.artifacts import ArtifactStore
from modelforge.storage.database import StorageDatabase


@dataclass
class Project:
    id: str
    name: str
    description: str
    workspace_path: str
    mode: str = "local"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^\w\-]+", "-", name.strip().lower(), flags=re.UNICODE).strip("-")
    return slug[:32] or "project"


class ProjectManager:
    """Manages isolated ModelForge projects across CLI, API, and Flutter GUI."""

    def __init__(
        self,
        db: StorageDatabase,
        artifacts: ArtifactStore,
        events: EventBus,
        confirmations: ConfirmationManager,
    ) -> None:
        self._db = db
        self._artifacts = artifacts
        self._events = events
        self._confirmations = confirmations

    def create_project(
        self,
        name: str,
        description: str = "",
        mode: str = "local",
        tags: Optional[List[str]] = None,
        template: str = "standard",
    ) -> Project:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Project name must not be empty.")
        proj_id = f"proj-{_slugify(clean_name)[:16]}-{secrets.token_hex(3)}"
        ws = self._artifacts.ensure_project_workspace(proj_id)
        now = time.time()
        tag_list = tags or []
        meta = {"template": template, "languages": ["ru", "tg", "en"]}

        # Scaffold real starter files inside the isolated workspace
        manifest = {
            "id": proj_id,
            "name": clean_name,
            "description": description,
            "mode": mode,
            "tags": tag_list,
            "created_at": now,
        }
        (ws / "modelforge.project.yaml").write_text(
            yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        (ws / "README.md").write_text(
            f"# {clean_name}\n\n{description or 'Проект лаборатории ModelForge AI.'}\n\n"
            f"- Project ID: `{proj_id}`\n- Mode: `{mode}`\n",
            encoding="utf-8",
        )
        (ws / "src" / "__init__.py").write_text(
            f'"""Workspace package for project {clean_name}."""\n\nSUPPORTED_LANGUAGES = ["ru", "en"]\nMAX_RAM_MB = 1024\nATTENTION_TYPE = "causal_mha"\n',
            encoding="utf-8",
        )
        (ws / "src" / "pipeline.py").write_text(
            '"""Project data & model pipeline configuration."""\n\n'
            "from src import ATTENTION_TYPE, MAX_RAM_MB, SUPPORTED_LANGUAGES\n\n\n"
            "def get_pipeline_config() -> dict:\n"
            "    return {\n"
            '        "languages": list(SUPPORTED_LANGUAGES),\n'
            '        "max_ram_mb": MAX_RAM_MB,\n'
            '        "attention": ATTENTION_TYPE,\n'
            '        "status": "ok",\n'
            "    }\n",
            encoding="utf-8",
        )
        (ws / "tests" / "test_pipeline.py").write_text(
            "from src.pipeline import get_pipeline_config\n\n\n"
            "def test_pipeline_config_valid():\n"
            "    cfg = get_pipeline_config()\n"
            '    assert cfg["status"] == "ok"\n'
            '    assert isinstance(cfg["languages"], list) and len(cfg["languages"]) >= 1\n'
            '    assert cfg["max_ram_mb"] > 0\n',
            encoding="utf-8",
        )

        proj = Project(
            id=proj_id,
            name=clean_name,
            description=description,
            workspace_path=str(ws),
            mode=mode,
            tags=tag_list,
            metadata=meta,
            created_at=now,
            updated_at=now,
        )
        self._db.upsert_row(
            "projects",
            {
                "id": proj.id,
                "name": proj.name,
                "description": proj.description,
                "workspace_path": proj.workspace_path,
                "mode": proj.mode,
                "tags_json": json.dumps(proj.tags, ensure_ascii=False),
                "metadata_json": json.dumps(proj.metadata, ensure_ascii=False),
                "created_at": proj.created_at,
                "updated_at": proj.updated_at,
            },
        )
        self._events.publish("project.created", proj.to_dict(), project_id=proj.id)
        return proj

    def _row_to_project(self, row: Dict[str, Any]) -> Project:
        return Project(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            workspace_path=row["workspace_path"],
            mode=row["mode"],
            tags=json.loads(row["tags_json"] or "[]"),
            metadata=json.loads(row["metadata_json"] or "{}"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get_project(self, project_id: str) -> Project:
        row = self._db.query_one("SELECT * FROM projects WHERE id = ?", (project_id,))
        if not row:
            raise KeyError(f"Project not found: {project_id}")
        return self._row_to_project(row)

    def list_projects(self, query: str = "") -> List[Project]:
        rows = self._db.query_all("SELECT * FROM projects ORDER BY updated_at DESC")
        projects = [self._row_to_project(r) for r in rows]
        if query.strip():
            q = query.strip().lower()
            projects = [
                p
                for p in projects
                if q in p.name.lower()
                or q in p.description.lower()
                or any(q in t.lower() for t in p.tags)
            ]
        return projects

    def list_project_files(self, project_id: str) -> List[Dict[str, Any]]:
        proj = self.get_project(project_id)
        ws = Path(proj.workspace_path).resolve()
        files: List[Dict[str, Any]] = []
        if not ws.exists():
            return files
        for p in sorted(ws.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                rel = p.relative_to(ws).as_posix()
                files.append(
                    {
                        "path": rel,
                        "size_bytes": p.stat().st_size,
                        "modified_at": p.stat().st_mtime,
                    }
                )
        return files

    def read_project_file(self, project_id: str, rel_path: str) -> str:
        proj = self.get_project(project_id)
        target = resolve_within_workspace(Path(proj.workspace_path), rel_path)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"File '{rel_path}' not found in project '{project_id}'.")
        return target.read_text(encoding="utf-8", errors="replace")

    def write_project_file(self, project_id: str, rel_path: str, content: str) -> Dict[str, Any]:
        proj = self.get_project(project_id)
        target = resolve_within_workspace(Path(proj.workspace_path), rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        self._db.execute(
            "UPDATE projects SET updated_at = ? WHERE id = ?",
            (time.time(), project_id),
        )
        return {"project_id": project_id, "path": rel_path, "size_bytes": len(content.encode("utf-8"))}

    def delete_project(self, project_id: str, confirmation_token: Optional[str] = None) -> Dict[str, Any]:
        proj = self.get_project(project_id)
        self._confirmations.require_or_verify(
            action=f"delete_project:{project_id}",
            scope=PermissionScope.WORKSPACE_DELETE,
            subject_id="user",
            summary=f"Delete project '{proj.name}' ({project_id}) and its workspace files.",
            details={"project_id": project_id, "workspace_path": proj.workspace_path},
            confirmation_token=confirmation_token,
        )
        ws = Path(proj.workspace_path)
        if ws.exists():
            shutil.rmtree(ws)
        self._db.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        self._events.publish("project.deleted", {"project_id": project_id}, project_id=project_id)
        return {"deleted": True, "project_id": project_id}
