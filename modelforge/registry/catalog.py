"""Unified Model, Agent, Dataset, Architecture, Project & Experiment Registry + Exporter (Section 9)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from modelforge.agents.runtime import AgentManager
from modelforge.architectures.models import SUPPORTED_ARCHITECTURES
from modelforge.architectures.pytorch_adapter import export_weights_to_pytorch
from modelforge.core.projects import ProjectManager
from modelforge.datasets.manager import DatasetManager
from modelforge.inference.gguf import export_gguf_v3
from modelforge.model_builder.builder import ModelBuilder
from modelforge.research.engine import ResearchEngine
from modelforge.storage.artifacts import ArtifactStore
from modelforge.training.checkpoints import export_safetensors, load_checkpoint


class UnifiedRegistry:
    """
    Specification Section 9:
    Searchable, taggable, versioned catalog across models, agents, datasets, architectures,
    projects, and experiments + real verified export to GGUF v3, Safetensors, PyTorch (.pt), JSON & YAML.
    """

    def __init__(
        self,
        projects: ProjectManager,
        models: ModelBuilder,
        agents: AgentManager,
        datasets: DatasetManager,
        research: ResearchEngine,
        artifacts: ArtifactStore,
    ) -> None:
        self._projects = projects
        self._models = models
        self._agents = agents
        self._datasets = datasets
        self._research = research
        self._artifacts = artifacts

    def search(self, query: str = "", tag: str = "", project_id: str = "") -> Dict[str, Any]:
        q = query.strip().lower()
        t = tag.strip().lower()

        def _match(name: str, tags: List[str], extra: str = "") -> bool:
            if q and q not in name.lower() and q not in extra.lower():
                return False
            if t and not any(t == x.lower() for x in tags):
                return False
            return True

        projects = [
            p.to_dict()
            for p in self._projects.list_projects()
            if (not project_id or p.id == project_id) and _match(p.name, p.tags, p.description)
        ]
        models = [
            m
            for m in self._models.list_models(project_id=project_id)
            if _match(m["name"], m["tags"], m["architecture"])
        ]
        agents = [
            a
            for a in self._agents.list_agents(project_id=project_id)
            if _match(a["name"], [a["agent_type"]], a["goal"])
        ]
        datasets = [
            d
            for d in self._datasets.list_datasets(project_id=project_id)
            if _match(d["name"], d["tags"], d["license"])
        ]
        experiments = [
            e
            for e in self._research.list_experiments(project_id=project_id)
            if _match(e["goal"], [e["architecture"]], e["conclusion"])
        ]
        architectures = [
            {"id": arch, "supported": True}
            for arch in SUPPORTED_ARCHITECTURES
            if not q or q in arch.lower()
        ]

        return {
            "query": query,
            "tag": tag,
            "counts": {
                "projects": len(projects),
                "models": len(models),
                "agents": len(agents),
                "datasets": len(datasets),
                "experiments": len(experiments),
                "architectures": len(architectures),
            },
            "projects": projects,
            "models": models,
            "agents": agents,
            "datasets": datasets,
            "experiments": experiments,
            "architectures": architectures,
        }

    def export_model(self, model_id: str, fmt: str, output_path: Path | None = None) -> Dict[str, Any]:
        info = self._models.get_model(model_id)
        ckpt = info["latest_checkpoint"]
        kind = fmt.strip().lower().lstrip(".")
        out_dir = self._artifacts.exports_dir / model_id
        out_dir.mkdir(parents=True, exist_ok=True)

        if kind == "gguf":
            target = output_path or (out_dir / f"{info['name']}-v{info['version']}.gguf")
            return export_gguf_v3(ckpt, target)
        if kind == "safetensors":
            target = output_path or (out_dir / f"{info['name']}-v{info['version']}.safetensors")
            return export_safetensors(ckpt, target)
        if kind in ("pt", "pytorch"):
            loaded = load_checkpoint(ckpt)
            target = output_path or (out_dir / f"{info['name']}-v{info['version']}.pt")
            return export_weights_to_pytorch(loaded["weights"], info["spec"], target)
        if kind == "json":
            target = output_path or (out_dir / f"{info['name']}-v{info['version']}.json")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
            return {"format": "json", "path": str(target), "size_bytes": target.stat().st_size}
        if kind in ("yaml", "yml"):
            target = output_path or (out_dir / f"{info['name']}-v{info['version']}.yaml")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(yaml.safe_dump(info, allow_unicode=True, sort_keys=False), encoding="utf-8")
            return {"format": "yaml", "path": str(target), "size_bytes": target.stat().st_size}

        raise ValueError(
            f"Unsupported export format '{fmt}'. Supported formats: gguf, safetensors, pt (requires torch), json, yaml."
        )
