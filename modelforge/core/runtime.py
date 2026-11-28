"""Central ModelForgeCore assembling all engines, adapters, diagnostics & lifecycle."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from modelforge.agents.runtime import AgentManager
from modelforge.core.config import ExecutionMode, ModelForgeConfig
from modelforge.core.events import EventBus
from modelforge.core.projects import ProjectManager
from modelforge.core.state import SystemStateManager
from modelforge.core.task_manager import TaskManager
from modelforge.datasets.manager import DatasetManager
from modelforge.evaluation.metrics import EvaluationEngine
from modelforge.inference.engine import InferenceEngine
from modelforge.integrations.external import ComposioAdapter, GitHubIntegration
from modelforge.integrations.llm_providers import LLMProviderManager
from modelforge.integrations.mcp import MCPManager
from modelforge.integrations.plugins import PluginManager
from modelforge.model_builder.builder import ModelBuilder
from modelforge.orchestration.distributed import RemoteExecutorClient, WorkerRegistry
from modelforge.platforms.detector import detect_platform, get_resource_snapshot
from modelforge.prompt_edit.editor import PromptProjectEditor
from modelforge.registry.catalog import UnifiedRegistry
from modelforge.research.engine import ResearchEngine
from modelforge.security.audit import AuditLogger
from modelforge.security.permissions import ConfirmationManager
from modelforge.security.secrets import SecretManager
from modelforge.storage.artifacts import ArtifactStore
from modelforge.storage.database import StorageDatabase
from modelforge.storage.memory_store import MemoryAndRagStore
from modelforge.training.engine import TrainingEngine


class ModelForgeCore:
    """
    Single unified core shared identically by CLI (`modelforge`), FastAPI Backend (`/api/v1`),
    and Flutter / Web GUIs.
    """

    def __init__(
        self,
        home_dir: Optional[Path] = None,
        repo_root: Optional[Path] = None,
    ) -> None:
        self.repo_root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
        resolved_home = (
            home_dir
            or Path(os.environ.get("MODELFORGE_HOME", str(Path.home() / ".modelforge")))
        ).resolve()
        resolved_home.mkdir(parents=True, exist_ok=True)
        self.home_dir = resolved_home

        self.platform_info = detect_platform(target_path=self.home_dir)
        self.config = ModelForgeConfig.load_or_create(
            home_dir=self.home_dir,
            low_memory_default=self.platform_info.low_memory_mode,
        )
        # Re-sync low_memory_mode with config
        self.platform_info.low_memory_mode = self.config.low_memory_mode

        self.events = EventBus()
        self.state = SystemStateManager()
        self.confirmations = ConfirmationManager()
        self.secrets = SecretManager(self.home_dir)
        self.audit = AuditLogger(self.home_dir / "security" / "audit.jsonl")

        self.db = StorageDatabase(self.home_dir / "storage" / "modelforge.db")
        self.artifacts = ArtifactStore(self.home_dir / "storage")
        self.memory = MemoryAndRagStore(self.db)
        self.tasks = TaskManager(
            self.db,
            self.events,
            max_workers=1 if self.platform_info.low_memory_mode else 3,
        )
        self.workers = WorkerRegistry()

        self.projects = ProjectManager(self.db, self.artifacts, self.events, self.confirmations)
        self.datasets = DatasetManager(self.db, self.artifacts)
        self.models = ModelBuilder(self.db, self.artifacts, self.events, self.platform_info)
        self.training = TrainingEngine(
            self.db,
            self.artifacts,
            self.events,
            self.models,
            self.datasets,
            self.platform_info,
        )
        self.inference = InferenceEngine(
            self.models,
            self.events,
            mode_getter=lambda: self.config.mode.value,
        )
        self.evaluation = EvaluationEngine(self.models, self.datasets, self.inference)

        self.mcp = MCPManager(
            self.db,
            self.audit,
            offline_getter=lambda: self.config.is_offline,
        )
        self.agents = AgentManager(
            self.db,
            self.artifacts,
            self.memory,
            self.events,
            self.audit,
            mcp_caller=self.mcp.call_tool,
            offline_getter=lambda: self.config.is_offline,
        )
        self.prompt_editor = PromptProjectEditor(
            self.db,
            self.artifacts,
            self.projects,
            self.confirmations,
            self.events,
            self.audit,
            max_retries=self.config.limits.max_prompt_edit_retries,
        )
        self.research = ResearchEngine(
            self.db,
            self.events,
            self.models,
            self.datasets,
            self.training,
            self.evaluation,
            self.platform_info,
            offline_getter=lambda: self.config.is_offline,
        )
        self.providers = LLMProviderManager(
            self.db,
            self.secrets,
            self.models,
            self.inference,
            offline_getter=lambda: self.config.is_offline,
        )
        self.plugins = PluginManager(self.repo_root / "plugins", self.audit)
        self.github = GitHubIntegration(self.repo_root, offline_getter=lambda: self.config.is_offline)
        self.composio = ComposioAdapter(self.secrets, offline_getter=lambda: self.config.is_offline)
        self.registry = UnifiedRegistry(
            self.projects,
            self.models,
            self.agents,
            self.datasets,
            self.research,
            self.artifacts,
        )

    def set_mode(
        self,
        mode: str | ExecutionMode,
        remote_server_url: Optional[str] = None,
        low_memory_mode: Optional[bool] = None,
    ) -> Dict[str, Any]:
        new_mode = ExecutionMode(mode) if not isinstance(mode, ExecutionMode) else mode
        self.config.mode = new_mode
        if remote_server_url is not None:
            self.config.remote_server_url = remote_server_url.strip()
        if low_memory_mode is not None:
            self.config.low_memory_mode = low_memory_mode
            self.platform_info.low_memory_mode = low_memory_mode
        self.config.save()

        remote_status: Dict[str, Any] = {"checked": False}
        if new_mode in (ExecutionMode.REMOTE, ExecutionMode.HYBRID) and self.config.remote_server_url:
            client = RemoteExecutorClient(
                self.config.remote_server_url,
                timeout_sec=self.config.remote_timeout_sec,
                offline_mode=self.config.is_offline,
            )
            remote_status = {"checked": True, **client.check_health()}

        self.audit.record(
            actor="user",
            action="core.set_mode",
            status="ok",
            details={"mode": new_mode.value, "low_memory_mode": self.config.low_memory_mode},
        )
        self.events.publish(
            "core.mode_changed",
            {"mode": new_mode.value, "remote_status": remote_status},
        )
        return {
            "mode": new_mode.value,
            "low_memory_mode": self.config.low_memory_mode,
            "remote_server_url": self.config.remote_server_url,
            "remote_status": remote_status,
        }

    def chat_workspace(
        self,
        message: str,
        project_id: str = "",
        provider_id: str = "provider-local",
        model_id: str = "",
    ) -> Dict[str, Any]:
        """Interactive AI Chat / Prompt Workspace recording dialogue memory & routing requests."""
        owner = project_id or "global"
        self.memory.add_memory("dialogue", owner, message, role="user")

        rag_context = self.memory.search_rag(project_id, message, top_k=2) if project_id else []
        comp = self.providers.complete(
            prompt=message,
            provider_id=provider_id,
            model_id=model_id,
            max_tokens=40,
            temperature=0.7,
        )
        reply_text = comp.get("full_text") or comp.get("text") or ""
        self.memory.add_memory(
            "dialogue",
            owner,
            reply_text,
            role="assistant",
            metadata={"provider_id": provider_id, "rag_citations": [c["citation"] for c in rag_context]},
        )
        evt_payload = {
            "project_id": project_id,
            "user_message": message,
            "assistant_reply": reply_text,
            "provider_id": provider_id,
            "rag_citations": rag_context,
        }
        self.events.publish("chat.message", evt_payload, project_id=project_id)
        return evt_payload

    def capabilities(self) -> Dict[str, Any]:
        """Run a bounded real probe; legacy static platform limits remain separate."""
        from modelforge.platforms.capabilities import measure_capabilities
        return measure_capabilities(path=self.home_dir, low_memory_mode=self.platform_info.low_memory_mode)

    def doctor(self) -> Dict[str, Any]:
        """Full diagnostic check across DB, audit chain, platform, RAM, plugins, MCP & remote connectivity."""
        db_ok, db_msg = self.db.check_integrity()
        audit_ok, audit_entries, audit_err = self.audit.verify_integrity()
        res = get_resource_snapshot(self.home_dir)
        mcp_servers = self.mcp.list_servers()
        plugins = self.plugins.discover_plugins()

        remote_probe: Dict[str, Any] = {"configured": bool(self.config.remote_server_url)}
        if self.config.remote_server_url and not self.config.is_offline:
            client = RemoteExecutorClient(self.config.remote_server_url, timeout_sec=4.0)
            remote_probe.update(client.check_health())

        healthy = db_ok and audit_ok
        return {
            "healthy": healthy,
            "version": "1.0.0",
            "home_dir": str(self.home_dir),
            "mode": self.config.mode.value,
            "offline_network_blocked": self.config.is_offline,
            "database": {"ok": db_ok, "integrity": db_msg, "path": str(self.db.db_path)},
            "audit_log": {"ok": audit_ok, "entries": audit_entries, "error": audit_err},
            "secret_vault": {"backend": self.secrets.storage_backend, "keys_stored": len(self.secrets.list_secret_names())},
            "platform": self.platform_info.to_dict(),
            "resources": res.to_dict(),
            "storage_usage": self.artifacts.disk_usage_summary(),
            "counts": self.registry.search()["counts"],
            "mcp_servers_count": len(mcp_servers),
            "plugins_discovered": [p["id"] for p in plugins],
            "github": self.github.status(),
            "composio": self.composio.status(),
            "remote_probe": remote_probe,
            "future_extensibility": {
                "distributed_workers": "interface_ready_v1",
                "multimodal_modules": "NOT_IMPLEMENTED_IN_MVP",
                "cloud_gpu_billing": "NOT_IMPLEMENTED_IN_MVP_REQUIRES_CONFIRMATION",
                "self_improvement_automerge": "DISABLED_BY_SECURITY_POLICY_BRANCH_REVIEW_REQUIRED",
            },
        }

    def shutdown(self) -> None:
        self.tasks.shutdown()


_DEFAULT_CORE: Optional[ModelForgeCore] = None


def get_core(home_dir: Optional[Path] = None) -> ModelForgeCore:
    global _DEFAULT_CORE
    if home_dir is not None:
        return ModelForgeCore(home_dir=home_dir)
    if _DEFAULT_CORE is None:
        _DEFAULT_CORE = ModelForgeCore()
    return _DEFAULT_CORE
