"""Versioned FastAPI REST (`/api/v1/*`) + WebSocket (`/ws/events`) server for ModelForge Core."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from modelforge.agents.runtime import AgentSpec
from modelforge.backend.web_ui import WEB_STUDIO_HTML
from modelforge.core.runtime import ModelForgeCore, get_core
from modelforge.inference.engine import GenerationConfig
from modelforge.model_builder.builder import ModelSpec
from modelforge.security.permissions import ConfirmationRequiredError, PermissionDeniedError
from modelforge.security.sandbox import WorkspaceSandboxError
from modelforge.training.engine import TrainingRunConfig


class CreateProjectReq(BaseModel):
    name: str
    description: str = ""
    mode: str = "local"
    tags: List[str] = Field(default_factory=list)


class ModeSwitchReq(BaseModel):
    mode: str
    remote_server_url: Optional[str] = None
    low_memory_mode: Optional[bool] = None


class ChatReq(BaseModel):
    message: str
    project_id: str = ""
    provider_id: str = "provider-local"
    model_id: str = ""


class ModelPromptReq(BaseModel):
    project_id: str
    prompt: str
    loop_passes: int = Field(default=1, ge=1, le=8, strict=True)
    name: Optional[str] = None
    corpus_texts: Optional[List[str]] = None


class AutomaticModelReq(BaseModel):
    project_id: str
    purpose: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=1, max_length=100000)
    target_weight_mb: float = Field(default=1.0, ge=0.1, le=10, allow_inf_nan=False)
    loop_passes: int = Field(default=1, ge=1, le=8, strict=True)


class CreateModelSpecReq(BaseModel):
    project_id: str
    spec: ModelSpec
    corpus_texts: Optional[List[str]] = None


class TrainReq(BaseModel):
    model_id: str
    dataset_id: str = ""
    epochs: int = 4
    batch_size: int = 4
    learning_rate: float = 3e-3
    grad_accumulation: int = 1
    max_seq_len: int = 64
    mode: str = "scratch"
    resume_from_checkpoint: str = ""
    wait: bool = True


class GenerateReq(BaseModel):
    model_id: str
    prompt: str
    max_tokens: int = 32
    temperature: float = 0.8
    top_k: int = 20
    top_p: float = 0.92
    repetition_penalty: float = 1.15
    seed: int = 42


class ExportReq(BaseModel):
    format: str = "gguf"


class DatasetIngestReq(BaseModel):
    project_id: str
    name: str
    content: str
    format: str = "txt"
    license: str = "MIT"


class AgentPromptReq(BaseModel):
    project_id: str
    prompt: str
    name: Optional[str] = None


class CreateAgentSpecReq(BaseModel):
    project_id: str
    spec: AgentSpec


class AgentRunReq(BaseModel):
    prompt: str


class PromptEditReq(BaseModel):
    prompt: str
    target_rel_path: Optional[str] = None
    new_file_content: Optional[str] = None
    confirmation_token: Optional[str] = None


class ResearchStudyReq(BaseModel):
    project_id: str
    question: str
    baseline_arch: str = "decoder_transformer"
    candidate_arch: str = "hybrid_attention_ssm"
    epochs: int = 3
    d_model: int = 48
    n_layers: int = 2
    seed: int = 42


class AddMcpReq(BaseModel):
    name: str
    transport: str = "stdio"
    endpoint_or_cmd: str
    args: List[str] = Field(default_factory=list)
    allowed_tools: List[str] = Field(default_factory=lambda: ["*"])
    assigned_agents: List[str] = Field(default_factory=lambda: ["*"])


class McpCallReq(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    agent_id: str = "user"


class AddProviderReq(BaseModel):
    name: str
    provider_type: str
    base_url: str = ""
    default_model: str = ""
    api_key: Optional[str] = None
    is_paid: bool = False


class PluginRunReq(BaseModel):
    inputs: Dict[str, Any] = Field(default_factory=dict)


class FileWriteReq(BaseModel):
    path: str
    content: str


def create_app(core: Optional[ModelForgeCore] = None, home_dir: Optional[Path] = None) -> FastAPI:
    mf = core or get_core(home_dir=home_dir)

    app = FastAPI(
        title="ModelForge AI Core API",
        version="1.0.0",
        description="Unified REST & WebSocket API for ModelForge AI Desktop, Mobile & CLI.",
    )
    app.state.core = mf

    # Allow preview hosts (`*.e2b.app`), desktop localhost, and Flutter clients cleanly
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(ConfirmationRequiredError)
    async def _confirm_exc(_, exc: ConfirmationRequiredError):
        return JSONResponse(
            status_code=409,
            content={
                "error": "confirmation_required",
                "message": str(exc),
                "request_id": exc.request_id,
                "action": exc.action,
                "details": exc.details,
            },
        )

    @app.exception_handler(PermissionDeniedError)
    async def _perm_exc(_, exc: PermissionDeniedError):
        return JSONResponse(status_code=403, content={"error": "permission_denied", "detail": str(exc)})

    @app.exception_handler(WorkspaceSandboxError)
    async def _sandbox_exc(_, exc: WorkspaceSandboxError):
        return JSONResponse(status_code=403, content={"error": "workspace_escape_blocked", "detail": str(exc)})

    @app.exception_handler(KeyError)
    async def _key_exc(_, exc: KeyError):
        return JSONResponse(status_code=404, content={"error": "not_found", "detail": str(exc)})

    @app.get("/", response_class=HTMLResponse)
    async def web_studio() -> HTMLResponse:
        return HTMLResponse(WEB_STUDIO_HTML)

    @app.get("/api/v1/health")
    async def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "version": "1.0.0",
            "mode": mf.config.mode.value,
            "low_memory_mode": mf.platform_info.low_memory_mode,
            "state": mf.state.snapshot(),
        }

    @app.get("/api/v1/system/capabilities")
    def capabilities() -> Dict[str, Any]:
        # FastAPI runs synchronous endpoints in its threadpool: the probe can take time.
        return mf.capabilities()

    @app.get("/api/v1/doctor")
    async def doctor() -> Dict[str, Any]:
        return mf.doctor()

    @app.post("/api/v1/mode")
    async def switch_mode(req: ModeSwitchReq) -> Dict[str, Any]:
        return mf.set_mode(
            mode=req.mode,
            remote_server_url=req.remote_server_url,
            low_memory_mode=req.low_memory_mode,
        )

    @app.get("/api/v1/confirmations")
    async def list_confirmations() -> Dict[str, Any]:
        return {"pending": mf.confirmations.list_pending()}

    @app.post("/api/v1/confirmations/{request_id}/approve")
    async def approve_confirmation(request_id: str) -> Dict[str, Any]:
        token = mf.confirmations.approve(request_id)
        return {"request_id": request_id, "approval_token": token}

    # --- Projects & Workspace Explorer ---
    @app.get("/api/v1/projects")
    async def list_projects(q: str = "") -> Dict[str, Any]:
        return {"projects": [p.to_dict() for p in mf.projects.list_projects(query=q)]}

    @app.post("/api/v1/projects")
    async def create_project(req: CreateProjectReq) -> Dict[str, Any]:
        proj = mf.projects.create_project(
            name=req.name,
            description=req.description,
            mode=req.mode,
            tags=req.tags,
        )
        mf.state.set_active_project(proj.id)
        return proj.to_dict()

    @app.get("/api/v1/projects/{project_id}")
    async def get_project(project_id: str) -> Dict[str, Any]:
        return mf.projects.get_project(project_id).to_dict()

    @app.get("/api/v1/projects/{project_id}/files")
    async def get_project_files(project_id: str) -> Dict[str, Any]:
        return {"files": mf.projects.list_project_files(project_id)}

    @app.get("/api/v1/projects/{project_id}/file")
    async def read_project_file(project_id: str, path: str) -> Dict[str, Any]:
        return {"path": path, "content": mf.projects.read_project_file(project_id, path)}

    @app.post("/api/v1/projects/{project_id}/file")
    async def write_project_file(project_id: str, req: FileWriteReq) -> Dict[str, Any]:
        return mf.projects.write_project_file(project_id, req.path, req.content)

    @app.delete("/api/v1/projects/{project_id}")
    async def delete_project(project_id: str, confirmation_token: Optional[str] = None) -> Dict[str, Any]:
        return mf.projects.delete_project(project_id, confirmation_token=confirmation_token)

    # --- AI Chat / Prompt Workspace ---
    @app.post("/api/v1/chat")
    async def chat_workspace(req: ChatReq) -> Dict[str, Any]:
        return mf.chat_workspace(
            message=req.message,
            project_id=req.project_id,
            provider_id=req.provider_id,
            model_id=req.model_id,
        )

    # --- Datasets ---
    @app.get("/api/v1/datasets")
    async def list_datasets(project_id: str = "") -> Dict[str, Any]:
        return {"datasets": mf.datasets.list_datasets(project_id=project_id)}

    @app.post("/api/v1/datasets")
    async def ingest_dataset(req: DatasetIngestReq) -> Dict[str, Any]:
        return mf.datasets.ingest_dataset(
            project_id=req.project_id,
            name=req.name,
            raw_content=req.content,
            fmt=req.format,
            license_name=req.license,
        )

    # --- Models ---
    @app.get("/api/v1/models")
    async def list_models(project_id: str = "") -> Dict[str, Any]:
        return {"models": mf.models.list_models(project_id=project_id)}

    @app.get("/api/v1/models/{model_id}")
    async def get_model(model_id: str) -> Dict[str, Any]:
        return mf.models.get_model(model_id)

    @app.post("/api/v1/models/automatic")
    async def create_automatic_model(req: AutomaticModelReq) -> Dict[str, Any]:
        from dataclasses import asdict
        from modelforge.model_builder.automatic import plan_model
        from modelforge.tokenizers.tokenizer import train_tokenizer

        if not req.text.strip() or not req.purpose.strip():
            raise HTTPException(status_code=422, detail="Purpose and training text must not be empty.")
        # Ingest first, then train the tokenizer only on the training split.
        try:
            dataset = mf.datasets.ingest_dataset(
                project_id=req.project_id, name="Данные модели", raw_content=req.text,
                fmt="txt", license_name="user-provided",
            )
            train_texts = mf.datasets.load_splits(dataset["id"])["train"]
            tokenizer = train_tokenizer(train_texts, tokenizer_type="char", vocab_size=320)
            plan = plan_model(req.target_weight_mb, req.loop_passes, tokenizer.vocab_size, mf.platform_info)
            model = mf.models.create_model(
                req.project_id, plan.spec, corpus_texts=train_texts,
                provenance={"source": "automatic", "purpose": req.purpose},
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        result = asdict(plan)
        result["spec"] = plan.spec.model_dump()
        return {"model": model, "dataset": dataset, "plan": result}

    @app.post("/api/v1/models/from-prompt")
    async def create_model_from_prompt(req: ModelPromptReq) -> Dict[str, Any]:
        try:
            return mf.models.create_from_prompt(
                project_id=req.project_id,
                prompt=req.prompt,
                name=req.name,
                corpus_texts=req.corpus_texts,
                loop_passes=req.loop_passes,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/models")
    async def create_model_spec(req: CreateModelSpecReq) -> Dict[str, Any]:
        return mf.models.create_model(
            project_id=req.project_id,
            spec=req.spec,
            corpus_texts=req.corpus_texts,
        )

    @app.post("/api/v1/models/{model_id}/export")
    async def export_model(model_id: str, req: ExportReq) -> Dict[str, Any]:
        try:
            return mf.registry.export_model(model_id, req.format)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/v1/models/{model_id}/evaluate")
    async def evaluate_model(model_id: str, dataset_id: str = "") -> Dict[str, Any]:
        return mf.evaluation.evaluate_model(model_id, dataset_id=dataset_id)

    # --- Training & Tasks ---
    @app.post("/api/v1/training/run")
    async def start_training(req: TrainReq) -> Dict[str, Any]:
        info = mf.models.get_model(req.model_id)
        cfg = TrainingRunConfig(
            epochs=req.epochs,
            batch_size=req.batch_size,
            learning_rate=req.learning_rate,
            grad_accumulation=req.grad_accumulation,
            max_seq_len=req.max_seq_len,
            mode=req.mode,
            resume_from_checkpoint=req.resume_from_checkpoint,
        )

        def _job(ctx) -> Dict[str, Any]:
            res = mf.training.train(
                model_id=req.model_id,
                config=cfg,
                dataset_id=req.dataset_id,
                cancel_event=ctx.cancel_event,
                progress_cb=ctx.report,
            )
            mf.state.increment("models_trained")
            return res

        rec = mf.tasks.submit(
            task_type="training",
            title=f"Train {info['name']} ({req.mode})",
            fn=_job,
            project_id=info["project_id"],
            payload=req.model_dump(),
            wait=req.wait,
        )
        return rec.to_dict()

    @app.get("/api/v1/tasks")
    async def list_tasks(project_id: str = "") -> Dict[str, Any]:
        return {"tasks": [t.to_dict() for t in mf.tasks.list_tasks(project_id=project_id)]}

    @app.get("/api/v1/tasks/{task_id}")
    async def get_task(task_id: str) -> Dict[str, Any]:
        return mf.tasks.get_task(task_id).to_dict()

    @app.post("/api/v1/tasks/{task_id}/cancel")
    async def cancel_task(task_id: str) -> Dict[str, Any]:
        return mf.tasks.cancel(task_id).to_dict()

    # --- Inference ---
    @app.post("/api/v1/inference/generate")
    async def generate_text(req: GenerateReq) -> Dict[str, Any]:
        cfg = GenerationConfig(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            seed=req.seed,
        )
        out = mf.inference.generate(model_id=req.model_id, prompt=req.prompt, config=cfg)
        mf.state.increment("inferences_run")
        return out

    # --- Agents ---
    @app.get("/api/v1/agents")
    async def list_agents(project_id: str = "") -> Dict[str, Any]:
        return {"agents": mf.agents.list_agents(project_id=project_id)}

    @app.post("/api/v1/agents/from-prompt")
    async def create_agent_from_prompt(req: AgentPromptReq) -> Dict[str, Any]:
        return mf.agents.create_from_prompt(project_id=req.project_id, prompt=req.prompt, name=req.name)

    @app.post("/api/v1/agents")
    async def create_agent_spec(req: CreateAgentSpecReq) -> Dict[str, Any]:
        return mf.agents.create_agent(project_id=req.project_id, spec=req.spec)

    @app.post("/api/v1/agents/{agent_id}/run")
    async def run_agent(agent_id: str, req: AgentRunReq) -> Dict[str, Any]:
        out = mf.agents.run_agent(agent_id=agent_id, task_prompt=req.prompt)
        mf.state.increment("agent_runs")
        return out

    # --- Prompt-to-Project ---
    @app.post("/api/v1/projects/{project_id}/prompt-edit")
    async def prompt_edit_project(project_id: str, req: PromptEditReq) -> Dict[str, Any]:
        out = mf.prompt_editor.apply_edit(
            project_id=project_id,
            prompt=req.prompt,
            target_rel_path=req.target_rel_path,
            new_file_content=req.new_file_content,
            confirmation_token=req.confirmation_token,
        )
        mf.state.increment("prompt_edits")
        return out

    @app.get("/api/v1/projects/{project_id}/prompt-edits")
    async def list_prompt_edits(project_id: str) -> Dict[str, Any]:
        return {"edits": mf.prompt_editor.list_edits(project_id)}

    @app.post("/api/v1/prompt-edits/{edit_id}/rollback")
    async def rollback_prompt_edit(edit_id: str) -> Dict[str, Any]:
        return mf.prompt_editor.rollback_edit(edit_id)

    # --- Research Lab & Experiment Registry ---
    @app.post("/api/v1/research/study")
    async def run_research_study(req: ResearchStudyReq) -> Dict[str, Any]:
        out = mf.research.run_comparative_study(
            project_id=req.project_id,
            question=req.question,
            baseline_arch=req.baseline_arch,
            candidate_arch=req.candidate_arch,
            epochs=req.epochs,
            d_model=req.d_model,
            n_layers=req.n_layers,
            seed=req.seed,
        )
        mf.state.increment("experiments_run", 2)
        return out

    @app.get("/api/v1/experiments")
    async def list_experiments(project_id: str = "") -> Dict[str, Any]:
        return {"experiments": mf.research.list_experiments(project_id=project_id)}

    # --- Registry ---
    @app.get("/api/v1/registry")
    async def search_registry(q: str = "", tag: str = "", project_id: str = "") -> Dict[str, Any]:
        return mf.registry.search(query=q, tag=tag, project_id=project_id)

    # --- MCP, Plugins & Providers ---
    @app.get("/api/v1/mcp")
    async def list_mcp_servers() -> Dict[str, Any]:
        return {"servers": mf.mcp.list_servers(), "recent_calls": mf.mcp.list_call_logs(limit=25)}

    @app.post("/api/v1/mcp")
    async def add_mcp_server(req: AddMcpReq) -> Dict[str, Any]:
        return mf.mcp.add_server(
            name=req.name,
            transport=req.transport,
            endpoint_or_cmd=req.endpoint_or_cmd,
            args=req.args,
            allowed_tools=req.allowed_tools,
            assigned_agents=req.assigned_agents,
        )

    @app.post("/api/v1/mcp/{server_id}/probe")
    async def probe_mcp_server(server_id: str) -> Dict[str, Any]:
        return mf.mcp.probe_and_discover_tools(server_id)

    @app.post("/api/v1/mcp/{server_id}/call")
    async def call_mcp_tool(server_id: str, req: McpCallReq) -> Dict[str, Any]:
        return mf.mcp.call_tool(server_id, req.tool_name, req.arguments, agent_id=req.agent_id)

    @app.get("/api/v1/plugins")
    async def list_plugins() -> Dict[str, Any]:
        return {"plugins": mf.plugins.discover_plugins()}

    @app.post("/api/v1/plugins/{plugin_id}/run")
    async def run_plugin(plugin_id: str, req: PluginRunReq) -> Dict[str, Any]:
        return mf.plugins.run_plugin(plugin_id, req.inputs)

    @app.get("/api/v1/providers")
    async def list_providers() -> Dict[str, Any]:
        return {"providers": mf.providers.list_providers()}

    @app.post("/api/v1/providers")
    async def add_provider(req: AddProviderReq) -> Dict[str, Any]:
        return mf.providers.register_provider(
            name=req.name,
            provider_type=req.provider_type,
            base_url=req.base_url,
            default_model=req.default_model,
            api_key=req.api_key,
            is_paid=req.is_paid,
        )

    @app.post("/api/v1/providers/{provider_id}/test")
    async def test_provider(provider_id: str) -> Dict[str, Any]:
        return mf.providers.test_provider(provider_id)

    # --- WebSocket streaming events ---
    @app.websocket("/ws/events")
    async def ws_events(websocket: WebSocket) -> None:
        await websocket.accept()
        loop = asyncio.get_running_loop()
        q, unsub = mf.events.subscribe_async(loop=loop)
        try:
            await websocket.send_json(
                {"topic": "ws.connected", "payload": {"mode": mf.config.mode.value, "version": "1.0.0"}}
            )
            while True:
                evt = await q.get()
                await websocket.send_json(evt)
        except WebSocketDisconnect:
            pass
        finally:
            unsub()

    return app
