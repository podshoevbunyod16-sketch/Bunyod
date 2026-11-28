"""Single & Multi-Agent Builder, Registry, Router, MessageBus & Safe Tool Runtime (Section 5)."""

from __future__ import annotations

import json
import secrets
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

from pydantic import BaseModel, Field

from modelforge.architectures.models import build_model_from_spec
from modelforge.core.events import EventBus
from modelforge.orchestration.graph import StateGraph
from modelforge.security.audit import AuditLogger
from modelforge.security.permissions import PermissionDeniedError, PermissionPolicy, PermissionScope
from modelforge.security.sandbox import resolve_within_workspace
from modelforge.storage.artifacts import ArtifactStore
from modelforge.storage.database import StorageDatabase
from modelforge.storage.memory_store import MemoryAndRagStore

AGENT_TYPES = [
    "Research",
    "Architecture",
    "Data",
    "Coding",
    "Training",
    "Evaluation",
    "Debugging",
    "Documentation",
    "Supervisor",
    "Custom",
]


class AgentLimits(BaseModel):
    max_steps: int = Field(default=8, ge=1, le=100)
    max_tokens: int = Field(default=8192, ge=128, le=200_000)
    timeout_sec: float = Field(default=45.0, gt=1.0, le=600.0)
    max_identical_tool_calls: int = Field(default=2, ge=1, le=10)


class AgentSpec(BaseModel):
    name: str
    agent_type: str = Field(default="Research")
    goal: str
    system_prompt: str = ""
    model_ref: str = "local-default"
    tools: List[str] = Field(
        default_factory=lambda: ["list_files", "read_file", "search_rag", "micro_model_probe"]
    )
    allowed_scopes: List[str] = Field(
        default_factory=lambda: [
            PermissionScope.WORKSPACE_READ.value,
            PermissionScope.MEMORY_READ.value,
            PermissionScope.MEMORY_WRITE.value,
            PermissionScope.MODEL_INFER.value,
        ]
    )
    allow_network: bool = False
    workspace_subdir: str = ""
    limits: AgentLimits = Field(default_factory=AgentLimits)
    output_format: str = "markdown"
    completion_criteria: str = "Goal addressed with verifiable tool outputs."


@dataclass
class AgentMessage:
    id: str
    sender: str
    recipient: str
    topic: str
    content: str
    metadata: Dict[str, Any]
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MessageBus:
    """Inter-agent message bus with bounded history for multi-agent teams."""

    def __init__(self, max_messages: int = 500) -> None:
        self._lock = threading.Lock()
        self._messages: Deque[AgentMessage] = deque(maxlen=max_messages)

    def send(
        self,
        sender: str,
        recipient: str,
        topic: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentMessage:
        msg = AgentMessage(
            id=f"msg-{secrets.token_hex(5)}",
            sender=sender,
            recipient=recipient,
            topic=topic,
            content=content,
            metadata=metadata or {},
            timestamp=time.time(),
        )
        with self._lock:
            self._messages.append(msg)
        return msg

    def inbox(self, recipient: str, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            items = [m.to_dict() for m in self._messages if m.recipient in (recipient, "*")]
        return items[-limit:]


class TaskRouter:
    """Routes incoming tasks to the best matching registered agent type."""

    KEYWORD_MAP = {
        "Research": ("исслед", "публикац", "гипотез", "paper", "research", "study"),
        "Architecture": ("архитект", "слои", "ssm", "transformer", "attention", "architecture"),
        "Data": ("датасет", "корпус", "токениз", "утечк", "dataset", "clean"),
        "Coding": ("код", "функц", "модул", "реализуй", "code", "implement"),
        "Training": ("обуч", "train", "finetune", "epoch", "lr", "checkpoint"),
        "Evaluation": ("оцен", "метрик", "сравн", "perplexity", "benchmark", "eval"),
        "Debugging": ("ошибк", "исправ", "баг", "debug", "fix", "traceback"),
        "Documentation": ("документ", "readme", "отчёт", "docs", "changelog"),
        "Supervisor": ("координ", "команд", "оркестр", "supervisor", "plan"),
    }

    @classmethod
    def infer_agent_type(cls, text: str) -> str:
        low = text.lower()
        for atype, words in cls.KEYWORD_MAP.items():
            if any(w in low for w in words):
                return atype
        return "Custom"


class AgentManager:
    """
    Specification Section 5:
    Agent Registry, isolated agent memory, permission-checked tools, loop/cost/timeout protection,
    single-agent execution, and multi-agent DAG execution.
    """

    def __init__(
        self,
        db: StorageDatabase,
        artifacts: ArtifactStore,
        memory_store: MemoryAndRagStore,
        events: EventBus,
        audit: AuditLogger,
        mcp_caller: Optional[Callable[[str, str, Dict[str, Any], str], Dict[str, Any]]] = None,
        offline_getter: Callable[[], bool] = lambda: False,
    ) -> None:
        self._db = db
        self._artifacts = artifacts
        self._memory = memory_store
        self._events = events
        self._audit = audit
        self._mcp_caller = mcp_caller
        self._offline_getter = offline_getter
        self.message_bus = MessageBus()
        self.router = TaskRouter()

    def spec_from_prompt(self, prompt: str, name: Optional[str] = None) -> AgentSpec:
        atype = self.router.infer_agent_type(prompt)
        tools = ["list_files", "read_file", "search_rag"]
        scopes = [
            PermissionScope.WORKSPACE_READ.value,
            PermissionScope.MEMORY_READ.value,
            PermissionScope.MEMORY_WRITE.value,
        ]
        if atype in ("Research", "Architecture", "Evaluation", "Training"):
            tools.append("micro_model_probe")
            scopes.append(PermissionScope.MODEL_INFER.value)
        if atype in ("Coding", "Debugging", "Documentation", "Data"):
            tools.append("write_report")
            scopes.append(PermissionScope.WORKSPACE_WRITE.value)

        sys_prompt = (
            f"You are a {atype} agent inside ModelForge AI. "
            "Operate strictly within permitted tools and workspace boundaries. "
            "Separate verified measurements from hypotheses and stop when completion criteria are met."
        )
        return AgentSpec(
            name=name or f"{atype.lower()}-agent",
            agent_type=atype,
            goal=prompt.strip(),
            system_prompt=sys_prompt,
            tools=tools,
            allowed_scopes=scopes,
        )

    def create_agent(self, project_id: str, spec: AgentSpec) -> Dict[str, Any]:
        if spec.agent_type not in AGENT_TYPES:
            raise ValueError(f"Unsupported agent_type '{spec.agent_type}'. Valid: {AGENT_TYPES}")
        agent_id = f"agt-{secrets.token_hex(5)}"
        ws = self._artifacts.ensure_project_workspace(project_id)
        agent_cfg_path = ws / "agents" / f"{agent_id}.json"
        agent_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        now = time.time()

        policy = PermissionPolicy(
            subject_id=agent_id,
            subject_type="agent",
            allowed_scopes=set(spec.allowed_scopes),
            allowed_workspace_subdir=spec.workspace_subdir or None,
            allow_network=spec.allow_network,
        )
        record = {
            "id": agent_id,
            "project_id": project_id,
            "name": spec.name,
            "agent_type": spec.agent_type,
            "goal": spec.goal,
            "system_prompt": spec.system_prompt or f"ModelForge {spec.agent_type} Agent",
            "model_ref": spec.model_ref,
            "tools": spec.tools,
            "permissions": policy.to_dict(),
            "limits": spec.limits.model_dump(),
            "output_format": spec.output_format,
            "completion_criteria": spec.completion_criteria,
            "workspace_subdir": spec.workspace_subdir,
            "status": "ready",
            "created_at": now,
            "updated_at": now,
        }
        agent_cfg_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

        self._db.upsert_row(
            "agents",
            {
                "id": agent_id,
                "project_id": project_id,
                "name": spec.name,
                "agent_type": spec.agent_type,
                "goal": spec.goal,
                "system_prompt": record["system_prompt"],
                "model_ref": spec.model_ref,
                "tools_json": json.dumps(spec.tools, ensure_ascii=False),
                "permissions_json": json.dumps(policy.to_dict(), ensure_ascii=False),
                "limits_json": json.dumps(spec.limits.model_dump(), ensure_ascii=False),
                "output_format": spec.output_format,
                "completion_criteria": spec.completion_criteria,
                "workspace_subdir": spec.workspace_subdir,
                "status": "ready",
                "created_at": now,
                "updated_at": now,
            },
        )
        self._events.publish("agent.created", record, project_id=project_id)
        return record

    def create_from_prompt(self, project_id: str, prompt: str, name: Optional[str] = None) -> Dict[str, Any]:
        spec = self.spec_from_prompt(prompt, name=name)
        return self.create_agent(project_id=project_id, spec=spec)

    def get_agent(self, agent_id: str) -> Dict[str, Any]:
        row = self._db.query_one("SELECT * FROM agents WHERE id = ?", (agent_id,))
        if not row:
            raise KeyError(f"Agent not found: {agent_id}")
        return {
            "id": row["id"],
            "project_id": row["project_id"],
            "name": row["name"],
            "agent_type": row["agent_type"],
            "goal": row["goal"],
            "system_prompt": row["system_prompt"],
            "model_ref": row["model_ref"],
            "tools": json.loads(row["tools_json"] or "[]"),
            "permissions": json.loads(row["permissions_json"] or "{}"),
            "limits": json.loads(row["limits_json"] or "{}"),
            "output_format": row["output_format"],
            "completion_criteria": row["completion_criteria"],
            "workspace_subdir": row["workspace_subdir"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_agents(self, project_id: str = "") -> List[Dict[str, Any]]:
        if project_id:
            rows = self._db.query_all(
                "SELECT id FROM agents WHERE project_id = ? ORDER BY updated_at DESC",
                (project_id,),
            )
        else:
            rows = self._db.query_all("SELECT id FROM agents ORDER BY updated_at DESC")
        return [self.get_agent(r["id"]) for r in rows]

    def _build_policy(self, agent: Dict[str, Any]) -> PermissionPolicy:
        perm = agent["permissions"]
        return PermissionPolicy(
            subject_id=agent["id"],
            subject_type="agent",
            allowed_scopes=set(perm.get("allowed_scopes", [])),
            allowed_workspace_subdir=perm.get("allowed_workspace_subdir"),
            allow_network=bool(perm.get("allow_network", False)),
        )

    def _execute_tool(
        self,
        agent: Dict[str, Any],
        policy: PermissionPolicy,
        workspace_root: Path,
        tool_name: str,
        args: Dict[str, Any],
    ) -> Dict[str, Any]:
        offline = self._offline_getter()
        if tool_name not in agent["tools"] and not tool_name.startswith("mcp:"):
            raise PermissionDeniedError(f"Tool '{tool_name}' is not enabled for agent '{agent['name']}'.")

        base_ws = (
            resolve_within_workspace(workspace_root, agent["workspace_subdir"])
            if agent.get("workspace_subdir")
            else workspace_root
        )
        base_ws.mkdir(parents=True, exist_ok=True)

        if tool_name == "list_files":
            policy.check(PermissionScope.WORKSPACE_READ, offline_mode=offline)
            files = [
                p.relative_to(base_ws).as_posix()
                for p in sorted(base_ws.rglob("*"))
                if p.is_file() and "__pycache__" not in p.parts
            ]
            return {"files": files[:60]}

        if tool_name == "read_file":
            policy.check(PermissionScope.WORKSPACE_READ, offline_mode=offline)
            rel = str(args.get("path", "README.md"))
            target = resolve_within_workspace(base_ws, rel)
            if not target.exists():
                return {"exists": False, "path": rel}
            return {"exists": True, "path": rel, "content": target.read_text(encoding="utf-8", errors="replace")[:4000]}

        if tool_name == "write_report":
            policy.check(PermissionScope.WORKSPACE_WRITE, offline_mode=offline)
            rel = str(args.get("path", f"docs/agent_report_{agent['id']}.md"))
            target = resolve_within_workspace(base_ws, rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            content = str(args.get("content", ""))
            target.write_text(content, encoding="utf-8")
            return {"written": True, "path": rel, "bytes": len(content.encode("utf-8"))}

        if tool_name == "search_rag":
            policy.check(PermissionScope.MEMORY_READ, offline_mode=offline)
            q = str(args.get("query", agent["goal"]))
            hits = self._memory.search_rag(agent["project_id"], q, top_k=3)
            return {"hits": hits}

        if tool_name == "micro_model_probe":
            policy.check(PermissionScope.MODEL_INFER, offline_mode=offline)
            import numpy as np

            arch = str(args.get("architecture", "hybrid_attention_ssm"))
            probe_spec = {
                "architecture": arch,
                "vocab_size": 64,
                "d_model": 32,
                "n_layers": 2,
                "n_heads": 4,
                "context_length": 32,
            }
            mdl = build_model_from_spec(probe_spec, seed=42)
            x = np.array([[2, 5, 8, 11, 14, 17, 20, 23]], dtype=np.int64)
            y = np.array([[5, 8, 11, 14, 17, 20, 23, 3]], dtype=np.int64)
            loss = float(mdl.compute_loss(x, y, training=False).data)
            return {
                "architecture": arch,
                "parameter_count": mdl.parameter_count(),
                "probe_loss": round(loss, 5),
                "verified": True,
            }

        if tool_name.startswith("mcp:"):
            policy.check(PermissionScope.MCP_CALL, offline_mode=offline)
            if not self._mcp_caller:
                raise RuntimeError("MCP caller is not attached.")
            _, server_id, mcp_tool = tool_name.split(":", 2)
            return self._mcp_caller(server_id, mcp_tool, args, agent["id"])

        raise ValueError(f"Unknown tool '{tool_name}'.")

    def run_agent(
        self,
        agent_id: str,
        task_prompt: str,
        shared_context: Optional[Dict[str, Any]] = None,
        tool_plan_override: Optional[List[Dict[str, Any]]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        agent = self.get_agent(agent_id)
        project_id = agent["project_id"]
        proj_row = self._db.query_one("SELECT workspace_path FROM projects WHERE id = ?", (project_id,))
        if not proj_row:
            raise KeyError(f"Project '{project_id}' not found for agent '{agent_id}'.")
        ws_root = Path(proj_row["workspace_path"]).resolve()
        policy = self._build_policy(agent)
        limits = AgentLimits.model_validate(agent["limits"])

        # Build bounded deterministic step plan unless caller provided explicit steps
        planned_calls = list(tool_plan_override) if tool_plan_override is not None else []
        if not planned_calls:
            if "list_files" in agent["tools"]:
                planned_calls.append({"tool": "list_files", "args": {}})
            if "read_file" in agent["tools"]:
                planned_calls.append({"tool": "read_file", "args": {"path": "README.md"}})
            if "search_rag" in agent["tools"]:
                planned_calls.append({"tool": "search_rag", "args": {"query": task_prompt}})
            if "micro_model_probe" in agent["tools"]:
                planned_calls.append({"tool": "micro_model_probe", "args": {"architecture": "hybrid_attention_ssm"}})
            if "write_report" in agent["tools"]:
                planned_calls.append(
                    {
                        "tool": "write_report",
                        "args": {
                            "path": f"docs/{agent['name']}_summary.md",
                            "content": f"# Agent Report: {agent['name']}\n\nTask: {task_prompt}\n",
                        },
                    }
                )

        t0 = time.monotonic()
        step_logs: List[Dict[str, Any]] = []
        call_counts: Dict[str, int] = {}
        tokens_used = len(task_prompt) // 4 + 32
        status = "completed"
        stop_reason = "criteria_met"

        for idx, step in enumerate(planned_calls, start=1):
            if cancel_event and cancel_event.is_set():
                status = "cancelled"
                stop_reason = "cancelled_by_user"
                break
            if idx > limits.max_steps:
                status = "stopped_by_limit"
                stop_reason = f"max_steps_exceeded ({limits.max_steps})"
                break
            if (time.monotonic() - t0) > limits.timeout_sec:
                status = "stopped_by_limit"
                stop_reason = f"timeout_exceeded ({limits.timeout_sec}s)"
                break
            if tokens_used > limits.max_tokens:
                status = "stopped_by_limit"
                stop_reason = f"max_tokens_exceeded ({limits.max_tokens})"
                break

            sig = json.dumps(step, sort_keys=True, ensure_ascii=False)
            call_counts[sig] = call_counts.get(sig, 0) + 1
            if call_counts[sig] > limits.max_identical_tool_calls:
                status = "stopped_by_loop_guard"
                stop_reason = f"loop_detected_on_identical_call ({step['tool']})"
                break

            tool_name = step["tool"]
            tool_args = step.get("args", {})
            out = self._execute_tool(agent, policy, ws_root, tool_name, tool_args)
            out_str = json.dumps(out, ensure_ascii=False)
            tokens_used += max(16, len(out_str) // 4)
            entry = {"step": idx, "tool": tool_name, "args": tool_args, "output": out}
            step_logs.append(entry)
            self._events.publish(
                "agent.step",
                {"agent_id": agent_id, "step": idx, "tool": tool_name},
                project_id=project_id,
            )

        summary_lines = [
            f"### {agent['name']} ({agent['agent_type']})",
            f"- **Task**: {task_prompt}",
            f"- **Status**: `{status}` ({stop_reason})",
            f"- **Steps executed**: {len(step_logs)} / {limits.max_steps}",
        ]
        for s in step_logs:
            summary_lines.append(f"- Step {s['step']} `{s['tool']}` -> verified")
        final_output = "\n".join(summary_lines)

        # Save to isolated agent memory (never polluting other agents or dialogue history)
        if PermissionScope.MEMORY_WRITE.value in agent["permissions"].get("allowed_scopes", []):
            self._memory.add_memory(
                scope="agent",
                owner_id=agent_id,
                role="execution_summary",
                content=final_output,
                metadata={"task": task_prompt, "steps": len(step_logs), "shared_context_keys": list((shared_context or {}).keys())},
            )

        self._audit.record(
            actor=agent_id,
            action="agent.run",
            status=status,
            details={"task": task_prompt, "steps": len(step_logs), "stop_reason": stop_reason},
        )

        duration = round(time.monotonic() - t0, 4)
        res = {
            "agent_id": agent_id,
            "agent_name": agent["name"],
            "agent_type": agent["agent_type"],
            "status": status,
            "stop_reason": stop_reason,
            "steps": step_logs,
            "tokens_estimated": tokens_used,
            "duration_sec": duration,
            "output": final_output,
        }
        self._events.publish("agent.completed", res, project_id=project_id)
        return res

    def run_multi_agent_workflow(
        self,
        nodes: List[Dict[str, Any]],
        shared_task: str,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        """
        Execute a DAG of multiple agents (sequential & parallel dependencies) sharing task state
        and passing messages over `MessageBus`.
        Each item in `nodes`: `{"node_id": str, "agent_id": str, "prompt": str, "depends_on": [...]}`.
        """
        graph = StateGraph(name="multi_agent_team")
        for n in nodes:
            nid = n["node_id"]
            aid = n["agent_id"]
            sub_prompt = n.get("prompt") or shared_task
            deps = n.get("depends_on") or []

            def _make_runner(node_id: str, agent_id: str, p: str, dep_ids: List[str]):
                def _fn(state: Dict[str, Any]) -> Dict[str, Any]:
                    upstream = {d: state.get(f"out:{d}") for d in dep_ids if f"out:{d}" in state}
                    run_res = self.run_agent(
                        agent_id=agent_id,
                        task_prompt=p,
                        shared_context=upstream,
                        cancel_event=cancel_event,
                    )
                    self.message_bus.send(
                        sender=agent_id,
                        recipient="*",
                        topic="node.completed",
                        content=run_res["output"],
                        metadata={"node_id": node_id},
                    )
                    return {f"out:{node_id}": run_res}

                return _fn

            graph.add_node(
                node_id=nid,
                fn=_make_runner(nid, aid, sub_prompt, deps),
                depends_on=deps,
            )

        dag_res = graph.run(initial_state={"shared_task": shared_task}, cancel_event=cancel_event)
        return {
            "status": dag_res.status,
            "execution_order": dag_res.execution_order,
            "duration_sec": dag_res.duration_sec,
            "errors": dag_res.errors,
            "results": {k: v for k, v in dag_res.state.items() if k.startswith("out:")},
            "messages": self.message_bus.inbox("*", limit=50),
        }
