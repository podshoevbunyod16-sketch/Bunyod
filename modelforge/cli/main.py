"""Unified CLI implementing all Section 13 commands for Desktop, Linux, Windows, macOS & Termux ARM64."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Optional

from modelforge.core.runtime import ModelForgeCore
from modelforge.inference.engine import GenerationConfig
from modelforge.training.engine import TrainingRunConfig


def _print_output(data: Any, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2))


def _ensure_default_project(core: ModelForgeCore, project_id: str) -> str:
    if project_id:
        return project_id
    existing = core.projects.list_projects()
    if existing:
        return existing[0].id
    created = core.projects.create_project(
        name="Default ModelForge Project",
        description="Auto-initialized workspace from CLI",
    )
    return created.id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modelforge",
        description="ModelForge AI — Cross-platform AI Model, Agent, Research & Prompt-to-Project CLI",
    )
    parser.add_argument("--home", default="", help="Override MODELFORGE_HOME directory")
    parser.add_argument("--json", action="store_true", help="Output structured JSON")

    sub = parser.add_subparsers(dest="command")

    # modelforge chat
    p_chat = sub.add_parser("chat", help="Interactive or single-turn AI Chat / Prompt Workspace")
    p_chat.add_argument("message", nargs="?", default="", help="Prompt message (omit for interactive REPL)")
    p_chat.add_argument("--project", default="", help="Project ID")
    p_chat.add_argument("--provider", default="provider-local", help="LLM Provider ID")
    p_chat.add_argument("--model", default="", help="Explicit Model ID")

    # modelforge create {project|model|agent}
    p_create = sub.add_parser("create", help="Create a project, neural model, or autonomous agent")
    create_sub = p_create.add_subparsers(dest="create_kind", required=True)

    c_proj = create_sub.add_parser("project", help="Create an isolated project workspace")
    c_proj.add_argument("--name", required=True, help="Project name")
    c_proj.add_argument("--description", default="", help="Project description")

    c_mdl = create_sub.add_parser("model", help="Create a model & tokenizer from prompt or architecture")
    c_mdl.add_argument("--prompt", default="Создай компактную языковую модель для русского и таджикского текста с экспериментальной архитектурой")
    c_mdl.add_argument("--name", default=None, help="Optional model name")
    c_mdl.add_argument("--project", default="", help="Project ID")
    c_mdl.add_argument("--arch", default="", help="Explicit architecture override")

    c_agt = create_sub.add_parser("agent", help="Create an autonomous agent from prompt")
    c_agt.add_argument("--prompt", default="Создай исследовательского агента для безопасных тестов маленьких моделей")
    c_agt.add_argument("--name", default=None, help="Agent name")
    c_agt.add_argument("--project", default="", help="Project ID")

    # modelforge train
    p_train = sub.add_parser("train", help="Train a model from scratch with checkpointing")
    p_train.add_argument("--model", required=True, help="Model ID")
    p_train.add_argument("--dataset", default="", help="Dataset ID (optional)")
    p_train.add_argument("--epochs", type=int, default=4)
    p_train.add_argument("--batch-size", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=3e-3)
    p_train.add_argument("--resume", default="", help="Checkpoint path to resume from")

    # modelforge finetune
    p_ft = sub.add_parser("finetune", help="Fine-tune or transfer-learn an existing trained model")
    p_ft.add_argument("--model", required=True, help="Model ID")
    p_ft.add_argument("--dataset", default="", help="Dataset ID or inline corpus")
    p_ft.add_argument("--text-file", default="", help="Optional TXT/JSONL file to ingest for fine-tuning")
    p_ft.add_argument("--epochs", type=int, default=3)
    p_ft.add_argument("--lr", type=float, default=1.5e-3)
    p_ft.add_argument("--transfer", action="store_true", help="Freeze embedding & prefix layers (transfer learning)")

    # modelforge research
    p_res = sub.add_parser("research", help="Run controlled comparative architecture study")
    p_res.add_argument("--question", default="Compare decoder_transformer and hybrid_attention_ssm on RU-TJ text")
    p_res.add_argument("--project", default="", help="Project ID")
    p_res.add_argument("--baseline", default="decoder_transformer")
    p_res.add_argument("--candidate", default="hybrid_attention_ssm")
    p_res.add_argument("--epochs", type=int, default=3)

    # modelforge benchmark
    p_bench = sub.add_parser("benchmark", help="Evaluate or compare models with real metrics")
    p_bench.add_argument("--models", nargs="+", required=True, help="One or more Model IDs")
    p_bench.add_argument("--dataset", default="", help="Optional Dataset ID")

    # modelforge project {edit|list|rollback}
    p_proj = sub.add_parser("project", help="Project workspace operations & Prompt-to-Project editing")
    proj_sub = p_proj.add_subparsers(dest="project_cmd", required=True)

    pe = proj_sub.add_parser("edit", help="Edit a project via natural-language prompt with diff & tests")
    pe.add_argument("--project", default="", help="Project ID")
    pe.add_argument("--prompt", required=True, help="Natural language edit instruction")

    pl = proj_sub.add_parser("list", help="List projects")
    pl.add_argument("--query", default="")

    prb = proj_sub.add_parser("rollback", help="Rollback a Prompt-to-Project edit by Edit ID")
    prb.add_argument("--edit-id", required=True, help="Edit ID (`edit-...`)")

    # modelforge model {list|generate}
    p_model = sub.add_parser("model", help="List models or run text generation")
    model_sub = p_model.add_subparsers(dest="model_cmd", required=True)
    ml = model_sub.add_parser("list", help="List registered models")
    ml.add_argument("--project", default="")

    mg = model_sub.add_parser("generate", help="Generate text from a model checkpoint")
    mg.add_argument("--model", required=True, help="Model ID")
    mg.add_argument("--prompt", default="Привет")
    mg.add_argument("--max-tokens", type=int, default=28)
    mg.add_argument("--temperature", type=float, default=0.8)

    # modelforge agent {list|run}
    p_agent = sub.add_parser("agent", help="List or execute autonomous agents")
    agent_sub = p_agent.add_subparsers(dest="agent_cmd", required=True)
    al = agent_sub.add_parser("list", help="List registered agents")
    al.add_argument("--project", default="")

    ar = agent_sub.add_parser("run", help="Execute a bounded task with an agent")
    ar.add_argument("--agent", required=True, help="Agent ID")
    ar.add_argument("--prompt", required=True, help="Task prompt")

    # modelforge mcp {list|probe|call}
    p_mcp = sub.add_parser("mcp", help="Inspect, probe, and invoke MCP servers")
    mcp_sub = p_mcp.add_subparsers(dest="mcp_cmd", required=True)
    mcp_sub.add_parser("list", help="List registered MCP servers")

    mp = mcp_sub.add_parser("probe", help="Probe an MCP server and discover tools")
    mp.add_argument("--server", default="mcp-builtin-local")

    mc = mcp_sub.add_parser("call", help="Call an MCP tool")
    mc.add_argument("--server", default="mcp-builtin-local")
    mc.add_argument("--tool", required=True)
    mc.add_argument("--args", default="{}", help="JSON arguments object")

    # modelforge doctor
    sub.add_parser("doctor", help="Run full platform, DB, security, RAM, MCP & plugin diagnostics")

    # modelforge export
    p_exp = sub.add_parser("export", help="Export a model to gguf, safetensors, pt, json, or yaml")
    p_exp.add_argument("--model", required=True, help="Model ID")
    p_exp.add_argument("--format", default="gguf", choices=["gguf", "safetensors", "pt", "json", "yaml"])
    p_exp.add_argument("--output", default="", help="Optional output file path")

    # modelforge server
    p_srv = sub.add_parser("server", help="Start ModelForge FastAPI + WebSocket Core server")
    p_srv.add_argument("--host", default="0.0.0.0")
    p_srv.add_argument("--port", type=int, default=8765)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    home = Path(args.home).resolve() if args.home else None
    core = ModelForgeCore(home_dir=home)

    try:
        if not args.command:
            info = core.doctor()
            _print_output(
                {
                    "app": "ModelForge AI",
                    "version": info["version"],
                    "mode": info["mode"],
                    "low_memory_mode": info["platform"]["low_memory_mode"],
                    "counts": info["counts"],
                    "hint": "Run `modelforge --help` or `modelforge server`.",
                },
                as_json=args.json,
            )
            return 0

        if args.command == "doctor":
            _print_output(core.doctor(), as_json=args.json)
            return 0

        if args.command == "chat":
            pid = _ensure_default_project(core, args.project)
            if args.message:
                out = core.chat_workspace(
                    message=args.message,
                    project_id=pid,
                    provider_id=args.provider,
                    model_id=args.model,
                )
                _print_output(out, as_json=args.json)
                return 0
            print("ModelForge AI Interactive Chat (type ':quit' to exit)")
            while True:
                try:
                    line = input("you> ").strip()
                except EOFError:
                    break
                if line in (":quit", ":exit", "quit", "exit"):
                    break
                if not line:
                    continue
                resp = core.chat_workspace(message=line, project_id=pid, provider_id=args.provider, model_id=args.model)
                print(f"modelforge> {resp['assistant_reply']}")
            return 0

        if args.command == "create":
            if args.create_kind == "project":
                proj = core.projects.create_project(name=args.name, description=args.description)
                _print_output(proj.to_dict(), as_json=args.json)
                return 0
            if args.create_kind == "model":
                pid = _ensure_default_project(core, args.project)
                spec = core.models.spec_from_prompt(args.prompt, name=args.name)
                if args.arch:
                    spec = spec.model_copy(update={"architecture": args.arch})
                mdl = core.models.create_model(project_id=pid, spec=spec, provenance={"prompt": args.prompt})
                _print_output(mdl, as_json=args.json)
                return 0
            if args.create_kind == "agent":
                pid = _ensure_default_project(core, args.project)
                agt = core.agents.create_from_prompt(project_id=pid, prompt=args.prompt, name=args.name)
                _print_output(agt, as_json=args.json)
                return 0

        if args.command == "train":
            cfg = TrainingRunConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                mode="scratch",
                resume_from_checkpoint=args.resume,
            )
            res = core.training.train(model_id=args.model, config=cfg, dataset_id=args.dataset)
            _print_output(res, as_json=args.json)
            return 0

        if args.command == "finetune":
            minfo = core.models.get_model(args.model)
            ds_id = args.dataset
            if args.text_file:
                ingested = core.datasets.ingest_file(minfo["project_id"], Path(args.text_file))
                ds_id = ingested["id"]
            cfg = TrainingRunConfig(
                epochs=args.epochs,
                learning_rate=args.lr,
                mode="transfer" if args.transfer else "finetune",
            )
            res = core.training.train(model_id=args.model, config=cfg, dataset_id=ds_id)
            _print_output(res, as_json=args.json)
            return 0

        if args.command == "research":
            pid = _ensure_default_project(core, args.project)
            study = core.research.run_comparative_study(
                project_id=pid,
                question=args.question,
                baseline_arch=args.baseline,
                candidate_arch=args.candidate,
                epochs=args.epochs,
            )
            _print_output(study, as_json=args.json)
            return 0

        if args.command == "benchmark":
            if len(args.models) == 1:
                out = core.evaluation.evaluate_model(args.models[0], dataset_id=args.dataset)
            else:
                out = core.evaluation.compare_models(args.models, dataset_id=args.dataset)
            _print_output(out, as_json=args.json)
            return 0

        if args.command == "project":
            if args.project_cmd == "list":
                _print_output([p.to_dict() for p in core.projects.list_projects(args.query)], as_json=args.json)
                return 0
            if args.project_cmd == "edit":
                pid = _ensure_default_project(core, args.project)
                res = core.prompt_editor.apply_edit(project_id=pid, prompt=args.prompt)
                _print_output(res, as_json=args.json)
                return 0
            if args.project_cmd == "rollback":
                res = core.prompt_editor.rollback_edit(args.edit_id)
                _print_output(res, as_json=args.json)
                return 0

        if args.command == "model":
            if args.model_cmd == "list":
                _print_output(core.models.list_models(project_id=args.project), as_json=args.json)
                return 0
            if args.model_cmd == "generate":
                out = core.inference.generate(
                    model_id=args.model,
                    prompt=args.prompt,
                    config=GenerationConfig(max_tokens=args.max_tokens, temperature=args.temperature),
                )
                _print_output(out, as_json=args.json)
                return 0

        if args.command == "agent":
            if args.agent_cmd == "list":
                _print_output(core.agents.list_agents(project_id=args.project), as_json=args.json)
                return 0
            if args.agent_cmd == "run":
                out = core.agents.run_agent(agent_id=args.agent, task_prompt=args.prompt)
                _print_output(out, as_json=args.json)
                return 0

        if args.command == "mcp":
            if args.mcp_cmd == "list":
                _print_output(core.mcp.list_servers(), as_json=args.json)
                return 0
            if args.mcp_cmd == "probe":
                _print_output(core.mcp.probe_and_discover_tools(args.server), as_json=args.json)
                return 0
            if args.mcp_cmd == "call":
                parsed_args = json.loads(args.args)
                _print_output(core.mcp.call_tool(args.server, args.tool, parsed_args), as_json=args.json)
                return 0

        if args.command == "export":
            out_path = Path(args.output) if args.output else None
            _print_output(core.registry.export_model(args.model, args.format, output_path=out_path), as_json=args.json)
            return 0

        if args.command == "server":
            import uvicorn

            from modelforge.backend.api import create_app

            app = create_app(core=core)
            uvicorn.run(app, host=args.host, port=args.port, log_level="info")
            return 0

        parser.print_help()
        return 1
    finally:
        if getattr(args, "command", None) != "server":
            core.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
