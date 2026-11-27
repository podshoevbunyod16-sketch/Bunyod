#!/usr/bin/env python3
"""End-to-end verification & benchmark script for ModelForge AI (no mocks)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from modelforge.core.runtime import ModelForgeCore
from modelforge.inference.engine import GenerationConfig
from modelforge.inference.gguf import inspect_gguf_v3
from modelforge.training.engine import TrainingRunConfig


def run_smoke_verification(work_home: Path) -> dict:
    core = ModelForgeCore(home_dir=work_home)
    try:
        proj = core.projects.create_project(
            name="Bilingual RU-TJ Verification",
            description="End-to-end smoke verification of ModelForge AI MVP",
            tags=["ru", "tg", "mvp"],
        )

        # 1. Create compact RU-TJ hybrid model & BPE tokenizer
        mdl = core.models.create_from_prompt(
            project_id=proj.id,
            prompt="Создай компактную языковую модель для русского и таджикского текста с экспериментальной архитектурой",
        )

        # 2. Real training loop with weight-delta & loss check
        train_out = core.training.train(
            model_id=mdl["id"],
            config=TrainingRunConfig(
                epochs=8,
                batch_size=4,
                learning_rate=6e-3,
                max_seq_len=48,
                seed=42,
            ),
        )
        assert train_out["metrics"]["weights_changed"] is True, "Weights did not mutate!"
        assert train_out["metrics"]["final_train_loss"] < train_out["metrics"]["initial_train_loss"]

        # 3. Reload checkpoint & generate text
        gen_out = core.inference.generate(
            model_id=mdl["id"],
            prompt="Салом ҷаҳон! Привет",
            config=GenerationConfig(max_tokens=24, temperature=0.65, seed=42),
        )
        assert gen_out["tokens_generated"] > 0

        # 4. Export GGUF v3 & Safetensors
        gguf_out = core.registry.export_model(mdl["id"], "gguf")
        gguf_header = inspect_gguf_v3(Path(gguf_out["path"]))
        st_out = core.registry.export_model(mdl["id"], "safetensors")

        # 5. Create & run autonomous Research agent
        agent = core.agents.create_from_prompt(
            project_id=proj.id,
            prompt="Создай исследовательского агента для проверки компактных моделей",
        )
        agent_run = core.agents.run_agent(
            agent_id=agent["id"],
            task_prompt="Проверь гибридную архитектуру и файлы рабочей директории",
        )
        assert agent_run["status"] == "completed"

        # 6. Prompt-to-Project edit + diff + rollback
        edit_out = core.prompt_editor.apply_edit(
            project_id=proj.id,
            prompt="Добавь таджикский язык и снизь RAM",
        )
        assert edit_out["test_report"]["passed"] is True
        rb_out = core.prompt_editor.rollback_edit(edit_out["id"])
        assert rb_out["rolled_back"] is True

        # 7. Built-in MCP STDIO server probe & call
        mcp_probe = core.mcp.probe_and_discover_tools("mcp-builtin-local")
        mcp_call = core.mcp.call_tool(
            "mcp-builtin-local",
            "analyze_corpus_stats",
            {"text": "Салом ҷаҳон! Компактная модель работает."},
        )
        assert mcp_call["status"] == "ok"

        doc = core.doctor()
        assert doc["healthy"] is True

        return {
            "smoke_passed": True,
            "platform": doc["platform"],
            "model_spec": mdl["spec"],
            "training_metrics": {
                "initial_train_loss": train_out["metrics"]["initial_train_loss"],
                "final_train_loss": train_out["metrics"]["final_train_loss"],
                "perplexity": train_out["metrics"]["perplexity"],
                "weight_l2_delta": train_out["metrics"]["weight_l2_delta"],
                "weights_changed": train_out["metrics"]["weights_changed"],
                "checkpoint_sha256": train_out["checkpoint"]["sha256"],
                "duration_sec": train_out["metrics"]["duration_sec"],
            },
            "generation": {
                "prompt": gen_out["prompt"],
                "generated_text": gen_out["generated_text"],
                "full_text": gen_out["full_text"],
                "tokens_generated": gen_out["tokens_generated"],
                "tokens_per_sec": gen_out["tokens_per_sec"],
            },
            "exports": {
                "gguf_v3_valid": gguf_header["valid_gguf"],
                "gguf_size_bytes": gguf_out["size_bytes"],
                "safetensors_size_bytes": st_out["size_bytes"],
            },
            "agent_status": agent_run["status"],
            "prompt_edit_diff_lines": len(edit_out["diff_text"].splitlines()),
            "rollback_verified": rb_out["rolled_back"],
            "mcp_tools_discovered": [t["name"] for t in mcp_probe["tools"]],
        }
    finally:
        core.shutdown()


def main() -> int:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    with tempfile.TemporaryDirectory() as tmp:
        report = run_smoke_verification(Path(tmp) / "mf_smoke")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
