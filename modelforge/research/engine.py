"""Research Engine: Question -> Sources -> Hypothesis -> Plan -> Controlled Experiment -> Conclusion."""

from __future__ import annotations

import json
import secrets
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional

from modelforge.core.events import EventBus
from modelforge.datasets.manager import DatasetManager
from modelforge.evaluation.metrics import EvaluationEngine
from modelforge.model_builder.builder import DEFAULT_MULTILINGUAL_SEED_CORPUS, ModelBuilder, ModelSpec
from modelforge.platforms.detector import PlatformInfo, get_resource_snapshot
from modelforge.security.sandbox import sanitize_untrusted_content
from modelforge.storage.database import StorageDatabase
from modelforge.training.engine import TrainingEngine, TrainingRunConfig


def _current_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except Exception:
        return "working-tree"


class ResearchEngine:
    """
    Specification Section 7:
    - Full cycle: question -> source review -> hypothesis -> plan -> implementation -> test ->
      measurement -> comparison -> conclusion -> next experiment.
    - Separates verified facts from hypotheses and assumptions.
    - Preserves negative results and failure reasons.
    - Never fabricates citations, publications, URLs, or metrics.
    """

    def __init__(
        self,
        db: StorageDatabase,
        events: EventBus,
        model_builder: ModelBuilder,
        dataset_manager: DatasetManager,
        training_engine: TrainingEngine,
        evaluation_engine: EvaluationEngine,
        platform_info: PlatformInfo,
        offline_getter: Callable[[], bool] = lambda: False,
    ) -> None:
        self._db = db
        self._events = events
        self._models = model_builder
        self._datasets = dataset_manager
        self._training = training_engine
        self._eval = evaluation_engine
        self._platform = platform_info
        self._offline_getter = offline_getter

    def collect_sources(
        self,
        query: str,
        user_sources: Optional[List[Dict[str, str]]] = None,
        search_github_online: bool = False,
    ) -> List[Dict[str, Any]]:
        """Record real user-provided or live GitHub API sources with URL, access date & untrusted tag."""
        collected: List[Dict[str, Any]] = []
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        for src in user_sources or []:
            collected.append(
                {
                    "title": src.get("title", "User-provided reference"),
                    "url": src.get("url", "local://user-note"),
                    "accessed_at": now_iso,
                    "license": src.get("license", "unspecified"),
                    "rationale": src.get("rationale", "Provided explicitly for research context."),
                    "source_type": "user_provided",
                }
            )

        if search_github_online and not self._offline_getter():
            try:
                import httpx

                with httpx.Client(timeout=8.0) as client:
                    resp = client.get(
                        "https://api.github.com/search/repositories",
                        params={"q": query, "per_page": 3, "sort": "stars"},
                        headers={"Accept": "application/vnd.github+json"},
                    )
                    if resp.status_code == 200:
                        items = resp.json().get("items", [])
                        for item in items:
                            sanitized = sanitize_untrusted_content(
                                item.get("html_url", ""),
                                str(item.get("description") or ""),
                            )
                            collected.append(
                                {
                                    "title": item.get("full_name", ""),
                                    "url": item.get("html_url", ""),
                                    "accessed_at": now_iso,
                                    "license": (item.get("license") or {}).get("spdx_id") or "unspecified",
                                    "rationale": sanitized["content"],
                                    "trust_level": sanitized["trust_level"],
                                    "source_type": "github_api_live",
                                }
                            )
            except Exception as exc:
                collected.append(
                    {
                        "title": "Online repository lookup skipped/failed",
                        "url": "https://api.github.com",
                        "accessed_at": now_iso,
                        "license": "n/a",
                        "rationale": f"Live lookup error: {type(exc).__name__}: {exc}",
                        "source_type": "diagnostic_note",
                    }
                )
        return collected

    def run_comparative_study(
        self,
        project_id: str,
        question: str,
        baseline_arch: str = "decoder_transformer",
        candidate_arch: str = "hybrid_attention_ssm",
        epochs: int = 3,
        d_model: int = 48,
        n_layers: int = 2,
        seed: int = 42,
        dataset_id: str = "",
        user_sources: Optional[List[Dict[str, str]]] = None,
        search_github_online: bool = False,
    ) -> Dict[str, Any]:
        sources = self.collect_sources(
            query=question,
            user_sources=user_sources,
            search_github_online=search_github_online,
        )

        # Ensure a fixed shared dataset so both architectures are tested under identical conditions
        if not dataset_id:
            ds = self._datasets.ingest_dataset(
                project_id=project_id,
                name="research-controlled-corpus",
                raw_content="\n".join(DEFAULT_MULTILINGUAL_SEED_CORPUS),
                fmt="txt",
                license_name="MIT",
                seed=seed,
            )
            dataset_id = ds["id"]
            dataset_hash = ds["sha256"]
        else:
            ds = self._datasets.get_dataset(dataset_id)
            dataset_hash = ds["sha256"]

        hypothesis = (
            f"HYPOTHESIS (unverified prior to run): On identical seed={seed} and dataset sha256={dataset_hash[:12]}, "
            f"candidate architecture '{candidate_arch}' achieves competitive or lower validation loss than "
            f"baseline '{baseline_arch}' at matching d_model={d_model}, n_layers={n_layers}."
        )
        assumptions = [
            f"Fixed small multilingual RU/TJ/EN corpus (dataset_id={dataset_id}).",
            f"Identical hyperparameters: d_model={d_model}, n_layers={n_layers}, epochs={epochs}, seed={seed}.",
            "Results reflect micro-scale convergence and must not be extrapolated to billion-parameter scale without new experiments.",
        ]

        code_ver = _current_git_commit()
        runs_recorded: List[Dict[str, Any]] = []

        for role, arch in [("baseline", baseline_arch), ("candidate", candidate_arch)]:
            exp_id = f"exp-{secrets.token_hex(5)}"
            t0 = time.monotonic()
            spec = ModelSpec(
                name=f"{role}-{arch}",
                architecture=arch,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=4 if d_model % 4 == 0 else 2,
                vocab_size=256,
                context_length=64,
                batch_size=4,
                epochs=epochs,
                seed=seed,
            )
            mdl = self._models.create_model(project_id=project_id, spec=spec)
            t_cfg = TrainingRunConfig(
                epochs=epochs,
                batch_size=4,
                learning_rate=3e-3,
                max_seq_len=48,
                seed=seed,
            )
            train_out = self._training.train(model_id=mdl["id"], config=t_cfg, dataset_id=dataset_id)
            eval_out = self._eval.evaluate_model(model_id=mdl["id"], dataset_id=dataset_id)
            dur = round(time.monotonic() - t0, 4)
            res_snap = get_resource_snapshot().to_dict()

            runs_recorded.append(
                {
                    "exp_id": exp_id,
                    "role": role,
                    "architecture": arch,
                    "model_id": mdl["id"],
                    "config": {**spec.model_dump(), "training": t_cfg.to_dict()},
                    "duration_sec": dur,
                    "resources": res_snap,
                    "metrics": {
                        "initial_train_loss": train_out["metrics"]["initial_train_loss"],
                        "final_train_loss": train_out["metrics"]["final_train_loss"],
                        "val_loss": eval_out["loss"],
                        "perplexity": eval_out["perplexity"],
                        "next_token_accuracy": eval_out["next_token_accuracy"],
                        "parameter_count": eval_out["parameter_count"],
                        "weights_changed": train_out["metrics"]["weights_changed"],
                    },
                    "checkpoint_path": train_out["checkpoint"]["checkpoint_path"],
                    "logs": [f"Epoch {h['epoch']}: train={h['train_loss']} val={h['val_loss']}" for h in train_out["metrics"]["history"]],
                }
            )

        base_run, cand_run = runs_recorded[0], runs_recorded[1]
        cand_better = cand_run["metrics"]["val_loss"] <= base_run["metrics"]["val_loss"]
        is_negative = not cand_better

        if cand_better:
            conclusion = (
                f"VERIFIED FACT: Candidate '{candidate_arch}' achieved val_loss={cand_run['metrics']['val_loss']} "
                f"(ppl={cand_run['metrics']['perplexity']}) vs baseline '{baseline_arch}' "
                f"val_loss={base_run['metrics']['val_loss']} (ppl={base_run['metrics']['perplexity']}). Hypothesis supported on this corpus."
            )
            next_step = (
                f"Test '{candidate_arch}' across 3 random seeds (42, 1337, 2026) and longer sequence length (128) "
                "to verify variance."
            )
        else:
            conclusion = (
                f"NEGATIVE RESULT PRESERVED: Candidate '{candidate_arch}' yielded higher val_loss="
                f"{cand_run['metrics']['val_loss']} than baseline '{baseline_arch}' "
                f"val_loss={base_run['metrics']['val_loss']}. Hypothesis refuted under current learning rate / epoch budget."
            )
            next_step = (
                f"Tune learning rate (1e-3 vs 5e-3) or state decay initialization for '{candidate_arch}' before discarding."
            )

        now = time.time()
        saved_experiments: List[Dict[str, Any]] = []
        for r in runs_recorded:
            neg_flag = int(is_negative and r["role"] == "candidate")
            row = {
                "id": r["exp_id"],
                "project_id": project_id,
                "goal": question,
                "hypothesis": hypothesis,
                "sources_json": json.dumps(sources, ensure_ascii=False),
                "code_version": code_ver,
                "architecture": r["architecture"],
                "config_json": json.dumps(r["config"], ensure_ascii=False),
                "dataset_id": dataset_id,
                "dataset_hash": dataset_hash,
                "seed": seed,
                "environment_json": json.dumps(
                    {"python": self._platform.python_version, "os": self._platform.os_name},
                    ensure_ascii=False,
                ),
                "hardware_json": json.dumps(
                    {
                        "arch": self._platform.architecture,
                        "cpu_count": self._platform.cpu_count,
                        "total_ram_mb": self._platform.total_ram_mb,
                        "low_memory_mode": self._platform.low_memory_mode,
                    },
                    ensure_ascii=False,
                ),
                "duration_sec": r["duration_sec"],
                "resources_json": json.dumps(r["resources"], ensure_ascii=False),
                "metrics_json": json.dumps(r["metrics"], ensure_ascii=False),
                "checkpoint_path": r["checkpoint_path"],
                "logs_json": json.dumps(r["logs"], ensure_ascii=False),
                "status": "completed",
                "is_negative_result": neg_flag,
                "conclusion": conclusion,
                "next_experiment_suggestion": next_step,
                "created_at": now,
                "updated_at": now,
            }
            self._db.upsert_row("experiments", row)
            saved_experiments.append(self.get_experiment(r["exp_id"]))

        report = {
            "question": question,
            "facts_vs_hypotheses": {
                "prior_hypothesis": hypothesis,
                "assumptions": assumptions,
                "measured_facts": {
                    "baseline": base_run["metrics"],
                    "candidate": cand_run["metrics"],
                    "dataset_sha256": dataset_hash,
                    "seed": seed,
                },
            },
            "sources": sources,
            "is_negative_result": is_negative,
            "conclusion": conclusion,
            "next_experiment_suggestion": next_step,
            "experiments": saved_experiments,
        }
        self._events.publish("research.study_completed", report, project_id=project_id)
        return report

    def get_experiment(self, exp_id: str) -> Dict[str, Any]:
        row = self._db.query_one("SELECT * FROM experiments WHERE id = ?", (exp_id,))
        if not row:
            raise KeyError(f"Experiment not found: {exp_id}")
        return {
            "id": row["id"],
            "project_id": row["project_id"],
            "goal": row["goal"],
            "hypothesis": row["hypothesis"],
            "sources": json.loads(row["sources_json"] or "[]"),
            "code_version": row["code_version"],
            "architecture": row["architecture"],
            "config": json.loads(row["config_json"] or "{}"),
            "dataset_id": row["dataset_id"],
            "dataset_hash": row["dataset_hash"],
            "seed": int(row["seed"]),
            "environment": json.loads(row["environment_json"] or "{}"),
            "hardware": json.loads(row["hardware_json"] or "{}"),
            "duration_sec": float(row["duration_sec"]),
            "resources": json.loads(row["resources_json"] or "{}"),
            "metrics": json.loads(row["metrics_json"] or "{}"),
            "checkpoint_path": row["checkpoint_path"],
            "logs": json.loads(row["logs_json"] or "[]"),
            "status": row["status"],
            "is_negative_result": bool(row["is_negative_result"]),
            "conclusion": row["conclusion"],
            "next_experiment_suggestion": row["next_experiment_suggestion"],
            "created_at": row["created_at"],
        }

    def list_experiments(self, project_id: str = "") -> List[Dict[str, Any]]:
        if project_id:
            rows = self._db.query_all(
                "SELECT id FROM experiments WHERE project_id = ? ORDER BY created_at DESC",
                (project_id,),
            )
        else:
            rows = self._db.query_all("SELECT id FROM experiments ORDER BY created_at DESC")
        return [self.get_experiment(r["id"]) for r in rows]
