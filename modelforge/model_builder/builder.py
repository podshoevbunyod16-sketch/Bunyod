"""Model specification, resource estimator & prompt-driven model creation (Section 3)."""

from __future__ import annotations

import json
import secrets
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from modelforge.architectures.models import SUPPORTED_ARCHITECTURES, build_model_from_spec
from modelforge.core.events import EventBus
from modelforge.platforms.detector import PlatformInfo
from modelforge.storage.artifacts import ArtifactStore
from modelforge.storage.database import StorageDatabase
from modelforge.tokenizers.tokenizer import train_tokenizer, verify_multilingual_unicode

DEFAULT_MULTILINGUAL_SEED_CORPUS = [
    "Привет! Это компактная языковая модель лаборатории ModelForge AI.",
    "Модель поддерживает русский и таджикский текст и обучается воспроизводимо.",
    "Салом ҷаҳон! Забони тоҷикӣ ва забони русӣ дар як модели компактӣ кор мекунанд.",
    "Дар шаҳри Хуҷанд ва Душанбе пажӯҳишгарон меъмориҳои нави зеҳни сунъиро месанҷанд.",
    "Экспериментальная гибридная архитектура объединяет блоки внимания и пространства состояний.",
    "ModelForge AI builds, trains, evaluates, and exports compact neural language models.",
    "Токенизатор корректно кодирует кириллицу и таджикские буквы: ғ, ӣ, қ, ӯ, ҳ, ҷ.",
    "Обучение сохраняет проверяемые чекпоинты, метрики потерь и перплексии.",
]


class ModelSpec(BaseModel):
    name: str
    architecture: str = Field(default="decoder_transformer")
    n_layers: int = Field(default=2, ge=1, le=24)
    d_model: int = Field(default=64, ge=16, le=2048)
    n_heads: int = Field(default=4, ge=1, le=32)
    vocab_size: int = Field(default=320, ge=32, le=65536)
    context_length: int = Field(default=128, ge=16, le=4096)
    positional_encoding: str = Field(default="rope")  # rope | learned | sinusoidal
    activation: str = Field(default="gelu")  # gelu | silu | relu | tanh
    norm_type: str = Field(default="rmsnorm")  # rmsnorm | layernorm
    loss: str = Field(default="cross_entropy")
    optimizer: str = Field(default="adamw")  # adamw | sgd
    scheduler: str = Field(default="cosine_warmup")  # cosine_warmup | linear | constant
    learning_rate: float = Field(default=3e-3, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    batch_size: int = Field(default=4, ge=1, le=256)
    grad_accumulation: int = Field(default=1, ge=1, le=64)
    epochs: int = Field(default=5, ge=1, le=500)
    dropout: float = Field(default=0.05, ge=0.0, lt=0.9)
    precision: str = Field(default="float32")
    tokenizer_type: str = Field(default="bpe")  # bpe | char
    languages: List[str] = Field(default_factory=lambda: ["ru", "tg", "en"])
    custom_blocks: List[str] = Field(default_factory=lambda: ["ssm", "attention"])
    seed: int = Field(default=42)


def estimate_training_resources(spec: ModelSpec, platform_info: PlatformInfo) -> Dict[str, Any]:
    """
    Specification Section 4:
    Preliminary feasibility & resource estimate; explicitly marked as approximate (`is_approximate=True`).
    Never promise training large LLMs on a smartphone / 3 GB RAM device.
    """
    probe_model = build_model_from_spec(spec.model_dump(), seed=spec.seed)
    params = probe_model.parameter_count()
    bytes_per_param = 4 if spec.precision == "float32" else 2
    # Weights + grads + Adam moments (m, v) ~ 4x parameter bytes + activation workspace
    weights_grads_opt_mb = (params * bytes_per_param * 4) / (1024 * 1024)
    activations_mb = (
        spec.batch_size * spec.context_length * spec.d_model * spec.n_layers * 16
    ) / (1024 * 1024)
    overhead_mb = 65.0
    approx_peak_ram_mb = round(weights_grads_opt_mb + activations_mb + overhead_mb, 2)
    approx_sec_per_epoch = round(
        max(0.15, (params * spec.context_length) / max(1.0, platform_info.cpu_count * 4.5e7)),
        2,
    )

    warnings: List[str] = []
    feasible_locally = True
    if params > platform_info.recommended_max_params:
        warnings.append(
            f"Parameter count ({params:,}) exceeds recommended local limit "
            f"({platform_info.recommended_max_params:,}) for this device."
        )
        if platform_info.low_memory_mode:
            feasible_locally = False
            warnings.append(
                "Low Memory Mode is active: large model training on a low-RAM/mobile device is blocked; "
                "reduce model dimensions or explicitly select Remote/Hybrid Mode."
            )
    if approx_peak_ram_mb > platform_info.available_ram_mb * 0.75:
        feasible_locally = False
        warnings.append(
            f"Estimated peak RAM (~{approx_peak_ram_mb} MB) exceeds 75% of available RAM "
            f"({platform_info.available_ram_mb} MB)."
        )

    return {
        "is_approximate": True,
        "disclaimer": "Approximate heuristic estimate based on parameter tensor sizes and sequence dimensions.",
        "parameter_count": params,
        "approx_peak_ram_mb": approx_peak_ram_mb,
        "approx_sec_per_epoch": approx_sec_per_epoch,
        "low_memory_mode": platform_info.low_memory_mode,
        "feasible_locally": feasible_locally,
        "warnings": warnings,
    }


class ModelBuilder:
    """Analyzes natural-language prompts, creates verified ModelSpecs, tokenizers, and workspace code."""

    def __init__(
        self,
        db: StorageDatabase,
        artifacts: ArtifactStore,
        events: EventBus,
        platform_info: PlatformInfo,
    ) -> None:
        self._db = db
        self._artifacts = artifacts
        self._events = events
        self._platform = platform_info

    def spec_from_prompt(self, prompt: str, name: Optional[str] = None) -> ModelSpec:
        p = prompt.lower()
        langs: List[str] = []
        if any(w in p for w in ("русс", "russian", "кирилл", "ru")):
            langs.append("ru")
        if any(w in p for w in ("таджик", "тоҷик", "tajik", "tg")):
            langs.append("tg")
        if any(w in p for w in ("англ", "english", "en")):
            langs.append("en")
        if not langs:
            langs = ["ru", "tg", "en"]

        # Choose architecture from prompt intent
        arch = "decoder_transformer"
        if any(w in p for w in ("эксперимент", "гибрид", "experimental", "hybrid")):
            arch = "hybrid_attention_ssm"
        elif any(w in p for w in ("ssm", "mamba", "state space", "пространств")):
            arch = "ssm"
        elif "lstm" in p:
            arch = "lstm"
        elif "gru" in p:
            arch = "gru"
        elif "rnn" in p:
            arch = "rnn"
        elif any(w in p for w in ("encoder-decoder", "seq2seq", "перевод")):
            arch = "encoder_decoder_transformer"
        elif "encoder" in p or "энкодер" in p:
            arch = "encoder_transformer"
        elif "custom" in p or "пользовательск" in p:
            arch = "custom_block"

        tok_type = "char" if any(w in p for w in ("char", "символьн", "посимвольн")) else "bpe"

        if self._platform.low_memory_mode or any(w in p for w in ("компакт", "tiny", "маленьк", "small", "low ram")):
            d_model = 48
            n_layers = 2
            n_heads = 4
            ctx_len = 96
            batch_size = 4
            vocab_size = 320
        else:
            d_model = 64
            n_layers = 3
            n_heads = 4
            ctx_len = 128
            batch_size = 8
            vocab_size = 384

        derived_name = name or f"{arch.replace('_', '-')}-{'-'.join(langs)}"
        return ModelSpec(
            name=derived_name,
            architecture=arch,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            vocab_size=vocab_size,
            context_length=ctx_len,
            batch_size=batch_size,
            tokenizer_type=tok_type,
            languages=langs,
        )

    def create_model(
        self,
        project_id: str,
        spec: ModelSpec,
        corpus_texts: Optional[List[str]] = None,
        provenance: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if spec.architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(f"Unsupported architecture '{spec.architecture}'.")

        seed_texts = list(corpus_texts or DEFAULT_MULTILINGUAL_SEED_CORPUS)
        tokenizer = train_tokenizer(
            seed_texts,
            tokenizer_type=spec.tokenizer_type,
            vocab_size=spec.vocab_size,
        )
        unicode_check = verify_multilingual_unicode(tokenizer)
        actual_spec = spec.model_copy(update={"vocab_size": tokenizer.vocab_size})

        estimate = estimate_training_resources(actual_spec, self._platform)
        model_id = f"mdl-{secrets.token_hex(5)}"
        ws = self._artifacts.ensure_project_workspace(project_id)
        model_dir = ws / "models" / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        tok_path = model_dir / "tokenizer.json"
        tokenizer.save(tok_path)

        spec_dict = actual_spec.model_dump()
        (model_dir / "spec.json").write_text(
            json.dumps(spec_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # Generate runnable model entrypoint & unit test inside the project workspace
        (model_dir / "run_model.py").write_text(
            f'"""Auto-generated runnable entrypoint for {actual_spec.name} ({model_id})."""\n\n'
            "import json\nfrom pathlib import Path\n"
            "from modelforge.architectures.models import build_model_from_spec\n"
            "from modelforge.tokenizers.tokenizer import load_tokenizer\n\n"
            "HERE = Path(__file__).resolve().parent\n\n"
            "def load():\n"
            '    spec = json.loads((HERE / "spec.json").read_text(encoding="utf-8"))\n'
            '    tok = load_tokenizer(HERE / "tokenizer.json")\n'
            "    mdl = build_model_from_spec(spec, seed=spec.get('seed', 42))\n"
            "    return mdl, tok, spec\n",
            encoding="utf-8",
        )

        # Save initial untrained v1 checkpoint so every version is immediately inspectable/restorable
        from modelforge.training.checkpoints import save_checkpoint

        init_model = build_model_from_spec(spec_dict, seed=actual_spec.seed)
        ckpt_dir = self._artifacts.checkpoints_dir / model_id / "v1_init"
        ckpt_info = save_checkpoint(
            checkpoint_dir=ckpt_dir,
            model=init_model,
            spec=spec_dict,
            tokenizer_path=tok_path,
            step=0,
            epoch=0,
            metrics={"stage": "initialized"},
        )

        now = time.time()
        prov = {
            "created_from": provenance or {"source": "model_builder"},
            "unicode_verification": unicode_check,
            "resource_estimate": estimate,
            "runtime_requirements": {
                "python": ">=3.10",
                "engine": "modelforge_numpy_autograd",
                "optional_accelerators": ["pytorch", "llama.cpp_gguf"],
            },
        }
        self._db.upsert_row(
            "models",
            {
                "id": model_id,
                "project_id": project_id,
                "name": actual_spec.name,
                "architecture": actual_spec.architecture,
                "version": 1,
                "status": "initialized",
                "spec_json": json.dumps(spec_dict, ensure_ascii=False),
                "tokenizer_path": str(tok_path),
                "latest_checkpoint": ckpt_info["checkpoint_path"],
                "metrics_json": json.dumps({"parameter_count": estimate["parameter_count"]}, ensure_ascii=False),
                "provenance_json": json.dumps(prov, ensure_ascii=False),
                "tags_json": json.dumps(tags or actual_spec.languages, ensure_ascii=False),
                "created_at": now,
                "updated_at": now,
            },
        )
        self._db.upsert_row(
            "model_versions",
            {
                "id": f"{model_id}-v1",
                "model_id": model_id,
                "version": 1,
                "spec_json": json.dumps(spec_dict, ensure_ascii=False),
                "checkpoint_path": ckpt_info["checkpoint_path"],
                "checkpoint_sha256": ckpt_info["sha256"],
                "metrics_json": json.dumps({"stage": "initialized"}, ensure_ascii=False),
                "notes": "Initial model & tokenizer creation",
                "created_at": now,
            },
        )

        result = self.get_model(model_id)
        self._events.publish("model.created", result, project_id=project_id)
        return result

    def create_from_prompt(
        self,
        project_id: str,
        prompt: str,
        name: Optional[str] = None,
        corpus_texts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        spec = self.spec_from_prompt(prompt, name=name)
        return self.create_model(
            project_id=project_id,
            spec=spec,
            corpus_texts=corpus_texts,
            provenance={"source": "prompt", "prompt": prompt},
        )

    def get_model(self, model_id: str) -> Dict[str, Any]:
        row = self._db.query_one("SELECT * FROM models WHERE id = ?", (model_id,))
        if not row:
            raise KeyError(f"Model not found: {model_id}")
        versions = self._db.query_all(
            "SELECT * FROM model_versions WHERE model_id = ? ORDER BY version ASC",
            (model_id,),
        )
        return {
            "id": row["id"],
            "project_id": row["project_id"],
            "name": row["name"],
            "architecture": row["architecture"],
            "version": int(row["version"]),
            "status": row["status"],
            "spec": json.loads(row["spec_json"] or "{}"),
            "tokenizer_path": row["tokenizer_path"],
            "latest_checkpoint": row["latest_checkpoint"],
            "metrics": json.loads(row["metrics_json"] or "{}"),
            "provenance": json.loads(row["provenance_json"] or "{}"),
            "tags": json.loads(row["tags_json"] or "[]"),
            "versions": [
                {
                    "version": int(v["version"]),
                    "checkpoint_path": v["checkpoint_path"],
                    "checkpoint_sha256": v["checkpoint_sha256"],
                    "metrics": json.loads(v["metrics_json"] or "{}"),
                    "notes": v["notes"],
                    "created_at": v["created_at"],
                }
                for v in versions
            ],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_models(self, project_id: str = "") -> List[Dict[str, Any]]:
        if project_id:
            rows = self._db.query_all(
                "SELECT id FROM models WHERE project_id = ? ORDER BY updated_at DESC",
                (project_id,),
            )
        else:
            rows = self._db.query_all("SELECT id FROM models ORDER BY updated_at DESC")
        return [self.get_model(r["id"]) for r in rows]

    def restore_version(self, model_id: str, version: int) -> Dict[str, Any]:
        vrow = self._db.query_one(
            "SELECT * FROM model_versions WHERE model_id = ? AND version = ?",
            (model_id, version),
        )
        if not vrow:
            raise KeyError(f"Version {version} not found for model {model_id}")
        self._db.execute(
            "UPDATE models SET version = ?, spec_json = ?, latest_checkpoint = ?, metrics_json = ?, "
            "status = 'restored', updated_at = ? WHERE id = ?",
            (
                version,
                vrow["spec_json"],
                vrow["checkpoint_path"],
                vrow["metrics_json"],
                time.time(),
                model_id,
            ),
        )
        return self.get_model(model_id)
