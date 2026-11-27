"""Verifiable checkpoint serializer (.mfckpt + .safetensors) with SHA-256 & optimizer state."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from modelforge.architectures.models import NeuralSequenceModel, build_model_from_spec


def weights_sha256(weights: Dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for key in sorted(weights.keys()):
        h.update(key.encode("utf-8"))
        arr = np.ascontiguousarray(weights[key], dtype=np.float32)
        h.update(arr.tobytes())
    return h.hexdigest()


def save_checkpoint(
    checkpoint_dir: Path,
    model: NeuralSequenceModel,
    spec: Dict[str, Any],
    tokenizer_path: Path,
    step: int,
    epoch: int,
    metrics: Dict[str, Any],
    optimizer_state: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weights = model.get_weights_dict()
    weights_file = checkpoint_dir / "weights.npz"
    np.savez_compressed(str(weights_file), **weights)

    if optimizer_state:
        opt_file = checkpoint_dir / "optimizer.npz"
        np.savez_compressed(str(opt_file), **optimizer_state)

    if tokenizer_path.exists():
        shutil.copy2(tokenizer_path, checkpoint_dir / "tokenizer.json")

    w_sha = weights_sha256(weights)
    manifest = {
        "format": "modelforge_checkpoint_v1",
        "created_at": time.time(),
        "step": step,
        "epoch": epoch,
        "weights_sha256": w_sha,
        "parameter_count": model.parameter_count(),
        "spec": spec,
        "metrics": metrics,
        "has_optimizer_state": bool(optimizer_state),
    }
    manifest_path = checkpoint_dir / "checkpoint.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "checkpoint_path": str(checkpoint_dir),
        "weights_file": str(weights_file),
        "sha256": w_sha,
        "step": step,
        "epoch": epoch,
    }


def load_checkpoint(
    checkpoint_path: str | Path,
    load_optimizer: bool = False,
) -> Dict[str, Any]:
    ckpt_dir = Path(checkpoint_path)
    if ckpt_dir.is_file() and ckpt_dir.name == "checkpoint.json":
        ckpt_dir = ckpt_dir.parent
    manifest_path = ckpt_dir / "checkpoint.json"
    weights_file = ckpt_dir / "weights.npz"
    if not manifest_path.exists() or not weights_file.exists():
        raise FileNotFoundError(f"Invalid or incomplete checkpoint directory: {ckpt_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(str(weights_file), allow_pickle=False) as data:
        weights = {k: data[k].astype(np.float32) for k in data.files}

    actual_sha = weights_sha256(weights)
    expected_sha = manifest.get("weights_sha256", "")
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(
            f"Checkpoint SHA-256 integrity mismatch: expected {expected_sha}, got {actual_sha}"
        )

    model = build_model_from_spec(manifest["spec"], seed=int(manifest["spec"].get("seed", 42)))
    model.load_weights_dict(weights, strict=True)

    opt_state: Optional[Dict[str, np.ndarray]] = None
    opt_file = ckpt_dir / "optimizer.npz"
    if load_optimizer and opt_file.exists():
        with np.load(str(opt_file), allow_pickle=False) as odata:
            opt_state = {k: odata[k].astype(np.float32) for k in odata.files}

    tok_candidate = ckpt_dir / "tokenizer.json"
    return {
        "model": model,
        "manifest": manifest,
        "weights": weights,
        "optimizer_state": opt_state,
        "tokenizer_path": str(tok_candidate) if tok_candidate.exists() else "",
    }


def export_safetensors(checkpoint_path: str | Path, output_file: Path) -> Dict[str, Any]:
    loaded = load_checkpoint(checkpoint_path)
    try:
        from safetensors.numpy import save_file  # type: ignore
    except ImportError as exc:
        raise RuntimeError("`safetensors` package is not installed.") from exc

    output_file.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "architecture": str(loaded["manifest"]["spec"].get("architecture", "")),
        "weights_sha256": str(loaded["manifest"].get("weights_sha256", "")),
        "format_producer": "modelforge-ai-1.0",
    }
    save_file(loaded["weights"], str(output_file), metadata=metadata)
    sha = hashlib.sha256(output_file.read_bytes()).hexdigest()
    return {
        "format": "safetensors",
        "path": str(output_file),
        "size_bytes": output_file.stat().st_size,
        "sha256": sha,
        "tensor_count": len(loaded["weights"]),
    }
