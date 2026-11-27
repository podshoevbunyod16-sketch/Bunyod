"""Lazy-loaded optional PyTorch runtime adapter with ABI & platform safety check."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def is_pytorch_available() -> bool:
    try:
        import torch  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def export_weights_to_pytorch(weights: Dict[str, Any], spec: Dict[str, Any], target_path: Path) -> Dict[str, Any]:
    """Export trained ModelForge weights to a real `.pt` file when PyTorch is installed."""
    if not is_pytorch_available():
        raise RuntimeError(
            "PyTorch (`torch`) is not installed in this environment. "
            "Use `.mfckpt`, `.safetensors`, or `.gguf` export, or install `pip install modelforge-ai[torch]`."
        )
    import torch  # type: ignore

    state_dict = {k: torch.from_numpy(v.copy()) for k, v in weights.items()}
    target_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"spec": spec, "state_dict": state_dict}, str(target_path))
    return {"path": str(target_path), "tensors": len(state_dict), "format": "pytorch_pt"}
