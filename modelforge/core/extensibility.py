"""Section 18 Future Extensibility Contracts, Post-Training Quantization & Pruning Hooks.

Provides clean, typed contracts and working reference algorithms for:
1. Automated AI R&D Lab cycles (`AutoRnDCycleSpec`)
2. Agent Evolution fitness & mutation contracts (`AgentEvolutionCandidate`)
3. Distributed Worker Nodes (`WorkerNodeHeartbeat`, `DistributedTaskLease`)
4. Multimodal Model Adapters (`MultimodalModalityAdapter`)
5. Post-Training Scalar INT8 Quantization, Magnitude Pruning & Distillation Specs
6. Team Collaboration RBAC (`WorkspaceMemberRole`)
7. Branch-Gated Self-Improvement (`SelfImprovementProposal` — strictly forbids direct edits to `main`)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Protocol, Tuple
import numpy as np


@dataclass
class AutoRnDCycleSpec:
    cycle_id: str
    research_question: str
    max_hypotheses: int = 3
    budget_seconds: float = 300.0
    preserve_negative_results: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentEvolutionCandidate:
    candidate_id: str
    parent_agent_id: str
    system_prompt_variant: str
    temperature: float = 0.2
    benchmark_score: float = 0.0
    evaluated_tasks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkerNodeHeartbeat:
    worker_id: str
    hostname: str
    arch: str
    ram_free_mb: int
    accelerators: List[str] = field(default_factory=list)
    status: str = "idle"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MultimodalModalityAdapter(Protocol):
    """Protocol for vision/audio/tabular encoders projecting into `d_model` token embeddings."""

    modality: str

    def encode_to_embeddings(self, raw_batch: Any, d_model: int) -> np.ndarray:
        ...


@dataclass
class DistillationConfig:
    teacher_model_id: str
    student_model_id: str
    temperature: float = 2.0
    alpha_kl: float = 0.5
    alpha_ce: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkspaceMemberRole:
    user_id: str
    project_id: str
    role: str = "researcher"  # viewer | researcher | maintainer | admin

    def can_approve_destructive(self) -> bool:
        return self.role in ("maintainer", "admin")


@dataclass
class SelfImprovementProposal:
    """Section 18: Platform self-improvement MUST occur on an isolated branch, pass tests, and receive user approval."""

    proposal_id: str
    target_branch: str
    summary: str
    tests_passed: bool = False
    user_approved: bool = False

    def validate_merge_gate(self) -> Tuple[bool, str]:
        if self.target_branch in ("main", "master"):
            return False, "Direct self-modification on 'main'/'master' is strictly prohibited."
        if not self.tests_passed:
            return False, "Automated test suite has not passed on the isolated proposal branch."
        if not self.user_approved:
            return False, "Explicit human approval is required before applying self-improvement changes."
        return True, "Approved for controlled merge."


def quantize_state_dict_int8(
    tensors: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Symmetric per-tensor INT8 post-training quantization with scale factors."""
    q_tensors: Dict[str, np.ndarray] = {}
    scales: Dict[str, float] = {}
    for name, arr in tensors.items():
        f32 = np.asarray(arr, dtype=np.float32)
        max_abs = float(np.max(np.abs(f32))) if f32.size > 0 else 0.0
        scale = max_abs / 127.0 if max_abs > 1e-12 else 1.0
        q = np.clip(np.round(f32 / scale), -127, 127).astype(np.int8)
        q_tensors[name] = q
        scales[name] = scale
    return q_tensors, scales


def dequantize_state_dict_int8(
    q_tensors: Dict[str, np.ndarray],
    scales: Dict[str, float],
) -> Dict[str, np.ndarray]:
    """Reconstruct FP32 weights from symmetric INT8 tensors + scales."""
    return {
        name: q.astype(np.float32) * float(scales.get(name, 1.0))
        for name, q in q_tensors.items()
    }


def prune_state_dict_by_magnitude(
    tensors: Dict[str, np.ndarray],
    sparsity: float = 0.2,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Unstructured magnitude pruning zeroing the smallest `sparsity` fraction of weights."""
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must be in [0.0, 1.0)")
    pruned: Dict[str, np.ndarray] = {}
    actual_sparsity: Dict[str, float] = {}
    for name, arr in tensors.items():
        out = np.asarray(arr, dtype=np.float32).copy()
        if out.ndim >= 2 and sparsity > 0.0 and out.size > 0:
            thresh = float(np.quantile(np.abs(out), sparsity))
            mask = np.abs(out) >= thresh
            out *= mask
        zeros = float(np.sum(out == 0.0)) / max(1, out.size)
        pruned[name] = out
        actual_sparsity[name] = round(zeros, 4)
    return pruned, actual_sparsity
