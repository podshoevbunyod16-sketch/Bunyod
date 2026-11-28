"""Tests for Section 18 extensibility contracts, INT8 quantization, magnitude pruning & self-improvement gates."""

from __future__ import annotations

import numpy as np
import pytest

from modelforge.core.extensibility import (
    AgentEvolutionCandidate,
    AutoRnDCycleSpec,
    DistillationConfig,
    SelfImprovementProposal,
    WorkerNodeHeartbeat,
    WorkspaceMemberRole,
    dequantize_state_dict_int8,
    prune_state_dict_by_magnitude,
    quantize_state_dict_int8,
)


def test_int8_quantization_and_magnitude_pruning() -> None:
    rng = np.random.default_rng(42)
    weights = {
        "layer0.wq": rng.normal(0.0, 0.5, size=(32, 32)).astype(np.float32),
        "layer0.bias": np.zeros((32,), dtype=np.float32),
    }
    q_weights, scales = quantize_state_dict_int8(weights)
    assert q_weights["layer0.wq"].dtype == np.int8
    restored = dequantize_state_dict_int8(q_weights, scales)
    max_err = float(np.max(np.abs(restored["layer0.wq"] - weights["layer0.wq"])))
    assert max_err < 0.02

    pruned, stats = prune_state_dict_by_magnitude(weights, sparsity=0.25)
    assert stats["layer0.wq"] >= 0.24
    assert pruned["layer0.wq"].shape == (32, 32)
    with pytest.raises(ValueError):
        prune_state_dict_by_magnitude(weights, sparsity=1.0)


def test_self_improvement_branch_and_approval_gates() -> None:
    blocked_main = SelfImprovementProposal(
        proposal_id="sip-1",
        target_branch="main",
        summary="Attempt direct edit on main",
        tests_passed=True,
        user_approved=True,
    )
    ok, reason = blocked_main.validate_merge_gate()
    assert not ok and "prohibited" in reason

    untested = SelfImprovementProposal(
        proposal_id="sip-2",
        target_branch="self-improve/opt-attention",
        summary="Optimize causal mask cache",
        tests_passed=False,
        user_approved=True,
    )
    assert not untested.validate_merge_gate()[0]

    unapproved = SelfImprovementProposal(
        proposal_id="sip-3",
        target_branch="self-improve/opt-attention",
        summary="Optimize causal mask cache",
        tests_passed=True,
        user_approved=False,
    )
    assert not unapproved.validate_merge_gate()[0]

    valid = SelfImprovementProposal(
        proposal_id="sip-4",
        target_branch="self-improve/opt-attention",
        summary="Optimize causal mask cache",
        tests_passed=True,
        user_approved=True,
    )
    assert valid.validate_merge_gate()[0] is True

    rnd = AutoRnDCycleSpec(cycle_id="rnd-1", research_question="Compare GRU vs SSM on Tajik")
    cand = AgentEvolutionCandidate(candidate_id="ev-1", parent_agent_id="agt-1", system_prompt_variant="v2")
    hb = WorkerNodeHeartbeat(worker_id="wrk-1", hostname="node-a", arch="x86_64", ram_free_mb=8192)
    distill = DistillationConfig(teacher_model_id="mdl-t", student_model_id="mdl-s")
    role_viewer = WorkspaceMemberRole(user_id="u1", project_id="p1", role="viewer")
    role_admin = WorkspaceMemberRole(user_id="u2", project_id="p1", role="admin")

    assert rnd.preserve_negative_results is True
    assert cand.to_dict()["candidate_id"] == "ev-1"
    assert hb.to_dict()["ram_free_mb"] == 8192
    assert distill.to_dict()["temperature"] == 2.0
    assert not role_viewer.can_approve_destructive()
    assert role_admin.can_approve_destructive()
