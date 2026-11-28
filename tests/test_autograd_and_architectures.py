"""Unit tests for reverse-mode autograd numerical gradcheck & all 10 neural architectures."""

from __future__ import annotations

import numpy as np
import pytest

from modelforge.architectures.models import SUPPORTED_ARCHITECTURES, build_model_from_spec
from modelforge.architectures.tensor_engine import (
    Tensor,
    embedding_lookup,
    layer_norm,
    rms_norm,
    sequence_cross_entropy,
)


def _numerical_grad(fn, arr: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    grad = np.zeros_like(arr, dtype=np.float32)
    it = np.nditer(arr, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = float(arr[idx])
        arr[idx] = orig + eps
        loss_pos = float(fn())
        arr[idx] = orig - eps
        loss_neg = float(fn())
        arr[idx] = orig
        grad[idx] = (loss_pos - loss_neg) / (2.0 * eps)
        it.iternext()
    return grad


def test_autograd_finite_difference_gradcheck():
    rng = np.random.default_rng(123)
    w_emb = Tensor(rng.normal(0, 0.3, (12, 6)).astype(np.float32), requires_grad=True)
    w_proj = Tensor(rng.normal(0, 0.3, (6, 12)).astype(np.float32), requires_grad=True)
    gamma = Tensor(np.ones((6,), dtype=np.float32), requires_grad=True)
    beta = Tensor(np.zeros((6,), dtype=np.float32), requires_grad=True)

    ids = np.array([[2, 4, 7], [3, 5, 9]], dtype=np.int64)
    targets = np.array([[4, 7, 3], [5, 9, 0]], dtype=np.int64)

    def _forward_loss() -> Tensor:
        emb = embedding_lookup(w_emb, ids)
        normed = layer_norm(emb, gamma, beta)
        act = normed.gelu() + rms_norm(emb, gamma).silu()
        logits = act @ w_proj
        return sequence_cross_entropy(logits, targets, ignore_index=0)

    loss = _forward_loss()
    loss.backward()

    for param in (w_emb, w_proj, gamma, beta):
        analytic = param.grad.copy()
        numeric = _numerical_grad(lambda: _forward_loss().data, param.data)
        np.testing.assert_allclose(analytic, numeric, rtol=2e-2, atol=2e-3)


@pytest.mark.parametrize("arch", SUPPORTED_ARCHITECTURES)
def test_all_supported_architectures_forward_and_backward(arch: str):
    spec = {
        "name": f"test-{arch}",
        "architecture": arch,
        "vocab_size": 48,
        "d_model": 32,
        "n_layers": 2,
        "n_heads": 4,
        "context_length": 24,
        "dropout": 0.0,
        "positional_encoding": "rope",
        "custom_blocks": ["ssm", "attention"],
    }
    model = build_model_from_spec(spec, seed=7)
    assert model.parameter_count() > 0

    x = np.array([[2, 5, 8, 12, 19], [2, 10, 14, 3, 0]], dtype=np.int64)
    y = np.array([[5, 8, 12, 19, 3], [10, 14, 3, 0, 0]], dtype=np.int64)

    loss = model.compute_loss(x, y, training=True)
    assert np.isfinite(loss.data) and float(loss.data) > 0.0
    loss.backward()

    grad_norms = [float(np.linalg.norm(p.grad)) for p in model.trainable_parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert any(gn > 0.0 for gn in grad_norms)


def test_loop_transformer_reuses_parameters_and_trains():
    spec = {"architecture": "tiny_transformer", "vocab_size": 16, "d_model": 16,
            "n_layers": 1, "n_heads": 4, "context_length": 16, "dropout": 0.0}
    single = build_model_from_spec(spec, seed=123)
    loop = build_model_from_spec({**spec, "loop_passes": 3}, seed=123)
    assert single.parameter_count() == loop.parameter_count()
    ids = np.array([[1, 2, 3, 4]], dtype=np.int64)
    targets = np.array([[2, 3, 4, 5]], dtype=np.int64)
    assert not np.allclose(single.forward(ids).data, loop.forward(ids).data)
    loss = loop.compute_loss(ids, targets)
    loss.backward()
    param = loop.params["blocks.0.wq"]
    assert param.grad is not None and np.isfinite(param.grad).all()
    numeric = _numerical_grad(lambda: loop.compute_loss(ids, targets, training=False).data, param.data)
    np.testing.assert_allclose(param.grad, numeric, rtol=4e-2, atol=2e-3)


@pytest.mark.parametrize("passes", [0, 9, -1, 1.5, True])
def test_loop_passes_rejects_invalid_values(passes):
    with pytest.raises(ValueError):
        build_model_from_spec({"loop_passes": passes})


def test_loop_rejects_non_transformer_and_spec_roundtrip():
    from modelforge.model_builder.builder import ModelSpec
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ModelSpec(name="test", architecture="rnn", loop_passes=3)
    spec = ModelSpec(name="test", architecture="tiny_transformer", loop_passes=3)
    assert ModelSpec.model_validate_json(spec.model_dump_json()).loop_passes == 3
    assert ModelSpec(name="legacy").loop_passes == 1


def test_autograd_handles_deep_graph_without_recursion():
    x = Tensor(np.array(1.0, dtype=np.float32), requires_grad=True)
    y = x
    for _ in range(2000):
        y = y + 1.0
    y.backward()
    assert float(x.grad) == 1.0
