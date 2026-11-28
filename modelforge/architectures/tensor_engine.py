"""Exact reverse-mode automatic differentiation tensor engine over NumPy."""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Set, Tuple

import numpy as np


def _unbroadcast(grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """Sum out broadcasted dimensions so gradient matches `target_shape` exactly."""
    if grad.shape == target_shape:
        return grad
    g = grad
    while g.ndim > len(target_shape):
        g = g.sum(axis=0)
    for axis, (g_dim, t_dim) in enumerate(zip(g.shape, target_shape)):
        if t_dim == 1 and g_dim != 1:
            g = g.sum(axis=axis, keepdims=True)
    return g.reshape(target_shape)


class Tensor:
    """Differentiable N-dimensional array with reverse-mode backpropagation."""

    def __init__(
        self,
        data: np.ndarray | float | list,
        requires_grad: bool = False,
        _children: Tuple["Tensor", ...] = (),
        _op: str = "",
    ) -> None:
        arr = data if isinstance(data, np.ndarray) else np.asarray(data, dtype=np.float32)
        if arr.dtype not in (np.float32, np.float64):
            arr = arr.astype(np.float32)
        self.data: np.ndarray = arr
        self.requires_grad: bool = requires_grad
        self.grad: Optional[np.ndarray] = np.zeros_like(self.data) if requires_grad else None
        self._prev: Tuple["Tensor", ...] = _children
        self._op: str = _op
        self._backward: Callable[[], None] = lambda: None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def zero_grad(self) -> None:
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def _acc_grad(self, g: np.ndarray) -> None:
        if self.requires_grad:
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            self.grad = self.grad + _unbroadcast(g.astype(self.data.dtype, copy=False), self.data.shape)

    def backward(self) -> None:
        topo: List[Tensor] = []
        visited: Set[int] = set()

        # Iterative post-order traversal avoids Python's recursion limit for
        # deep graphs, including multiple passes through shared blocks.
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for child in reversed(node._prev):
                if child.requires_grad and id(child) not in visited:
                    stack.append((child, False))
        self.grad = np.ones_like(self.data, dtype=self.data.dtype)
        for node in reversed(topo):
            node._backward()

    # --- Basic arithmetic ---
    def __add__(self, other: "Tensor | float | np.ndarray") -> "Tensor":
        oth = other if isinstance(other, Tensor) else Tensor(other)
        req = self.requires_grad or oth.requires_grad
        out = Tensor(self.data + oth.data, requires_grad=req, _children=(self, oth) if req else (), _op="+")
        if req:

            def _bw() -> None:
                if out.grad is None:
                    return
                if self.requires_grad:
                    self._acc_grad(out.grad)
                if oth.requires_grad:
                    oth._acc_grad(out.grad)

            out._backward = _bw
        return out

    def __radd__(self, other: float | np.ndarray) -> "Tensor":
        return self.__add__(other)

    def __neg__(self) -> "Tensor":
        req = self.requires_grad
        out = Tensor(-self.data, requires_grad=req, _children=(self,) if req else (), _op="neg")
        if req:

            def _bw() -> None:
                if out.grad is not None and self.requires_grad:
                    self._acc_grad(-out.grad)

            out._backward = _bw
        return out

    def __sub__(self, other: "Tensor | float | np.ndarray") -> "Tensor":
        oth = other if isinstance(other, Tensor) else Tensor(other)
        req = self.requires_grad or oth.requires_grad
        out = Tensor(self.data - oth.data, requires_grad=req, _children=(self, oth) if req else (), _op="-")
        if req:

            def _bw() -> None:
                if out.grad is None:
                    return
                if self.requires_grad:
                    self._acc_grad(out.grad)
                if oth.requires_grad:
                    oth._acc_grad(-out.grad)

            out._backward = _bw
        return out

    def __mul__(self, other: "Tensor | float | np.ndarray") -> "Tensor":
        oth = other if isinstance(other, Tensor) else Tensor(other)
        req = self.requires_grad or oth.requires_grad
        out = Tensor(self.data * oth.data, requires_grad=req, _children=(self, oth) if req else (), _op="*")
        if req:

            def _bw() -> None:
                if out.grad is None:
                    return
                if self.requires_grad:
                    self._acc_grad(out.grad * oth.data)
                if oth.requires_grad:
                    oth._acc_grad(out.grad * self.data)

            out._backward = _bw
        return out

    def __rmul__(self, other: float | np.ndarray) -> "Tensor":
        return self.__mul__(other)

    def matmul(self, other: "Tensor") -> "Tensor":
        req = self.requires_grad or other.requires_grad
        out_data = np.matmul(self.data, other.data)
        out = Tensor(out_data, requires_grad=req, _children=(self, other) if req else (), _op="@")
        if req:

            def _bw() -> None:
                if out.grad is None:
                    return
                g = out.grad
                if self.requires_grad:
                    b_T = np.swapaxes(other.data, -1, -2)
                    self._acc_grad(np.matmul(g, b_T))
                if other.requires_grad:
                    a_T = np.swapaxes(self.data, -1, -2)
                    other._acc_grad(np.matmul(a_T, g))

            out._backward = _bw
        return out

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return self.matmul(other)

    def reshape(self, *shape: int) -> "Tensor":
        orig_shape = self.data.shape
        req = self.requires_grad
        out = Tensor(self.data.reshape(*shape), requires_grad=req, _children=(self,) if req else (), _op="reshape")
        if req:

            def _bw() -> None:
                if out.grad is not None:
                    self._acc_grad(out.grad.reshape(orig_shape))

            out._backward = _bw
        return out

    def transpose(self, *axes: int) -> "Tensor":
        req = self.requires_grad
        out = Tensor(np.transpose(self.data, axes), requires_grad=req, _children=(self,) if req else (), _op="transpose")
        if req:
            inv_axes = tuple(np.argsort(axes))

            def _bw() -> None:
                if out.grad is not None:
                    self._acc_grad(np.transpose(out.grad, inv_axes))

            out._backward = _bw
        return out

    # --- Nonlinearities & Normalizations ---
    def relu(self) -> "Tensor":
        req = self.requires_grad
        mask = (self.data > 0).astype(self.data.dtype)
        out = Tensor(self.data * mask, requires_grad=req, _children=(self,) if req else (), _op="relu")
        if req:

            def _bw() -> None:
                if out.grad is not None:
                    self._acc_grad(out.grad * mask)

            out._backward = _bw
        return out

    def tanh(self) -> "Tensor":
        req = self.requires_grad
        t = np.tanh(self.data)
        out = Tensor(t, requires_grad=req, _children=(self,) if req else (), _op="tanh")
        if req:

            def _bw() -> None:
                if out.grad is not None:
                    self._acc_grad(out.grad * (1.0 - t * t))

            out._backward = _bw
        return out

    def sigmoid(self) -> "Tensor":
        req = self.requires_grad
        clipped = np.clip(self.data, -30.0, 30.0)
        s = 1.0 / (1.0 + np.exp(-clipped))
        out = Tensor(s, requires_grad=req, _children=(self,) if req else (), _op="sigmoid")
        if req:

            def _bw() -> None:
                if out.grad is not None:
                    self._acc_grad(out.grad * s * (1.0 - s))

            out._backward = _bw
        return out

    def silu(self) -> "Tensor":
        return self * self.sigmoid()

    def gelu(self) -> "Tensor":
        # Tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        req = self.requires_grad
        x = self.data
        c = np.float32(np.sqrt(2.0 / np.pi))
        inner = c * (x + np.float32(0.044715) * (x ** 3))
        th = np.tanh(inner)
        y = np.float32(0.5) * x * (np.float32(1.0) + th)
        out = Tensor(y, requires_grad=req, _children=(self,) if req else (), _op="gelu")
        if req:

            def _bw() -> None:
                if out.grad is not None:
                    sech2 = np.float32(1.0) - th * th
                    d_inner = c * (np.float32(1.0) + np.float32(3.0 * 0.044715) * x * x)
                    dy_dx = np.float32(0.5) * (np.float32(1.0) + th) + np.float32(0.5) * x * sech2 * d_inner
                    self._acc_grad(out.grad * dy_dx)

            out._backward = _bw
        return out

    def softmax(self, axis: int = -1) -> "Tensor":
        req = self.requires_grad
        shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
        exps = np.exp(shifted)
        probs = exps / np.sum(exps, axis=axis, keepdims=True)
        out = Tensor(probs, requires_grad=req, _children=(self,) if req else (), _op="softmax")
        if req:

            def _bw() -> None:
                if out.grad is not None:
                    dot = np.sum(out.grad * probs, axis=axis, keepdims=True)
                    self._acc_grad(probs * (out.grad - dot))

            out._backward = _bw
        return out

    def dropout(self, p: float, rng: np.random.Generator, training: bool = True) -> "Tensor":
        if not training or p <= 0.0:
            return self
        keep = 1.0 - p
        mask = (rng.random(self.data.shape) < keep).astype(self.data.dtype) / np.float32(keep)
        return self * mask


class Parameter(Tensor):
    """Named learnable tensor parameter."""

    def __init__(self, name: str, data: np.ndarray, trainable: bool = True) -> None:
        super().__init__(data.astype(np.float32), requires_grad=trainable)
        self.name = name
        self.trainable = trainable


def embedding_lookup(weight: Tensor, indices: np.ndarray) -> Tensor:
    """Differentiable token embedding lookup `weight[indices]` -> (*indices.shape, dim)."""
    req = weight.requires_grad
    idx = np.asarray(indices, dtype=np.int64)
    out_data = weight.data[idx]
    out = Tensor(out_data, requires_grad=req, _children=(weight,) if req else (), _op="embed")
    if req:

        def _bw() -> None:
            if out.grad is None or not weight.requires_grad:
                return
            grad_w = np.zeros_like(weight.data)
            np.add.at(grad_w, idx, out.grad)
            weight._acc_grad(grad_w)

        out._backward = _bw
    return out


def layer_norm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
    req = x.requires_grad or gamma.requires_grad or beta.requires_grad
    mean = np.mean(x.data, axis=-1, keepdims=True)
    var = np.var(x.data, axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + np.float32(eps))
    x_hat = (x.data - mean) * inv_std
    y = x_hat * gamma.data + beta.data
    children = tuple(t for t in (x, gamma, beta) if t.requires_grad)
    out = Tensor(y, requires_grad=req, _children=children, _op="layernorm")
    if req:
        d = np.float32(x.data.shape[-1])

        def _bw() -> None:
            if out.grad is None:
                return
            dy = out.grad
            if gamma.requires_grad:
                gamma._acc_grad(np.sum(dy * x_hat, axis=tuple(range(dy.ndim - 1))))
            if beta.requires_grad:
                beta._acc_grad(np.sum(dy, axis=tuple(range(dy.ndim - 1))))
            if x.requires_grad:
                dx_hat = dy * gamma.data
                dx = (inv_std / d) * (
                    d * dx_hat
                    - np.sum(dx_hat, axis=-1, keepdims=True)
                    - x_hat * np.sum(dx_hat * x_hat, axis=-1, keepdims=True)
                )
                x._acc_grad(dx)

        out._backward = _bw
    return out


def rms_norm(x: Tensor, gamma: Tensor, eps: float = 1e-5) -> Tensor:
    req = x.requires_grad or gamma.requires_grad
    ms = np.mean(x.data * x.data, axis=-1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(ms + np.float32(eps))
    x_hat = x.data * inv_rms
    y = x_hat * gamma.data
    children = tuple(t for t in (x, gamma) if t.requires_grad)
    out = Tensor(y, requires_grad=req, _children=children, _op="rmsnorm")
    if req:

        def _bw() -> None:
            if out.grad is None:
                return
            dy = out.grad
            if gamma.requires_grad:
                gamma._acc_grad(np.sum(dy * x_hat, axis=tuple(range(dy.ndim - 1))))
            if x.requires_grad:
                dx_hat = dy * gamma.data
                mean_term = np.mean(dx_hat * x_hat, axis=-1, keepdims=True)
                dx = inv_rms * (dx_hat - x_hat * mean_term)
                x._acc_grad(dx)

        out._backward = _bw
    return out


def stack_time(steps: Iterable[Tensor], axis: int = 1) -> Tensor:
    """Stack a list of per-timestep `(B, D)` tensors into `(B, T, D)`."""
    step_list = list(steps)
    req = any(s.requires_grad for s in step_list)
    data = np.stack([s.data for s in step_list], axis=axis)
    children = tuple(s for s in step_list if s.requires_grad)
    out = Tensor(data, requires_grad=req, _children=children, _op="stack")
    if req:

        def _bw() -> None:
            if out.grad is None:
                return
            grads = np.moveaxis(out.grad, axis, 0)
            for s, g in zip(step_list, grads):
                if s.requires_grad:
                    s._acc_grad(g)

        out._backward = _bw
    return out


def sequence_cross_entropy(logits: Tensor, targets: np.ndarray, ignore_index: int = 0) -> Tensor:
    """
    Numerically stable cross-entropy loss over `(B, T, V)` logits and `(B, T)` integer targets,
    ignoring `ignore_index` (pad_id=0).
    """
    b, t, v = logits.shape
    flat_logits = logits.data.reshape(-1, v)
    flat_targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    valid_mask = flat_targets != ignore_index
    valid_count = int(np.sum(valid_mask))
    if valid_count == 0:
        return Tensor(np.array(0.0, dtype=np.float32), requires_grad=logits.requires_grad)

    shifted = flat_logits - np.max(flat_logits, axis=-1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    log_probs = shifted - log_sum_exp

    safe_targets = np.where(valid_mask, flat_targets, 0)
    nll = -log_probs[np.arange(flat_logits.shape[0]), safe_targets]
    loss_val = np.sum(nll * valid_mask) / float(valid_count)

    req = logits.requires_grad
    out = Tensor(
        np.array(loss_val, dtype=np.float32),
        requires_grad=req,
        _children=(logits,) if req else (),
        _op="cross_entropy",
    )
    if req:
        probs = np.exp(log_probs)

        def _bw() -> None:
            if out.grad is None:
                return
            grad = probs.copy()
            grad[np.arange(flat_logits.shape[0]), safe_targets] -= 1.0
            grad[~valid_mask] = 0.0
            grad = (grad / float(valid_count)) * float(out.grad)
            logits._acc_grad(grad.reshape(b, t, v))

        out._backward = _bw
    return out
