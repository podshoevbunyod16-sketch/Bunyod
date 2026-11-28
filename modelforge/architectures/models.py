"""Configurable neural sequence architectures (GPT/Decoder, Encoder, Seq2Seq, RNN/LSTM/GRU, SSM, Hybrid)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from modelforge.architectures.tensor_engine import (
    Parameter,
    Tensor,
    embedding_lookup,
    layer_norm,
    rms_norm,
    sequence_cross_entropy,
    stack_time,
)

SUPPORTED_ARCHITECTURES = [
    "decoder_transformer",
    "tiny_transformer",
    "encoder_transformer",
    "encoder_decoder_transformer",
    "rnn",
    "lstm",
    "gru",
    "ssm",
    "hybrid_attention_ssm",
    "custom_block",
]


def _apply_activation(x: Tensor, kind: str) -> Tensor:
    k = kind.lower()
    if k == "relu":
        return x.relu()
    if k in ("silu", "swish"):
        return x.silu()
    if k == "tanh":
        return x.tanh()
    return x.gelu()


def _sinusoidal_positions(seq_len: int, d_model: int) -> np.ndarray:
    pos = np.arange(seq_len, dtype=np.float32)[:, None]
    div = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * (-np.log(10000.0) / max(1, d_model)))
    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


def _apply_rope_numpy(q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
    """Rotary positional embedding on `(B, H, T, Dh)` for even Dh."""
    b, h, t, dh = q.shape
    if dh % 2 != 0:
        return q, k
    half = dh // 2
    freqs = np.exp(-np.log(10000.0) * np.arange(0, half, dtype=np.float32) / max(1, half))
    angles = np.arange(t, dtype=np.float32)[:, None] * freqs[None, :]
    cos = np.cos(angles)[None, None, :, :]
    sin = np.sin(angles)[None, None, :, :]

    def _rot(x: Tensor) -> Tensor:
        req = x.requires_grad
        x1 = x.data[..., :half]
        x2 = x.data[..., half:]
        out_data = np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
        out = Tensor(out_data, requires_grad=req, _children=(x,) if req else (), _op="rope")
        if req:

            def _bw() -> None:
                if out.grad is None:
                    return
                g1 = out.grad[..., :half]
                g2 = out.grad[..., half:]
                dx1 = g1 * cos + g2 * sin
                dx2 = -g1 * sin + g2 * cos
                x._acc_grad(np.concatenate([dx1, dx2], axis=-1))

            out._backward = _bw
        return out

    return _rot(q), _rot(k)


class NeuralSequenceModel:
    """
    Unified runnable & trainable neural sequence model supporting all Section 3 architectures:
    - decoder_transformer / tiny_transformer (GPT-like causal LM)
    - encoder_transformer
    - encoder_decoder_transformer
    - rnn / lstm / gru
    - ssm (diagonal state space model)
    - hybrid_attention_ssm (alternating SSM + causal self-attention)
    - custom_block (user-defined block sequence)
    """

    def __init__(self, spec: Dict[str, Any], seed: int = 42) -> None:
        self.spec = dict(spec)
        self.architecture: str = str(spec.get("architecture", "decoder_transformer")).lower()
        self.loop_passes = spec.get("loop_passes", 1)
        if isinstance(self.loop_passes, bool) or not isinstance(self.loop_passes, int) or not 1 <= self.loop_passes <= 8:
            raise ValueError("loop_passes must be an integer between 1 and 8.")
        if self.loop_passes > 1 and self.architecture not in ("decoder_transformer", "tiny_transformer"):
            raise ValueError("loop_passes > 1 is supported only for decoder transformers.")
        if self.architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(f"Unsupported architecture '{self.architecture}'. Supported: {SUPPORTED_ARCHITECTURES}")

        self.vocab_size: int = int(spec.get("vocab_size", 384))
        self.d_model: int = int(spec.get("d_model", 64))
        self.n_layers: int = int(spec.get("n_layers", 2))
        self.n_heads: int = int(spec.get("n_heads", 4))
        if self.d_model % self.n_heads != 0:
            # Adjust heads cleanly to divide d_model
            for candidate in (4, 2, 1):
                if self.d_model % candidate == 0:
                    self.n_heads = candidate
                    break
        self.context_length: int = int(spec.get("context_length", 128))
        self.dropout: float = float(spec.get("dropout", 0.05))
        self.activation: str = str(spec.get("activation", "gelu"))
        self.norm_type: str = str(spec.get("norm_type", "rmsnorm")).lower()
        self.positional_encoding: str = str(spec.get("positional_encoding", "rope")).lower()
        self.custom_blocks: List[str] = list(spec.get("custom_blocks") or ["ssm", "attention"])

        self.rng = np.random.default_rng(seed)
        self.params: Dict[str, Parameter] = {}
        self._block_plan = self._resolve_block_plan()
        self._init_parameters()

    def _resolve_block_plan(self) -> List[str]:
        arch = self.architecture
        if arch in ("decoder_transformer", "tiny_transformer"):
            return ["causal_attn" for _ in range(self.n_layers)]
        if arch == "encoder_transformer":
            return ["bi_attn" for _ in range(self.n_layers)]
        if arch == "encoder_decoder_transformer":
            return ["seq2seq_block" for _ in range(self.n_layers)]
        if arch in ("rnn", "lstm", "gru", "ssm"):
            return [arch for _ in range(self.n_layers)]
        if arch == "hybrid_attention_ssm":
            return ["ssm" if i % 2 == 0 else "causal_attn" for i in range(self.n_layers)]
        if arch == "custom_block":
            normalized = []
            for b in self.custom_blocks:
                b_low = b.lower()
                if b_low in ("attention", "attn", "causal_attn"):
                    normalized.append("causal_attn")
                elif b_low in ("ssm", "mamba", "s4"):
                    normalized.append("ssm")
                elif b_low in ("gru", "lstm", "rnn"):
                    normalized.append(b_low)
                else:
                    normalized.append("mlp_only")
            if not normalized:
                normalized = ["causal_attn"]
            return [normalized[i % len(normalized)] for i in range(self.n_layers)]
        return ["causal_attn" for _ in range(self.n_layers)]

    def _add_param(self, name: str, shape: tuple[int, ...], scale: Optional[float] = None, init: str = "normal") -> Parameter:
        if init == "ones":
            arr = np.ones(shape, dtype=np.float32)
        elif init == "zeros":
            arr = np.zeros(shape, dtype=np.float32)
        else:
            std = scale if scale is not None else float(1.0 / np.sqrt(max(1, shape[0])))
            arr = self.rng.normal(0.0, std, size=shape).astype(np.float32)
        p = Parameter(name, arr, trainable=True)
        self.params[name] = p
        return p

    def _add_norm_params(self, prefix: str) -> None:
        self._add_param(f"{prefix}.gamma", (self.d_model,), init="ones")
        if self.norm_type != "rmsnorm":
            self._add_param(f"{prefix}.beta", (self.d_model,), init="zeros")

    def _init_parameters(self) -> None:
        d = self.d_model
        ff = d * 3
        self._add_param("tok_emb", (self.vocab_size, d), scale=0.02)
        if self.positional_encoding == "learned":
            self._add_param("pos_emb", (self.context_length, d), scale=0.02)

        for i, btype in enumerate(self._block_plan):
            pfx = f"blocks.{i}"
            self._add_norm_params(f"{pfx}.norm1")
            if btype in ("causal_attn", "bi_attn", "seq2seq_block"):
                self._add_param(f"{pfx}.wq", (d, d))
                self._add_param(f"{pfx}.wk", (d, d))
                self._add_param(f"{pfx}.wv", (d, d))
                self._add_param(f"{pfx}.wo", (d, d), scale=float(0.5 / np.sqrt(d * max(1, self.n_layers))))
                if btype == "seq2seq_block":
                    self._add_norm_params(f"{pfx}.enc_norm")
                    self._add_param(f"{pfx}.enc_wq", (d, d))
                    self._add_param(f"{pfx}.enc_wk", (d, d))
                    self._add_param(f"{pfx}.enc_wv", (d, d))
                    self._add_param(f"{pfx}.enc_wo", (d, d))
                    self._add_norm_params(f"{pfx}.cross_norm")
                    self._add_param(f"{pfx}.cwq", (d, d))
                    self._add_param(f"{pfx}.cwk", (d, d))
                    self._add_param(f"{pfx}.cwv", (d, d))
                    self._add_param(f"{pfx}.cwo", (d, d))
            elif btype == "rnn":
                self._add_param(f"{pfx}.wx", (d, d))
                self._add_param(f"{pfx}.wh", (d, d), scale=0.2 / np.sqrt(d))
                self._add_param(f"{pfx}.b", (d,), init="zeros")
            elif btype == "gru":
                self._add_param(f"{pfx}.wx", (d, 3 * d))
                self._add_param(f"{pfx}.wh", (d, 3 * d), scale=0.2 / np.sqrt(d))
                self._add_param(f"{pfx}.b", (3 * d,), init="zeros")
            elif btype == "lstm":
                self._add_param(f"{pfx}.wx", (d, 4 * d))
                self._add_param(f"{pfx}.wh", (d, 4 * d), scale=0.2 / np.sqrt(d))
                # Forget gate bias initialized slightly positive
                b_init = np.zeros((4 * d,), dtype=np.float32)
                b_init[d : 2 * d] = 1.0
                self.params[f"{pfx}.b"] = Parameter(f"{pfx}.b", b_init, trainable=True)
            elif btype == "ssm":
                # Diagonal State Space block: input projection + learnable diagonal decay + gate + output proj
                self._add_param(f"{pfx}.in_proj", (d, 2 * d))
                log_decay = np.linspace(-2.5, -0.4, d, dtype=np.float32)
                self.params[f"{pfx}.log_decay"] = Parameter(f"{pfx}.log_decay", log_decay, trainable=True)
                self._add_param(f"{pfx}.b_vec", (d,), init="ones")
                self._add_param(f"{pfx}.c_vec", (d,), init="ones")
                self._add_param(f"{pfx}.d_skip", (d,), init="ones")
                self._add_param(f"{pfx}.out_proj", (d, d))

            # Feed-forward sublayer on every block
            self._add_norm_params(f"{pfx}.norm2")
            self._add_param(f"{pfx}.ff_w1", (d, ff))
            self._add_param(f"{pfx}.ff_b1", (ff,), init="zeros")
            self._add_param(f"{pfx}.ff_w2", (ff, d), scale=float(0.5 / np.sqrt(ff * max(1, self.n_layers))))
            self._add_param(f"{pfx}.ff_b2", (d,), init="zeros")

        self._add_norm_params("final_norm")
        self._add_param("lm_head", (d, self.vocab_size), scale=0.02)

    def parameter_count(self) -> int:
        return int(sum(p.data.size for p in self.params.values()))

    def trainable_parameters(self) -> List[Parameter]:
        return [p for p in self.params.values() if p.requires_grad]

    def freeze_prefix_layers(self, num_layers_to_freeze: int = 1, freeze_embeddings: bool = True) -> List[str]:
        """Support real transfer learning by freezing embeddings and lower blocks."""
        frozen: List[str] = []
        for name, p in self.params.items():
            should_freeze = False
            if freeze_embeddings and name in ("tok_emb", "pos_emb"):
                should_freeze = True
            for idx in range(num_layers_to_freeze):
                if name.startswith(f"blocks.{idx}."):
                    should_freeze = True
            if should_freeze:
                p.requires_grad = False
                p.trainable = False
                p.grad = None
                frozen.append(name)
        return frozen

    def _norm(self, x: Tensor, prefix: str) -> Tensor:
        gamma = self.params[f"{prefix}.gamma"]
        if self.norm_type == "rmsnorm":
            return rms_norm(x, gamma)
        return layer_norm(x, gamma, self.params[f"{prefix}.beta"])

    def _mha(
        self,
        q_in: Tensor,
        kv_in: Tensor,
        wq: Parameter,
        wk: Parameter,
        wv: Parameter,
        wo: Parameter,
        causal: bool,
    ) -> Tensor:
        b, tq, d = q_in.shape
        tk = kv_in.shape[1]
        h = self.n_heads
        dh = d // h

        q = (q_in @ wq).reshape(b, tq, h, dh).transpose(0, 2, 1, 3)
        k = (kv_in @ wk).reshape(b, tk, h, dh).transpose(0, 2, 1, 3)
        v = (kv_in @ wv).reshape(b, tk, h, dh).transpose(0, 2, 1, 3)

        if self.positional_encoding == "rope" and tq == tk:
            q, k = _apply_rope_numpy(q, k)

        scale = np.float32(1.0 / np.sqrt(max(1, dh)))
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale
        if causal:
            mask = np.triu(np.full((tq, tk), -1e9, dtype=np.float32), k=1)
            scores = scores + mask[None, None, :, :]
        attn = scores.softmax(axis=-1)
        ctx = (attn @ v).transpose(0, 2, 1, 3).reshape(b, tq, d)
        return ctx @ wo

    def _slice_step(self, x: Tensor, t_idx: int) -> Tensor:
        req = x.requires_grad
        out = Tensor(x.data[:, t_idx, :], requires_grad=req, _children=(x,) if req else (), _op="slice_t")
        if req:

            def _bw() -> None:
                if out.grad is None:
                    return
                g = np.zeros_like(x.data)
                g[:, t_idx, :] = out.grad
                x._acc_grad(g)

            out._backward = _bw
        return out

    def _slice_cols(self, x: Tensor, start: int, end: int) -> Tensor:
        req = x.requires_grad
        out = Tensor(x.data[..., start:end], requires_grad=req, _children=(x,) if req else (), _op="slice_c")
        if req:

            def _bw() -> None:
                if out.grad is None:
                    return
                g = np.zeros_like(x.data)
                g[..., start:end] = out.grad
                x._acc_grad(g)

            out._backward = _bw
        return out

    def _run_recurrent_or_ssm(self, u: Tensor, pfx: str, btype: str) -> Tensor:
        b, t, d = u.shape
        if btype == "rnn":
            wx = self.params[f"{pfx}.wx"]
            wh = self.params[f"{pfx}.wh"]
            bias = self.params[f"{pfx}.b"]
            proj = (u @ wx) + bias
            h_prev = Tensor(np.zeros((b, d), dtype=np.float32))
            steps = []
            for ti in range(t):
                xt = self._slice_step(proj, ti)
                h_prev = (xt + (h_prev @ wh)).tanh()
                steps.append(h_prev)
            return stack_time(steps, axis=1)

        if btype == "gru":
            wx = self.params[f"{pfx}.wx"]
            wh = self.params[f"{pfx}.wh"]
            bias = self.params[f"{pfx}.b"]
            x_all = (u @ wx) + bias
            h_prev = Tensor(np.zeros((b, d), dtype=np.float32))
            steps = []
            for ti in range(t):
                xt = self._slice_step(x_all, ti)
                ht_lin = h_prev @ wh
                z = (self._slice_cols(xt, 0, d) + self._slice_cols(ht_lin, 0, d)).sigmoid()
                r = (self._slice_cols(xt, d, 2 * d) + self._slice_cols(ht_lin, d, 2 * d)).sigmoid()
                cand = (self._slice_cols(xt, 2 * d, 3 * d) + (r * self._slice_cols(ht_lin, 2 * d, 3 * d))).tanh()
                h_prev = (Tensor(np.ones_like(z.data)) - z) * h_prev + z * cand
                steps.append(h_prev)
            return stack_time(steps, axis=1)

        if btype == "lstm":
            wx = self.params[f"{pfx}.wx"]
            wh = self.params[f"{pfx}.wh"]
            bias = self.params[f"{pfx}.b"]
            x_all = (u @ wx) + bias
            h_prev = Tensor(np.zeros((b, d), dtype=np.float32))
            c_prev = Tensor(np.zeros((b, d), dtype=np.float32))
            steps = []
            for ti in range(t):
                gates = self._slice_step(x_all, ti) + (h_prev @ wh)
                i_g = self._slice_cols(gates, 0, d).sigmoid()
                f_g = self._slice_cols(gates, d, 2 * d).sigmoid()
                o_g = self._slice_cols(gates, 2 * d, 3 * d).sigmoid()
                g_g = self._slice_cols(gates, 3 * d, 4 * d).tanh()
                c_prev = f_g * c_prev + i_g * g_g
                h_prev = o_g * c_prev.tanh()
                steps.append(h_prev)
            return stack_time(steps, axis=1)

        # Diagonal SSM (S4D-lite with gated projection)
        in_proj = self.params[f"{pfx}.in_proj"]
        log_decay = self.params[f"{pfx}.log_decay"]
        b_vec = self.params[f"{pfx}.b_vec"]
        c_vec = self.params[f"{pfx}.c_vec"]
        d_skip = self.params[f"{pfx}.d_skip"]
        out_proj = self.params[f"{pfx}.out_proj"]

        xz = u @ in_proj
        x_in = self._slice_cols(xz, 0, d)
        gate = self._slice_cols(xz, d, 2 * d).silu()

        # Decay in (0, 1): sigmoid(log_decay) keeps recurrence strictly stable & differentiable
        a_bar = log_decay.sigmoid()
        state = Tensor(np.zeros((b, d), dtype=np.float32))
        ys = []
        for ti in range(t):
            xt = self._slice_step(x_in, ti)
            state = state * a_bar + xt * b_vec
            yt = state * c_vec + xt * d_skip
            ys.append(yt)
        y_seq = stack_time(ys, axis=1)
        return (y_seq * gate) @ out_proj

    def forward(self, input_ids: np.ndarray, training: bool = False) -> Tensor:
        ids = np.asarray(input_ids, dtype=np.int64)
        if ids.ndim == 1:
            ids = ids[None, :]
        if ids.shape[1] > self.context_length:
            ids = ids[:, -self.context_length :]
        _, t = ids.shape

        x = embedding_lookup(self.params["tok_emb"], ids)
        if self.positional_encoding == "learned":
            pos_idx = np.arange(t, dtype=np.int64)[None, :]
            x = x + embedding_lookup(self.params["pos_emb"], pos_idx)
        elif self.positional_encoding == "sinusoidal":
            pe = _sinusoidal_positions(t, self.d_model)[None, :, :]
            x = x + pe

        if training and self.dropout > 0.0:
            x = x.dropout(self.dropout, self.rng, training=True)

        # Reuse the same block parameters on each pass; gradients accumulate
        # through every pass during training. This increases compute, not weights.
        for _ in range(self.loop_passes):
            for i, btype in enumerate(self._block_plan):
                pfx = f"blocks.{i}"
                normed = self._norm(x, f"{pfx}.norm1")
                if btype in ("causal_attn", "bi_attn"):
                    sub = self._mha(
                        normed,
                        normed,
                        self.params[f"{pfx}.wq"],
                        self.params[f"{pfx}.wk"],
                        self.params[f"{pfx}.wv"],
                        self.params[f"{pfx}.wo"],
                        causal=(btype == "causal_attn"),
                    )
                    x = x + (sub.dropout(self.dropout, self.rng, training) if training else sub)
                elif btype == "seq2seq_block":
                    enc_in = self._norm(x, f"{pfx}.enc_norm")
                    enc_repr = x + self._mha(
                        enc_in,
                        enc_in,
                        self.params[f"{pfx}.enc_wq"],
                        self.params[f"{pfx}.enc_wk"],
                        self.params[f"{pfx}.enc_wv"],
                        self.params[f"{pfx}.enc_wo"],
                        causal=False,
                    )
                    dec_self = x + self._mha(
                        normed,
                        normed,
                        self.params[f"{pfx}.wq"],
                        self.params[f"{pfx}.wk"],
                        self.params[f"{pfx}.wv"],
                        self.params[f"{pfx}.wo"],
                        causal=True,
                    )
                    cross_in = self._norm(dec_self, f"{pfx}.cross_norm")
                    x = dec_self + self._mha(
                        cross_in,
                        enc_repr,
                        self.params[f"{pfx}.cwq"],
                        self.params[f"{pfx}.cwk"],
                        self.params[f"{pfx}.cwv"],
                        self.params[f"{pfx}.cwo"],
                        causal=True,
                    )
                elif btype in ("rnn", "gru", "lstm", "ssm"):
                    sub = self._run_recurrent_or_ssm(normed, pfx, btype)
                    x = x + (sub.dropout(self.dropout, self.rng, training) if training else sub)

                # Feed-forward network
                ff_in = self._norm(x, f"{pfx}.norm2")
                h = _apply_activation((ff_in @ self.params[f"{pfx}.ff_w1"]) + self.params[f"{pfx}.ff_b1"], self.activation)
                ff_out = (h @ self.params[f"{pfx}.ff_w2"]) + self.params[f"{pfx}.ff_b2"]
                x = x + (ff_out.dropout(self.dropout, self.rng, training) if training else ff_out)

        x = self._norm(x, "final_norm")
        logits = x @ self.params["lm_head"]
        return logits

    def compute_loss(self, input_ids: np.ndarray, target_ids: np.ndarray, training: bool = True) -> Tensor:
        logits = self.forward(input_ids, training=training)
        return sequence_cross_entropy(logits, target_ids, ignore_index=0)

    def get_weights_dict(self) -> Dict[str, np.ndarray]:
        return {name: p.data.copy() for name, p in self.params.items()}

    def load_weights_dict(self, weights: Dict[str, np.ndarray], strict: bool = True) -> List[str]:
        loaded: List[str] = []
        for name, arr in weights.items():
            if name in self.params:
                if self.params[name].data.shape != arr.shape:
                    if strict:
                        raise ValueError(
                            f"Shape mismatch for '{name}': expected {self.params[name].data.shape}, got {arr.shape}"
                        )
                    continue
                self.params[name].data = arr.astype(np.float32, copy=True)
                loaded.append(name)
            elif strict:
                raise KeyError(f"Unexpected weight key '{name}' in checkpoint.")
        return loaded


def build_model_from_spec(spec: Dict[str, Any], seed: int = 42) -> NeuralSequenceModel:
    return NeuralSequenceModel(spec=spec, seed=seed)
