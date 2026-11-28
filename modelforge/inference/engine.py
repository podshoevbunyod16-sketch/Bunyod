"""Streaming & batch text generation with top-k, top-p, repetition penalty, cancel & profiling."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from modelforge.core.events import EventBus
from modelforge.model_builder.builder import ModelBuilder
from modelforge.platforms.detector import get_resource_snapshot
from modelforge.tokenizers.tokenizer import load_tokenizer
from modelforge.training.checkpoints import load_checkpoint


@dataclass
class GenerationConfig:
    max_tokens: int = 32
    temperature: float = 0.8
    top_k: int = 20
    top_p: float = 0.92
    repetition_penalty: float = 1.15
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _sample_next_token(
    logits_1d: np.ndarray,
    generated_ids: List[int],
    cfg: GenerationConfig,
    rng: np.random.Generator,
    ban_ids: tuple[int, ...] = (0, 1, 2),
) -> int:
    scores = logits_1d.astype(np.float64).copy()
    for bid in ban_ids:
        if 0 <= bid < scores.shape[0]:
            scores[bid] = -1e9

    if cfg.repetition_penalty != 1.0 and generated_ids:
        for prev_id in set(generated_ids[-64:]):
            if 0 <= prev_id < scores.shape[0]:
                if scores[prev_id] > 0:
                    scores[prev_id] /= cfg.repetition_penalty
                else:
                    scores[prev_id] *= cfg.repetition_penalty

    if cfg.temperature <= 1e-4:
        return int(np.argmax(scores))

    scores = scores / float(cfg.temperature)

    if cfg.top_k > 0 and cfg.top_k < scores.shape[0]:
        kth = np.partition(scores, -cfg.top_k)[-cfg.top_k]
        scores = np.where(scores < kth, -1e9, scores)

    shifted = scores - np.max(scores)
    probs = np.exp(shifted)
    probs = probs / np.sum(probs)

    if 0.0 < cfg.top_p < 1.0:
        sorted_idx = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_idx]
        cum = np.cumsum(sorted_probs)
        cutoff_mask = cum > cfg.top_p
        # Always keep at least the highest-probability token
        cutoff_mask[0] = False
        probs[sorted_idx[cutoff_mask]] = 0.0
        total = np.sum(probs)
        probs = probs / total if total > 0 else np.ones_like(probs) / len(probs)

    return int(rng.choice(len(probs), p=probs))


class InferenceEngine:
    """
    Specification Section 8:
    Streaming generation, cancellation, temperature/top-p/top-k/repetition_penalty/max_tokens,
    context window truncation, and speed/latency/RAM profiling.
    """

    def __init__(self, model_builder: ModelBuilder, events: EventBus, mode_getter=None) -> None:
        self._models = model_builder
        self._events = events
        self._mode_getter = mode_getter or (lambda: "local")

    def stream_generate(
        self,
        model_id: str,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        checkpoint_path: str = "",
        cancel_event: Optional[threading.Event] = None,
    ) -> Iterator[Dict[str, Any]]:
        cfg = config or GenerationConfig()
        info = self._models.get_model(model_id)
        ckpt = checkpoint_path or info["latest_checkpoint"]
        loaded = load_checkpoint(ckpt)
        model = loaded["model"]
        tok_path = Path(loaded["tokenizer_path"] or info["tokenizer_path"])
        tokenizer = load_tokenizer(tok_path)

        rng = np.random.default_rng(cfg.seed)
        prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        if not prompt_ids:
            prompt_ids = [tokenizer.bos_id]

        generated: List[int] = []
        ctx_limit = model.context_length
        t_start = time.monotonic()
        first_token_latency_ms: Optional[float] = None

        for step in range(cfg.max_tokens):
            if cancel_event and cancel_event.is_set():
                yield {"event": "cancelled", "tokens_generated": len(generated)}
                break

            window = (prompt_ids + generated)[-ctx_limit:]
            logits = model.forward(np.asarray([window], dtype=np.int64), training=False)
            next_logits = logits.data[0, -1, :]
            next_id = _sample_next_token(
                next_logits,
                generated_ids=generated,
                cfg=cfg,
                rng=rng,
                ban_ids=(tokenizer.pad_id, tokenizer.unk_id, tokenizer.bos_id),
            )
            if first_token_latency_ms is None:
                first_token_latency_ms = round((time.monotonic() - t_start) * 1000.0, 2)

            if next_id == tokenizer.eos_id and len(generated) >= 4:
                break
            generated.append(next_id)
            text_so_far = tokenizer.decode(generated, skip_special=True)
            piece = tokenizer.decode([next_id], skip_special=True)
            chunk = {
                "event": "token",
                "step": step + 1,
                "token_id": next_id,
                "token_piece": piece,
                "generated_text": text_so_far,
            }
            self._events.publish(
                "inference.token",
                {"model_id": model_id, **chunk},
                project_id=info["project_id"],
            )
            yield chunk

        total_sec = max(1e-6, time.monotonic() - t_start)
        rss_mb = get_resource_snapshot().process_rss_mb
        full_generated = tokenizer.decode(generated, skip_special=True)
        summary = {
            "event": "completed",
            "model_id": model_id,
            "checkpoint_path": ckpt,
            "execution_mode": self._mode_getter(),
            "prompt": prompt,
            "generated_text": full_generated,
            "full_text": prompt + full_generated,
            "tokens_generated": len(generated),
            "context_window_used": min(ctx_limit, len(prompt_ids) + len(generated)),
            "context_window_max": ctx_limit,
            "first_token_latency_ms": first_token_latency_ms or 0.0,
            "total_duration_ms": round(total_sec * 1000.0, 2),
            "tokens_per_sec": round(len(generated) / total_sec, 2),
            "process_rss_mb": rss_mb,
        }
        yield summary

    def generate(
        self,
        model_id: str,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        checkpoint_path: str = "",
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        last_summary: Dict[str, Any] = {}
        for evt in self.stream_generate(
            model_id=model_id,
            prompt=prompt,
            config=config,
            checkpoint_path=checkpoint_path,
            cancel_event=cancel_event,
        ):
            if evt.get("event") in ("completed", "cancelled"):
                last_summary = evt
        return last_summary
