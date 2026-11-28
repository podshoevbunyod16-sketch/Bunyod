"""Real model evaluation, chrF similarity, latency/memory benchmark & side-by-side comparison."""

from __future__ import annotations

import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from modelforge.datasets.manager import DatasetManager
from modelforge.inference.engine import GenerationConfig, InferenceEngine
from modelforge.model_builder.builder import DEFAULT_MULTILINGUAL_SEED_CORPUS, ModelBuilder
from modelforge.platforms.detector import get_resource_snapshot
from modelforge.tokenizers.tokenizer import load_tokenizer
from modelforge.training.checkpoints import load_checkpoint
from modelforge.training.engine import _build_token_windows


def compute_char_f1(prediction: str, reference: str, n: int = 3) -> float:
    """Character n-gram F1 (chrF-lite) for multilingual Russian/Tajik/English text quality."""
    if not prediction or not reference:
        return 0.0
    p_grams = Counter(prediction[i : i + n] for i in range(max(1, len(prediction) - n + 1)))
    r_grams = Counter(reference[i : i + n] for i in range(max(1, len(reference) - n + 1)))
    overlap = sum((p_grams & r_grams).values())
    prec = overlap / max(1, sum(p_grams.values()))
    rec = overlap / max(1, sum(r_grams.values()))
    if prec + rec == 0.0:
        return 0.0
    return round(2.0 * prec * rec / (prec + rec), 4)


class EvaluationEngine:
    """
    Specification Section 2 & 3:
    Computes ONLY real measured metrics (cross-entropy loss, perplexity, next-token accuracy,
    generation throughput tok/s, RSS RAM, parameter size) — zero fabricated numbers.
    """

    def __init__(
        self,
        model_builder: ModelBuilder,
        dataset_manager: DatasetManager,
        inference_engine: InferenceEngine,
    ) -> None:
        self._models = model_builder
        self._datasets = dataset_manager
        self._inference = inference_engine

    def evaluate_model(
        self,
        model_id: str,
        dataset_id: str = "",
        eval_texts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        info = self._models.get_model(model_id)
        loaded = load_checkpoint(info["latest_checkpoint"])
        model = loaded["model"]
        tokenizer = load_tokenizer(Path(loaded["tokenizer_path"] or info["tokenizer_path"]))

        if dataset_id:
            splits = self._datasets.load_splits(dataset_id)
            texts = splits["test"] or splits["val"] or splits["train"]
        elif eval_texts:
            texts = eval_texts
        else:
            texts = list(DEFAULT_MULTILINGUAL_SEED_CORPUS)

        seq_len = min(64, model.context_length)
        xs, ys = _build_token_windows(texts, tokenizer, seq_len=seq_len)

        t0 = time.monotonic()
        logits = model.forward(xs, training=False)
        loss = float(model.compute_loss(xs, ys, training=False).data)
        eval_ms = round((time.monotonic() - t0) * 1000.0, 2)

        preds = np.argmax(logits.data, axis=-1)
        mask = ys != 0
        correct = int(np.sum((preds == ys) & mask))
        total_tok = int(np.sum(mask))
        token_accuracy = round(correct / max(1, total_tok), 4)
        perplexity = round(math.exp(min(20.0, loss)), 4)

        sample_prompt = texts[0][:24] if texts else "Привет"
        gen = self._inference.generate(
            model_id=model_id,
            prompt=sample_prompt,
            config=GenerationConfig(max_tokens=16, temperature=0.5, seed=42),
        )
        chrf = compute_char_f1(gen.get("full_text", ""), texts[0]) if texts else 0.0
        snap = get_resource_snapshot()

        return {
            "model_id": model_id,
            "model_name": info["name"],
            "architecture": info["architecture"],
            "version": info["version"],
            "parameter_count": model.parameter_count(),
            "eval_windows": int(xs.shape[0]),
            "eval_tokens": total_tok,
            "loss": round(loss, 5),
            "perplexity": perplexity,
            "next_token_accuracy": token_accuracy,
            "chrf3_sample": chrf,
            "eval_forward_ms": eval_ms,
            "generation_tokens_per_sec": gen.get("tokens_per_sec", 0.0),
            "process_rss_mb": snap.process_rss_mb,
            "sample_generation": gen.get("full_text", ""),
        }

    def compare_models(
        self,
        model_ids: List[str],
        dataset_id: str = "",
        eval_texts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        results = [
            self.evaluate_model(mid, dataset_id=dataset_id, eval_texts=eval_texts)
            for mid in model_ids
        ]
        ranked = sorted(results, key=lambda r: (r["loss"], r["parameter_count"]))
        return {
            "compared_count": len(results),
            "best_by_loss": ranked[0]["model_id"] if ranked else None,
            "ranking": ranked,
        }
