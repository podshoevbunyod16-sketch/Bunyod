"""Real Training Engine: train from scratch, fine-tune, transfer learn, resume & resource guards."""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from modelforge.architectures.models import NeuralSequenceModel, build_model_from_spec
from modelforge.core.events import EventBus
from modelforge.datasets.manager import DatasetManager
from modelforge.model_builder.builder import DEFAULT_MULTILINGUAL_SEED_CORPUS, ModelBuilder
from modelforge.platforms.detector import PlatformInfo, get_resource_snapshot
from modelforge.storage.artifacts import ArtifactStore
from modelforge.storage.database import StorageDatabase
from modelforge.tokenizers.tokenizer import BaseTokenizer, load_tokenizer
from modelforge.training.checkpoints import load_checkpoint, save_checkpoint, weights_sha256


@dataclass
class TrainingRunConfig:
    epochs: int = 4
    batch_size: int = 4
    learning_rate: float = 3e-3
    weight_decay: float = 0.01
    grad_accumulation: int = 1
    max_grad_norm: float = 1.0
    max_seq_len: int = 64
    early_stopping_patience: int = 3
    max_seconds: int = 180
    max_ram_mb: int = 2048
    seed: int = 42
    mode: str = "scratch"  # scratch | finetune | transfer
    freeze_layers: int = 1
    resume_from_checkpoint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AdamWOptimizer:
    """Real AdamW / SGD parameter optimizer with state serialization for checkpoint resume."""

    def __init__(
        self,
        model: NeuralSequenceModel,
        lr: float = 3e-3,
        weight_decay: float = 0.01,
        kind: str = "adamw",
        state: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.model = model
        self.base_lr = lr
        self.lr = lr
        self.weight_decay = weight_decay
        self.kind = kind.lower()
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.step_num = 0
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}
        for name, p in model.params.items():
            self.m[name] = np.zeros_like(p.data)
            self.v[name] = np.zeros_like(p.data)
        if state:
            if "__step__" in state:
                self.step_num = int(state["__step__"][0])
            for name in model.params:
                if f"m.{name}" in state:
                    self.m[name] = state[f"m.{name}"].copy()
                if f"v.{name}" in state:
                    self.v[name] = state[f"v.{name}"].copy()

    def clip_grad_norm(self, max_norm: float) -> float:
        sq_sum = 0.0
        for p in self.model.trainable_parameters():
            if p.grad is not None:
                sq_sum += float(np.sum(p.grad * p.grad))
        total_norm = math.sqrt(sq_sum)
        if max_norm > 0 and total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-8)
            for p in self.model.trainable_parameters():
                if p.grad is not None:
                    p.grad = p.grad * np.float32(scale)
        return total_norm

    def step(self) -> None:
        self.step_num += 1
        for name, p in self.model.params.items():
            if not p.requires_grad or p.grad is None:
                continue
            g = p.grad
            if self.kind == "sgd":
                if self.weight_decay > 0:
                    g = g + np.float32(self.weight_decay) * p.data
                self.m[name] = np.float32(0.9) * self.m[name] + g
                p.data = p.data - np.float32(self.lr) * self.m[name]
            else:
                # Decoupled AdamW weight decay
                if self.weight_decay > 0 and p.data.ndim >= 2:
                    p.data = p.data * np.float32(1.0 - self.lr * self.weight_decay)
                self.m[name] = np.float32(self.beta1) * self.m[name] + np.float32(1.0 - self.beta1) * g
                self.v[name] = np.float32(self.beta2) * self.v[name] + np.float32(1.0 - self.beta2) * (g * g)
                m_hat = self.m[name] / np.float32(1.0 - (self.beta1 ** self.step_num))
                v_hat = self.v[name] / np.float32(1.0 - (self.beta2 ** self.step_num))
                p.data = p.data - np.float32(self.lr) * (m_hat / (np.sqrt(v_hat) + np.float32(self.eps)))

    def zero_grad(self) -> None:
        for p in self.model.params.values():
            p.zero_grad()

    def state_dict(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {"__step__": np.array([self.step_num], dtype=np.float32)}
        for name in self.model.params:
            out[f"m.{name}"] = self.m[name]
            out[f"v.{name}"] = self.v[name]
        return out


def _build_token_windows(
    texts: List[str],
    tokenizer: BaseTokenizer,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    stream: List[int] = []
    for txt in texts:
        stream.extend(tokenizer.encode(txt, add_bos=True, add_eos=True))
    # Guarantee enough tokens even for tiny datasets by repeating deterministically if shorter than seq_len + 2
    min_needed = max(seq_len + 2, 24)
    if len(stream) < min_needed:
        repeats = (min_needed // max(1, len(stream))) + 2
        stream = (stream * repeats)[:min_needed]

    win = min(seq_len, max(8, len(stream) - 1))
    stride = max(4, win // 2)
    xs: List[List[int]] = []
    ys: List[List[int]] = []
    for start in range(0, len(stream) - win, stride):
        chunk = stream[start : start + win + 1]
        xs.append(chunk[:-1])
        ys.append(chunk[1:])
    if not xs:
        xs.append(stream[:-1])
        ys.append(stream[1:])
    return np.asarray(xs, dtype=np.int64), np.asarray(ys, dtype=np.int64)


class TrainingEngine:
    """
    Specification Section 4:
    Real training loop with dataset validation, reproducible seeds, gradient accumulation,
    dynamic batch backoff, CPU/RAM/time guards, checkpointing, resume, early stopping,
    and mathematical verification of weight mutation (`weights_changed == True`).
    """

    def __init__(
        self,
        db: StorageDatabase,
        artifacts: ArtifactStore,
        events: EventBus,
        model_builder: ModelBuilder,
        dataset_manager: DatasetManager,
        platform_info: PlatformInfo,
    ) -> None:
        self._db = db
        self._artifacts = artifacts
        self._events = events
        self._models = model_builder
        self._datasets = dataset_manager
        self._platform = platform_info

    def train(
        self,
        model_id: str,
        config: TrainingRunConfig,
        dataset_id: str = "",
        inline_texts: Optional[List[str]] = None,
        cancel_event: Optional[threading.Event] = None,
        progress_cb: Optional[Callable[[str, float, str], None]] = None,
    ) -> Dict[str, Any]:
        model_info = self._models.get_model(model_id)
        project_id = model_info["project_id"]
        spec = dict(model_info["spec"])
        tok_path = Path(model_info["tokenizer_path"])
        tokenizer = load_tokenizer(tok_path)

        # Enforce Low Memory Mode safeguards automatically
        effective_batch = config.batch_size
        effective_seq = min(config.max_seq_len, int(spec.get("context_length", 128)))
        if self._platform.low_memory_mode:
            effective_batch = min(effective_batch, self._platform.recommended_max_batch_size)
            effective_seq = min(effective_seq, self._platform.recommended_max_seq_len)

        # Load or initialize model + optional resume/finetune checkpoint
        start_epoch = 0
        opt_state = None
        resume_ckpt = config.resume_from_checkpoint or (
            model_info["latest_checkpoint"] if config.mode in ("finetune", "transfer") else ""
        )
        if resume_ckpt and Path(resume_ckpt).exists():
            loaded = load_checkpoint(resume_ckpt, load_optimizer=(config.mode == "scratch" and bool(config.resume_from_checkpoint)))
            model = loaded["model"]
            if config.resume_from_checkpoint:
                start_epoch = int(loaded["manifest"].get("epoch", 0))
                opt_state = loaded["optimizer_state"]
        else:
            model = build_model_from_spec(spec, seed=config.seed)

        frozen_params: List[str] = []
        if config.mode == "transfer":
            frozen_params = model.freeze_prefix_layers(
                num_layers_to_freeze=max(1, min(config.freeze_layers, model.n_layers - 1 if model.n_layers > 1 else 0)),
                freeze_embeddings=True,
            )

        # Prepare train / val splits
        dataset_sha = "inline_seed_corpus"
        if dataset_id:
            ds_meta = self._datasets.get_dataset(dataset_id)
            dataset_sha = ds_meta["sha256"]
            splits = self._datasets.load_splits(dataset_id)
            train_texts = splits["train"]
            val_texts = splits["val"] or splits["train"][:1]
        elif inline_texts:
            ingested = self._datasets.ingest_dataset(
                project_id=project_id,
                name=f"{model_info['name']}-corpus",
                raw_content="\n".join(inline_texts),
                fmt="txt",
                license_name="user-provided",
                seed=config.seed,
            )
            dataset_id = ingested["id"]
            dataset_sha = ingested["sha256"]
            splits = self._datasets.load_splits(dataset_id)
            train_texts = splits["train"]
            val_texts = splits["val"] or splits["train"][:1]
        else:
            train_texts = list(DEFAULT_MULTILINGUAL_SEED_CORPUS)
            val_texts = DEFAULT_MULTILINGUAL_SEED_CORPUS[:2]

        x_train, y_train = _build_token_windows(train_texts, tokenizer, effective_seq)
        x_val, y_val = _build_token_windows(val_texts, tokenizer, effective_seq)

        initial_weights = model.get_weights_dict()
        initial_sha = weights_sha256(initial_weights)

        optimizer = AdamWOptimizer(
            model=model,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            kind=str(spec.get("optimizer", "adamw")),
            state=opt_state,
        )

        rng = np.random.default_rng(config.seed)
        n_samples = x_train.shape[0]
        total_epochs = start_epoch + config.epochs
        steps_per_epoch = max(1, math.ceil(n_samples / effective_batch))
        total_steps = max(1, config.epochs * steps_per_epoch)
        warmup_steps = max(1, total_steps // 6)

        history: List[Dict[str, Any]] = []
        best_val_loss = float("inf")
        bad_epochs = 0
        stopped_reason = "completed"
        global_step = optimizer.step_num
        t0 = time.monotonic()

        # Measure initial loss before step 1
        init_train_loss = float(model.compute_loss(x_train[: min(4, n_samples)], y_train[: min(4, n_samples)], training=False).data)

        for epoch_idx in range(start_epoch + 1, total_epochs + 1):
            if cancel_event and cancel_event.is_set():
                stopped_reason = "cancelled_by_user"
                break

            elapsed = time.monotonic() - t0
            snap = get_resource_snapshot()
            if elapsed > config.max_seconds:
                stopped_reason = f"max_time_limit_reached ({config.max_seconds}s)"
                break
            if snap.process_rss_mb > config.max_ram_mb:
                if effective_batch > 1:
                    effective_batch = max(1, effective_batch // 2)
                else:
                    stopped_reason = f"max_ram_limit_reached ({snap.process_rss_mb} MB > {config.max_ram_mb} MB)"
                    break
            elif snap.low_memory_pressure and effective_batch > 1:
                # Dynamic batch-size reduction under memory pressure (Section 4)
                effective_batch = max(1, effective_batch // 2)

            perm = rng.permutation(n_samples)
            optimizer.zero_grad()
            accum_counter = 0
            epoch_losses: List[float] = []
            grad_norm = 0.0

            for b_start in range(0, n_samples, effective_batch):
                if cancel_event and cancel_event.is_set():
                    stopped_reason = "cancelled_by_user"
                    break

                batch_idx = perm[b_start : b_start + effective_batch]
                bx = x_train[batch_idx]
                by = y_train[batch_idx]

                loss_tensor = model.compute_loss(bx, by, training=True)
                loss_val = float(loss_tensor.data)
                if not math.isfinite(loss_val):
                    stopped_reason = "diverged_nan_loss"
                    break
                epoch_losses.append(loss_val)

                scaled_loss = loss_tensor * (1.0 / float(max(1, config.grad_accumulation)))
                scaled_loss.backward()
                accum_counter += 1

                if accum_counter >= config.grad_accumulation or (b_start + effective_batch >= n_samples):
                    global_step += 1
                    # Cosine warmup schedule
                    sched_kind = str(spec.get("scheduler", "cosine_warmup")).lower()
                    progress_ratio = min(1.0, global_step / float(max(1, total_steps)))
                    if sched_kind == "cosine_warmup":
                        if global_step <= warmup_steps:
                            optimizer.lr = config.learning_rate * (global_step / float(warmup_steps))
                        else:
                            cos_prog = (global_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                            optimizer.lr = config.learning_rate * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * cos_prog)))
                    elif sched_kind == "linear":
                        optimizer.lr = config.learning_rate * max(0.1, 1.0 - 0.9 * progress_ratio)
                    else:
                        optimizer.lr = config.learning_rate

                    grad_norm = optimizer.clip_grad_norm(config.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    accum_counter = 0

            if stopped_reason not in ("completed",):
                break

            avg_train_loss = float(np.mean(epoch_losses)) if epoch_losses else init_train_loss
            val_loss = float(model.compute_loss(x_val[: min(8, x_val.shape[0])], y_val[: min(8, y_val.shape[0])], training=False).data)
            ppl = round(math.exp(min(20.0, val_loss)), 4)

            epoch_entry = {
                "epoch": epoch_idx,
                "step": global_step,
                "train_loss": round(avg_train_loss, 5),
                "val_loss": round(val_loss, 5),
                "perplexity": ppl,
                "lr": round(optimizer.lr, 6),
                "grad_norm": round(grad_norm, 4),
                "batch_size": effective_batch,
                "rss_mb": get_resource_snapshot().process_rss_mb,
            }
            history.append(epoch_entry)

            frac = (epoch_idx - start_epoch) / float(max(1, config.epochs))
            msg = (
                f"Epoch {epoch_idx}/{total_epochs} | train_loss={avg_train_loss:.4f} "
                f"| val_loss={val_loss:.4f} | ppl={ppl:.2f} | lr={optimizer.lr:.5f}"
            )
            if progress_cb:
                progress_cb(f"epoch_{epoch_idx}", frac, msg)
            self._events.publish(
                "training.step",
                {"model_id": model_id, **epoch_entry},
                project_id=project_id,
            )

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                bad_epochs = 0
            else:
                bad_epochs += 1
                if config.early_stopping_patience > 0 and bad_epochs >= config.early_stopping_patience:
                    stopped_reason = f"early_stopping (patience={config.early_stopping_patience})"
                    break

        duration_sec = round(time.monotonic() - t0, 4)
        final_weights = model.get_weights_dict()
        final_sha = weights_sha256(final_weights)
        l2_delta = float(
            math.sqrt(
                sum(
                    float(np.sum((final_weights[k] - initial_weights[k]) ** 2))
                    for k in final_weights
                )
            )
        )
        weights_changed = (final_sha != initial_sha) and (l2_delta > 0.0)
        final_train_loss = float(model.compute_loss(x_train[: min(4, n_samples)], y_train[: min(4, n_samples)], training=False).data)
        final_val_loss = history[-1]["val_loss"] if history else final_train_loss

        new_version = int(model_info["version"]) + 1
        ckpt_dir = self._artifacts.checkpoints_dir / model_id / f"v{new_version}_{config.mode}"
        metrics_summary = {
            "mode": config.mode,
            "stopped_reason": stopped_reason,
            "epochs_run": len(history),
            "global_step": global_step,
            "initial_train_loss": round(init_train_loss, 5),
            "final_train_loss": round(final_train_loss, 5),
            "final_val_loss": round(final_val_loss, 5),
            "perplexity": round(math.exp(min(20.0, final_val_loss)), 4),
            "initial_weights_sha256": initial_sha,
            "final_weights_sha256": final_sha,
            "weight_l2_delta": round(l2_delta, 6),
            "weights_changed": weights_changed,
            "frozen_parameters": frozen_params,
            "duration_sec": duration_sec,
            "dataset_id": dataset_id,
            "dataset_sha256": dataset_sha,
            "history": history,
        }

        ckpt_saved = save_checkpoint(
            checkpoint_dir=ckpt_dir,
            model=model,
            spec=spec,
            tokenizer_path=tok_path,
            step=global_step,
            epoch=start_epoch + len(history),
            metrics=metrics_summary,
            optimizer_state=optimizer.state_dict(),
        )

        now = time.time()
        status_label = "trained" if weights_changed else "stopped"
        self._db.execute(
            "UPDATE models SET version = ?, status = ?, latest_checkpoint = ?, metrics_json = ?, updated_at = ? WHERE id = ?",
            (
                new_version,
                status_label,
                ckpt_saved["checkpoint_path"],
                json.dumps(metrics_summary, ensure_ascii=False),
                now,
                model_id,
            ),
        )
        self._db.upsert_row(
            "model_versions",
            {
                "id": f"{model_id}-v{new_version}",
                "model_id": model_id,
                "version": new_version,
                "spec_json": json.dumps(spec, ensure_ascii=False),
                "checkpoint_path": ckpt_saved["checkpoint_path"],
                "checkpoint_sha256": ckpt_saved["sha256"],
                "metrics_json": json.dumps(metrics_summary, ensure_ascii=False),
                "notes": f"Training ({config.mode}): {stopped_reason}",
                "created_at": now,
            },
        )
        self._events.publish(
            "training.completed",
            {"model_id": model_id, "version": new_version, "checkpoint": ckpt_saved, "metrics": metrics_summary},
            project_id=project_id,
        )
        return {
            "model_id": model_id,
            "version": new_version,
            "checkpoint": ckpt_saved,
            "metrics": metrics_summary,
        }
