"""Small bounded on-device transformer trainer with no network or server dependencies.

The caller supplies a private storage directory. This module only needs NumPy and
ModelForge's tensor engine; it does not start a server or fetch data. It is not
an Android integration: the application must bundle Python/NumPy and call it.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from modelforge.architectures.models import build_model_from_spec
from modelforge.tokenizers.tokenizer import CharTokenizer

MAX_CHARS = 20_000
MAX_VOCAB = 512
CONTEXT = 16
MAX_WINDOWS = 24


def _validate_data(text: str, purpose: str, target_mb: float, passes: int, available_mb: int) -> None:
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_CHARS:
        raise ValueError(f'Training text must contain 1–{MAX_CHARS} characters.')
    if not isinstance(purpose, str) or not purpose.strip() or len(purpose) > 500:
        raise ValueError('A purpose of at most 500 characters is required.')
    if isinstance(target_mb, bool) or not isinstance(target_mb, (int, float)) or not math.isfinite(target_mb) or not 0.1 <= target_mb <= 10:
        raise ValueError('Target weight must be between 0.1 and 10 MiB.')
    if isinstance(passes, bool) or not isinstance(passes, int) or not 1 <= passes <= 8:
        raise ValueError('Passes must be an integer between 1 and 8.')
    if isinstance(available_mb, bool) or not isinstance(available_mb, int) or available_mb < 256:
        raise ValueError('At least 256 MiB of available memory is required.')


def train(home: str, purpose: str, text: str, target_mb: float = 1.0,
          passes: int = 1, available_mb: int = 512, max_seconds: float = 90) -> dict[str, Any]:
    """Train real transformer weights on caller-provided text and atomically save one model.

    Bounds limit resource use but are not a guarantee against OS termination.
    Re-training replaces the previous model only after a successful checkpoint.
    """
    _validate_data(text, purpose, target_mb, passes, available_mb)
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError('max_seconds must be positive.')
    tokenizer = CharTokenizer.train([text])
    if tokenizer.vocab_size > MAX_VOCAB:
        raise ValueError(f'Text contains too many distinct characters (maximum vocabulary {MAX_VOCAB}).')
    candidates = []
    for width in (16, 24, 32, 48, 64):
        spec = {'name': 'offline', 'architecture': 'tiny_transformer',
                'vocab_size': tokenizer.vocab_size, 'd_model': width, 'n_layers': 1,
                'n_heads': 4, 'context_length': CONTEXT, 'dropout': 0.0,
                'positional_encoding': 'sinusoidal', 'loop_passes': passes}
        model = build_model_from_spec(spec, seed=42)
        size_mb = model.parameter_count() * 4 / 1024**2
        if size_mb <= target_mb:
            candidates.append((size_mb, spec, model))
    if not candidates:
        raise ValueError('Requested weight is too small for this vocabulary.')
    size_mb, spec, model = candidates[-1]
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    if len(ids) < 3:
        raise ValueError('Training text is too short.')
    # Windows are bounded and evenly distributed over the whole text.
    length = min(CONTEXT, len(ids) - 1)
    positions = np.linspace(0, len(ids) - length - 1,
                            min(MAX_WINDOWS, max(1, len(ids) // length)), dtype=int)
    before = {name: p.data.copy() for name, p in model.params.items()}
    started = time.monotonic()
    losses: list[float] = []
    for epoch in range(2):
        for start in positions:
            if losses and time.monotonic() - started >= max_seconds:
                break
            chunk = ids[int(start):int(start) + length + 1]
            x = np.array([chunk[:-1]], dtype=np.int64)
            y = np.array([chunk[1:]], dtype=np.int64)
            for p in model.trainable_parameters():
                p.zero_grad()
            loss = model.compute_loss(x, y, training=True)
            if not np.isfinite(loss.data):
                raise ValueError('Training diverged; the previous model was kept.')
            loss.backward()
            grads = [p.grad for p in model.trainable_parameters() if p.grad is not None]
            norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads))
            if not math.isfinite(norm):
                raise ValueError('Non-finite gradients; the previous model was kept.')
            scale = min(1.0, 1.0 / max(norm, 1e-8))
            for p in model.trainable_parameters():
                if p.grad is not None:
                    p.data -= np.float32(0.03 * scale) * p.grad
            losses.append(float(loss.data))
        if time.monotonic() - started >= max_seconds:
            break
    changed = any(not np.array_equal(before[k], p.data) for k, p in model.params.items())
    if not changed:
        raise ValueError('No weights changed; the previous model was kept.')
    metadata = {'schema': 1, 'purpose': purpose.strip(), 'spec': spec,
                'vocabulary': tokenizer.id_to_token, 'steps': len(losses),
                'loss_first': losses[0], 'loss_last': losses[-1],
                'weight_mb': round(size_mb, 4), 'requested_weight_mb': target_mb,
                'seconds': round(time.monotonic() - started, 2)}
    folder = Path(home)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'offline_model.npz'
    fd, temporary = tempfile.mkstemp(prefix='.offline-', suffix='.npz', dir=folder)
    try:
        with os.fdopen(fd, 'wb') as f:
            np.savez_compressed(f, metadata=np.array(json.dumps(metadata, ensure_ascii=False)),
                                **model.get_weights_dict())
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return metadata


def generate(home: str, prompt: str, max_chars: int = 64) -> str:
    """Load the saved model and generate characters; no network or external service."""
    if not isinstance(prompt, str) or len(prompt) > 5000:
        raise ValueError('Prompt must contain at most 5000 characters.')
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or not 1 <= max_chars <= 128:
        raise ValueError('max_chars must be between 1 and 128.')
    with np.load(Path(home) / 'offline_model.npz', allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        if metadata.get('schema') != 1:
            raise ValueError('Unsupported model format.')
        tokenizer = CharTokenizer(metadata['vocabulary'])
        model = build_model_from_spec(metadata['spec'])
        model.load_weights_dict({key: archive[key] for key in archive.files if key != 'metadata'})
    ids = tokenizer.encode(prompt, add_bos=True)
    output = []
    for _ in range(max_chars):
        logits = model.forward(np.array([ids[-CONTEXT:]], dtype=np.int64)).data[0, -1]
        # Avoid emitting padding/unknown/BOS characters.
        logits[:3] = -np.inf
        token = int(np.argmax(logits))
        if token == tokenizer.eos_id:
            break
        output.append(token)
        ids.append(token)
    return tokenizer.decode(output)
