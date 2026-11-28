"""Conservative, deterministic sizing for small experimental decoder models."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from modelforge.model_builder.builder import ModelSpec, estimate_training_resources
from modelforge.platforms.detector import PlatformInfo


@dataclass(frozen=True)
class AutomaticPlan:
    spec: ModelSpec
    estimated_weight_mb: float
    estimated_peak_ram_mb: float
    target_weight_mb: float
    target_met: bool


def plan_model(target_weight_mb: float, loop_passes: int, vocab_size: int,
               platform: PlatformInfo) -> AutomaticPlan:
    """Choose a bounded architecture; weight target means FP32 weights, not RAM/APK size."""
    if not isfinite(target_weight_mb) or not 0.1 <= target_weight_mb <= 10:
        raise ValueError('Target weight must be between 0.1 and 10 MiB.')
    if not 1 <= vocab_size <= 320:
        raise ValueError('Unexpected tokenizer vocabulary size.')
    candidates = []
    for width in (16, 24, 32, 48, 64, 96, 128, 192, 256):
        spec = ModelSpec(name='Моя модель', architecture='tiny_transformer',
                         d_model=width, n_heads=4, n_layers=2, vocab_size=max(32, vocab_size),
                         context_length=64, batch_size=1, tokenizer_type='char',
                         loop_passes=loop_passes)
        estimate = estimate_training_resources(spec, platform)
        count = estimate['parameter_count']
        # Even on desktop this workflow is deliberately restricted to compact models.
        if (count > min(platform.recommended_max_params, 2_500_000)
                or estimate['approx_peak_ram_mb'] > platform.available_ram_mb * 0.5):
            continue
        size = count * 4 / (1024 * 1024)
        candidates.append((size, estimate, spec))
    if not candidates:
        raise ValueError('Not enough available RAM for a compact model.')
    under = [item for item in candidates if item[0] <= target_weight_mb]
    size, estimate, spec = max(under, key=lambda item: item[0]) if under else min(candidates, key=lambda item: item[0])
    return AutomaticPlan(spec=spec, estimated_weight_mb=round(size, 4),
                         estimated_peak_ram_mb=estimate['approx_peak_ram_mb'],
                         target_weight_mb=target_weight_mb, target_met=bool(under))
