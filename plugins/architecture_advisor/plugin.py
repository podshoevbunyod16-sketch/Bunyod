"""Built-in ModelForge plugin: Hardware-aware neural architecture advisor."""

from typing import Any, Dict


def run(inputs: Dict[str, Any]) -> Dict[str, Any]:
    ram_mb = int(inputs.get("ram_mb", 3072))
    ctx = int(inputs.get("context_length", 128))
    low_mem = ram_mb <= 3600

    if low_mem and ctx >= 256:
        rec = "ssm"
        why = "Linear-time diagonal SSM avoids quadratic attention memory on <=3.5 GB RAM devices."
    elif low_mem:
        rec = "hybrid_attention_ssm"
        why = "Alternating SSM and compact causal attention balances bilingual quality and low RAM footprint."
    else:
        rec = "decoder_transformer"
        why = "Sufficient RAM available for multi-head RoPE causal Transformer blocks."

    return {
        "recommended_architecture": rec,
        "low_memory_safe": True,
        "rationale": why,
    }
