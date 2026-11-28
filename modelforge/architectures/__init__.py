"""Neural architectures, reverse-mode autograd tensor engine & optional PyTorch adapter."""

from modelforge.architectures.models import (
    SUPPORTED_ARCHITECTURES,
    NeuralSequenceModel,
    build_model_from_spec,
)
from modelforge.architectures.pytorch_adapter import is_pytorch_available
from modelforge.architectures.tensor_engine import Parameter, Tensor, sequence_cross_entropy

__all__ = [
    "NeuralSequenceModel",
    "Parameter",
    "SUPPORTED_ARCHITECTURES",
    "Tensor",
    "build_model_from_spec",
    "is_pytorch_available",
    "sequence_cross_entropy",
]
