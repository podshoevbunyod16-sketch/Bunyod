"""Plan G — reliability, offline guarantees, and numbers on a phone-class device.

Three concerns, three modules, one rule: nothing here is allowed to *claim*
robustness.  Fuzzing reports crashes, the offline scan reports network imports,
the benchmark reports milliseconds and bytes — and the Termux note says what was
not measured.
"""
from __future__ import annotations

from .fuzz import FUZZ_SEED, fuzz_pipeline, generate_inputs, render_fuzz
from .offline import NETWORK_MODULES, scan_offline, scan_paths
from .perf import benchmark, render_benchmark, scale_test

__all__ = ["FUZZ_SEED", "generate_inputs", "fuzz_pipeline", "render_fuzz",
           "NETWORK_MODULES", "scan_offline", "scan_paths", "benchmark",
           "render_benchmark", "scale_test"]
