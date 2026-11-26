"""
Tool registry for v0.2 — external tools selected via associative search.

Each tool: name, description HV (for CAM selection), validated function.
Selection is by HV similarity (associative), execution is validated
(no arbitrary code). Unknown tool request -> refusal if sim < tau.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Any, Callable


class ToolRegistry:
    def __init__(self, D: int = 256, seed: int = 0, tau: float = 0.4):
        rng = np.random.default_rng(seed)
        self.D = D
        self.tau = tau
        self.rng = rng
        self.tools: Dict[str, Dict[str, Any]] = {}
        # built-in safe tools (pure functions, validated args)
        self._register("calculator_add", "add two integers", lambda a, b: int(a) + int(b))
        self._register("calculator_mul", "multiply two integers", lambda a, b: int(a) * int(b))
        self._register("string_upper", "convert string to uppercase", lambda s: str(s).upper())
        self._register("string_len", "length of string", lambda s: len(str(s)))

    def _register(self, name: str, desc: str, fn: Callable):
        hv = self.rng.choice(np.array([-1.0, 1.0]), size=self.D)
        self.tools[name] = {"desc": desc, "hv": hv, "fn": fn}

    def select(self, query_hv: np.ndarray, tau: float | None = None):
        tau = self.tau if tau is None else tau
        best_name, best_sim = None, -2.0
        for name, info in self.tools.items():
            sim = float(np.mean(info["hv"] * np.asarray(query_hv)))
            if sim > best_sim:
                best_sim, best_name = sim, name
        if best_sim < tau:
            return None, best_sim, True
        return best_name, best_sim, False

    def call(self, name: str, *args):
        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}")
        # basic validation: args must be int/str, bounded length
        for a in args:
            if isinstance(a, str) and len(a) > 200:
                raise ValueError("String arg too long")
            if isinstance(a, int) and abs(a) > 10**6:
                raise ValueError("Int arg out of bounds")
        return self.tools[name]["fn"](*args)

    def query_hv_for(self, name: str) -> np.ndarray:
        return self.tools[name]["hv"].copy()
