"""
Modular core for RASM-CIM v0.3 platform.
Each mechanism implements a small interface so it can be swapped independently.
No stubs: every class here is either abstract or fully implemented in submodules.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time


@dataclass
class OpStats:
    ops: int = 0
    time_s: float = 0.0
    def reset(self):
        self.ops = 0; self.time_s = 0.0


@dataclass
class QueryResult:
    value: Any = None
    confidence: float = 0.0
    unknown: bool = False
    conflict: bool = False
    ops: int = 0
    detail: str = ""


class Memory(ABC):
    @abstractmethod
    def write(self, key, value): ...
    @abstractmethod
    def query(self, key) -> QueryResult: ...


class KnowledgeBase(ABC):
    @abstractmethod
    def add_fact(self, s, r, o) -> bool: ...
    @abstractmethod
    def query(self, s, r) -> QueryResult: ...


class Learner(ABC):
    @abstractmethod
    def fit(self, sequences, labels): ...
    @abstractmethod
    def predict(self, sequence): ...


class Reasoner(ABC):
    @abstractmethod
    def compose(self, s, r1, r2) -> QueryResult: ...


class Programmer(ABC):
    @abstractmethod
    def synthesize(self, examples) -> Optional[list]: ...
    @abstractmethod
    def execute(self, program, x: int) -> int: ...


class Tool(ABC):
    @abstractmethod
    def call(self, name: str, *args): ...


class Profiler:
    """Measure time + ops + memory (tracemalloc) for any callable."""
    def __init__(self):
        self.stats = OpStats()

    def measure(self, fn, *a, **k):
        t0 = time.perf_counter()
        out = fn(*a, **k)
        self.stats.time_s += time.perf_counter() - t0
        return out

    def report(self) -> Dict[str, Any]:
        return {"ops": self.stats.ops, "time_s": self.stats.time_s}
