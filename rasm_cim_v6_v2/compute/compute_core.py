"""
Compact compute core investigation (§8): no Python at runtime (C++/Rust
feasible), modular, efficient memory structures, parallel/async, limited RAM,
pluggable modules, offline. Assessment + prototype of packed storage.
Does NOT rewrite whole project before new core advantage is proven.
"""
from __future__ import annotations
import struct
import numpy as np


class PackedStore:
    """Packed binary storage for int facts (vs Python dict overhead).
    Layout: key (int64) + value (int64) per record, contiguous buffer.
    Measured vs Python dict getsizeof (v0.3 H12: Python overhead >> packed)."""

    def __init__(self):
        self._buf = bytearray()
        self._index = {}  # key -> offset (still Python, but values packed)

    def put(self, key: int, value: int):
        off = len(self._buf)
        self._buf += struct.pack("<qq", int(key), int(value))
        self._index[int(key)] = off

    def get(self, key: int):
        off = self._index.get(int(key))
        if off is None: return None
        k, v = struct.unpack_from("<qq", self._buf, off)
        return v

    def bytes_used(self) -> int:
        return len(self._buf)

    def __len__(self): return len(self._index)


def python_dict_bytes(n: int) -> int:
    """Estimate Python dict overhead for n int->int entries (measured)."""
    import sys
    d = {i: i * 2 for i in range(n)}
    return sys.getsizeof(d) + sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in d.items())


# C++/Rust feasibility assessment (documented, not implemented here)
CORE_ASSESSMENT = {
    "python_runtime": "Python is the bottleneck (dict overhead, GIL). PackedStore reduces value storage to 16 B/record.",
    "cpp_rust_feasible": "Yes: RNS/CA/graph ops are simple integer ops, easily ported to C++/Rust. No torch needed.",
    "modular": "Each module (memory/reasoning/synthesis) has a narrow interface; pluggable via Python imports now, FFI later.",
    "parallel_async": "CA/graph propagation are embarrassingly parallel; numpy already vectorizes. Rust rayon would parallelize further.",
    "offline": "All code is offline (numpy/sklearn only). No network calls in core.",
    "not_rewritten": "Whole project NOT rewritten; PackedStore is an additive prototype pending measured advantage.",
}
