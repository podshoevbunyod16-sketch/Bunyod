"""Real GGUF v3 binary serializer & inspector + optional llama.cpp CLI adapter (Section 8 & 9)."""

from __future__ import annotations

import hashlib
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from modelforge.training.checkpoints import load_checkpoint

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
# GGUF metadata value types
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8
# GGML tensor type F32 = 0
GGML_TYPE_F32 = 0
GGUF_ALIGNMENT = 32


def _pack_string(s: str) -> bytes:
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _unpack_string(buf: bytes, offset: int) -> tuple[str, int]:
    (length,) = struct.unpack_from("<Q", buf, offset)
    offset += 8
    val = buf[offset : offset + length].decode("utf-8", errors="replace")
    return val, offset + length


def _align_offset(offset: int, alignment: int = GGUF_ALIGNMENT) -> int:
    rem = offset % alignment
    return offset if rem == 0 else offset + (alignment - rem)


def export_gguf_v3(checkpoint_path: str | Path, output_file: Path) -> Dict[str, Any]:
    """Serialize a trained ModelForge checkpoint into a spec-compliant GGUF v3 binary file."""
    loaded = load_checkpoint(checkpoint_path)
    spec = loaded["manifest"]["spec"]
    weights: Dict[str, np.ndarray] = loaded["weights"]

    kv_pairs: List[tuple[str, int, Any]] = [
        ("general.architecture", GGUF_TYPE_STRING, str(spec.get("architecture", "modelforge"))),
        ("general.name", GGUF_TYPE_STRING, str(spec.get("name", "modelforge-model"))),
        ("modelforge.context_length", GGUF_TYPE_UINT32, int(spec.get("context_length", 128))),
        ("modelforge.embedding_length", GGUF_TYPE_UINT32, int(spec.get("d_model", 64))),
        ("modelforge.block_count", GGUF_TYPE_UINT32, int(spec.get("n_layers", 2))),
        ("modelforge.vocab_size", GGUF_TYPE_UINT32, int(spec.get("vocab_size", 320))),
    ]

    out = bytearray()
    out.extend(GGUF_MAGIC)
    out.extend(struct.pack("<I", GGUF_VERSION))
    out.extend(struct.pack("<Q", len(weights)))
    out.extend(struct.pack("<Q", len(kv_pairs)))

    for k, vtype, val in kv_pairs:
        out.extend(_pack_string(k))
        out.extend(struct.pack("<I", vtype))
        if vtype == GGUF_TYPE_STRING:
            out.extend(_pack_string(str(val)))
        elif vtype == GGUF_TYPE_UINT32:
            out.extend(struct.pack("<I", int(val)))
        elif vtype == GGUF_TYPE_FLOAT32:
            out.extend(struct.pack("<f", float(val)))

    # Prepare contiguous F32 tensor payloads & relative offsets
    tensor_names = sorted(weights.keys())
    tensor_blobs: List[bytes] = []
    rel_offset = 0
    for name in tensor_names:
        arr = np.ascontiguousarray(weights[name], dtype=np.float32)
        dims = arr.shape
        out.extend(_pack_string(name))
        out.extend(struct.pack("<I", len(dims)))
        for d in reversed(dims):
            out.extend(struct.pack("<Q", int(d)))
        out.extend(struct.pack("<I", GGML_TYPE_F32))
        out.extend(struct.pack("<Q", rel_offset))
        blob = arr.tobytes()
        padded_len = _align_offset(len(blob), GGUF_ALIGNMENT)
        blob_padded = blob + (b"\x00" * (padded_len - len(blob)))
        tensor_blobs.append(blob_padded)
        rel_offset += padded_len

    aligned_data_start = _align_offset(len(out), GGUF_ALIGNMENT)
    out.extend(b"\x00" * (aligned_data_start - len(out)))
    for blob in tensor_blobs:
        out.extend(blob)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    raw_bytes = bytes(out)
    output_file.write_bytes(raw_bytes)
    return {
        "format": "gguf_v3",
        "path": str(output_file),
        "size_bytes": len(raw_bytes),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "tensor_count": len(weights),
        "metadata_count": len(kv_pairs),
    }


def inspect_gguf_v3(gguf_file: Path) -> Dict[str, Any]:
    """Read and validate a GGUF v2/v3 file header, metadata KVs, and tensor directory."""
    buf = gguf_file.read_bytes()
    if len(buf) < 24 or buf[:4] != GGUF_MAGIC:
        raise ValueError(f"File '{gguf_file}' is not a valid GGUF file (bad magic).")
    (version, tensor_count, kv_count) = struct.unpack_from("<IQQ", buf, 4)
    offset = 24
    metadata: Dict[str, Any] = {}
    for _ in range(kv_count):
        key, offset = _unpack_string(buf, offset)
        (vtype,) = struct.unpack_from("<I", buf, offset)
        offset += 4
        if vtype == GGUF_TYPE_STRING:
            val, offset = _unpack_string(buf, offset)
        elif vtype == GGUF_TYPE_UINT32:
            (val,) = struct.unpack_from("<I", buf, offset)
            offset += 4
        elif vtype == GGUF_TYPE_FLOAT32:
            (val,) = struct.unpack_from("<f", buf, offset)
            offset += 4
        else:
            break
        metadata[key] = val

    tensors: List[Dict[str, Any]] = []
    for _ in range(tensor_count):
        if offset >= len(buf):
            break
        tname, offset = _unpack_string(buf, offset)
        (n_dims,) = struct.unpack_from("<I", buf, offset)
        offset += 4
        dims = []
        for _ in range(n_dims):
            (d,) = struct.unpack_from("<Q", buf, offset)
            offset += 8
            dims.append(d)
        (ggml_type, rel_off) = struct.unpack_from("<IQ", buf, offset)
        offset += 12
        tensors.append({"name": tname, "shape": list(reversed(dims)), "ggml_type": ggml_type, "offset": rel_off})

    return {
        "valid_gguf": True,
        "version": version,
        "tensor_count": tensor_count,
        "kv_count": kv_count,
        "metadata": metadata,
        "tensors": tensors,
    }


def run_llama_cpp_gguf(gguf_path: Path, prompt: str, max_tokens: int = 64, temperature: float = 0.7) -> Dict[str, Any]:
    """Adapter for external `llama-cli` binary when installed (ARM64 Termux / Desktop CPU)."""
    binary = shutil.which("llama-cli")
    if not binary:
        raise RuntimeError(
            "llama.cpp CLI (`llama-cli`) is not installed in PATH. "
            "Install `llama.cpp` or use native ModelForge checkpoint inference."
        )
    completed = subprocess.run(
        [
            binary,
            "-m",
            str(gguf_path),
            "-p",
            prompt,
            "-n",
            str(max_tokens),
            "--temp",
            str(temperature),
            "--no-display-prompt",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"llama-cli failed: {completed.stderr.strip()}")
    return {"text": completed.stdout.strip(), "engine": "llama.cpp"}
