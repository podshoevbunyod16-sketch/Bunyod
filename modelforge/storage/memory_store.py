"""Separated Project / Agent / Dialogue memory + local RAG indexer with source citations."""

from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from modelforge.storage.database import StorageDatabase

VALID_MEMORY_SCOPES = {"project", "agent", "dialogue"}
_WORD_RE = re.compile(r"[\w\u0400-\u04FF]+", re.UNICODE)


def _tokenize_for_rag(text: str) -> List[str]:
    """Multilingual (Russian, Tajik Cyrillic, English) word + char-trigram tokenizer for RAG."""
    words = [w.lower() for w in _WORD_RE.findall(text)]
    trigrams: List[str] = []
    compact = " ".join(words)
    if len(compact) >= 3:
        for i in range(0, min(len(compact) - 2, 600), 2):
            trigrams.append(f"#{compact[i:i+3]}")
    return words + trigrams


class MemoryAndRagStore:
    """
    Specification Section 9:
    - Strictly separate project memory, agent memory, and dialogue history.
    - Support export and deletion of memory data.
    - Local RAG with indexing and exact citations (source path + line span + sha256).
    """

    def __init__(self, db: StorageDatabase) -> None:
        self._db = db

    def add_memory(
        self,
        scope: str,
        owner_id: str,
        content: str,
        role: str = "note",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if scope not in VALID_MEMORY_SCOPES:
            raise ValueError(f"Memory scope must be one of {sorted(VALID_MEMORY_SCOPES)}, got '{scope}'.")
        mem_id = f"mem-{secrets.token_hex(6)}"
        now = time.time()
        row = {
            "id": mem_id,
            "scope": scope,
            "owner_id": owner_id,
            "role": role,
            "content": content,
            "metadata_json": json.dumps(metadata or {}, ensure_ascii=False),
            "created_at": now,
        }
        self._db.upsert_row("memories", row)
        return {
            "id": mem_id,
            "scope": scope,
            "owner_id": owner_id,
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "created_at": now,
        }

    def list_memories(self, scope: str, owner_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        if scope not in VALID_MEMORY_SCOPES:
            raise ValueError(f"Invalid memory scope '{scope}'.")
        rows = self._db.query_all(
            "SELECT * FROM memories WHERE scope = ? AND owner_id = ? ORDER BY created_at DESC LIMIT ?",
            (scope, owner_id, limit),
        )
        out = []
        for r in reversed(rows):
            out.append(
                {
                    "id": r["id"],
                    "scope": r["scope"],
                    "owner_id": r["owner_id"],
                    "role": r["role"],
                    "content": r["content"],
                    "metadata": json.loads(r["metadata_json"] or "{}"),
                    "created_at": r["created_at"],
                }
            )
        return out

    def export_memories(self, scope: str, owner_id: str) -> Dict[str, Any]:
        items = self.list_memories(scope, owner_id, limit=10_000)
        return {
            "scope": scope,
            "owner_id": owner_id,
            "exported_at": time.time(),
            "count": len(items),
            "entries": items,
        }

    def clear_memories(self, scope: str, owner_id: str) -> int:
        if scope not in VALID_MEMORY_SCOPES:
            raise ValueError(f"Invalid memory scope '{scope}'.")
        existing = self._db.query_all(
            "SELECT id FROM memories WHERE scope = ? AND owner_id = ?",
            (scope, owner_id),
        )
        self._db.execute(
            "DELETE FROM memories WHERE scope = ? AND owner_id = ?",
            (scope, owner_id),
        )
        return len(existing)

    # --- Local RAG with exact chunk citations ---
    def index_document(
        self,
        project_id: str,
        source_path: str,
        text: str,
        lines_per_chunk: int = 18,
        overlap_lines: int = 3,
    ) -> Dict[str, Any]:
        source_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self._db.execute(
            "DELETE FROM rag_chunks WHERE project_id = ? AND source_path = ?",
            (project_id, source_path),
        )
        lines = text.splitlines() or [text]
        step = max(1, lines_per_chunk - overlap_lines)
        chunks_added = 0
        now = time.time()
        for chunk_idx, start in enumerate(range(0, len(lines), step)):
            seg = lines[start : start + lines_per_chunk]
            body = "\n".join(seg).strip()
            if not body:
                continue
            tokens = _tokenize_for_rag(body)
            chunk_id = f"rag-{secrets.token_hex(6)}"
            self._db.upsert_row(
                "rag_chunks",
                {
                    "id": chunk_id,
                    "project_id": project_id,
                    "source_path": source_path,
                    "source_sha256": source_sha,
                    "chunk_index": chunk_idx,
                    "start_line": start + 1,
                    "end_line": start + len(seg),
                    "content": body,
                    "tokens_json": json.dumps(tokens, ensure_ascii=False),
                    "created_at": now,
                },
            )
            chunks_added += 1
        return {
            "project_id": project_id,
            "source_path": source_path,
            "source_sha256": source_sha,
            "chunks_indexed": chunks_added,
        }

    def index_file(self, project_id: str, file_path: Path) -> Dict[str, Any]:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        return self.index_document(project_id, str(file_path), text)

    def search_rag(self, project_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q_tokens = _tokenize_for_rag(query)
        if not q_tokens:
            return []
        rows = self._db.query_all(
            "SELECT * FROM rag_chunks WHERE project_id = ?",
            (project_id,),
        )
        if not rows:
            return []

        doc_tokens = [json.loads(r["tokens_json"] or "[]") for r in rows]
        n_docs = len(rows)
        df: Counter[str] = Counter()
        for toks in doc_tokens:
            for t in set(toks):
                df[t] += 1
        avgdl = sum(len(t) for t in doc_tokens) / max(1, n_docs)

        scored: List[tuple[float, Dict[str, Any]]] = []
        k1 = 1.5
        b = 0.75
        for row, toks in zip(rows, doc_tokens):
            tf = Counter(toks)
            dl = len(toks) or 1
            score = 0.0
            for qt in q_tokens:
                if qt not in tf:
                    continue
                idf = math.log(1.0 + (n_docs - df[qt] + 0.5) / (df[qt] + 0.5))
                freq = tf[qt]
                denom = freq + k1 * (1.0 - b + b * (dl / max(1.0, avgdl)))
                score += idf * ((freq * (k1 + 1.0)) / denom)
            if score > 0.0:
                scored.append(
                    (
                        round(score, 4),
                        {
                            "chunk_id": row["id"],
                            "score": round(score, 4),
                            "content": row["content"],
                            "citation": {
                                "source_path": row["source_path"],
                                "source_sha256": row["source_sha256"],
                                "start_line": row["start_line"],
                                "end_line": row["end_line"],
                            },
                        },
                    )
                )
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:top_k]]
