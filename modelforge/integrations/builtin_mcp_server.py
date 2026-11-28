"""Reference local STDIO MCP JSON-RPC 2.0 server shipped with ModelForge for real verification."""

from __future__ import annotations

import json
import sys
from typing import Any, Dict


TOOLS = [
    {
        "name": "analyze_corpus_stats",
        "description": "Compute character, word, and Tajik/Russian Cyrillic statistics on text.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "estimate_kv_cache_mb",
        "description": "Calculate KV-cache memory in MB for given layers, heads, head_dim, and seq_len.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "n_layers": {"type": "integer"},
                "d_model": {"type": "integer"},
                "seq_len": {"type": "integer"},
            },
            "required": ["n_layers", "d_model", "seq_len"],
        },
    },
]


def _handle_request(req: Dict[str, Any]) -> Dict[str, Any] | None:
    method = req.get("method", "")
    req_id = req.get("id")
    params = req.get("params") or {}

    if method.startswith("notifications/"):
        return None

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "modelforge-builtin-mcp", "version": "1.0.0"},
                "capabilities": {"tools": {}},
            },
        }

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}

    if method == "tools/call":
        tname = params.get("name")
        args = params.get("arguments") or {}
        if tname == "analyze_corpus_stats":
            txt = str(args.get("text", ""))
            tajik_chars = sum(1 for c in txt if c in "ғӣқӯҳҷҒӢҚӮҲҶ")
            words = len(txt.split())
            out = {"chars": len(txt), "words": words, "tajik_specific_chars": tajik_chars}
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False)}]},
            }
        if tname == "estimate_kv_cache_mb":
            nl = int(args.get("n_layers", 2))
            dm = int(args.get("d_model", 64))
            sl = int(args.get("seq_len", 128))
            # 2 (K & V) * n_layers * seq_len * d_model * 4 bytes (fp32)
            mb = round((2 * nl * sl * dm * 4) / (1024 * 1024), 5)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps({"kv_cache_mb": mb})}]},
            }
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Unknown tool: {tname}"},
        }

    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unknown method: {method}"}}


def main() -> int:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = _handle_request(req)
            if resp is not None:
                sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                sys.stdout.flush()
        except Exception as exc:
            err = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}}
            sys.stdout.write(json.dumps(err) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
