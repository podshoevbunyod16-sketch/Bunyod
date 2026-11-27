"""Real MCP Manager supporting STDIO & Streamable HTTP JSON-RPC 2.0, permissions & call audit (Section 10)."""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from modelforge.security.audit import AuditLogger
from modelforge.security.permissions import PermissionDeniedError
from modelforge.security.sandbox import sanitize_untrusted_content
from modelforge.storage.database import StorageDatabase


class MCPManager:
    """
    Specification Section 10 & 16:
    - STDIO and Streamable HTTP transports.
    - Add, probe/verify, enable/disable servers, list tools, assign to agents, enforce permissions,
      sanitize untrusted MCP responses, and log every call in `mcp_call_logs`.
    """

    def __init__(
        self,
        db: StorageDatabase,
        audit: AuditLogger,
        offline_getter: Callable[[], bool] = lambda: False,
    ) -> None:
        self._db = db
        self._audit = audit
        self._offline_getter = offline_getter
        self._ensure_builtin_server()

    def _ensure_builtin_server(self) -> None:
        existing = self._db.query_one("SELECT id FROM mcp_servers WHERE id = 'mcp-builtin-local'")
        if not existing:
            now = time.time()
            self._db.upsert_row(
                "mcp_servers",
                {
                    "id": "mcp-builtin-local",
                    "name": "ModelForge Built-in Corpus & Memory Tools",
                    "transport": "stdio",
                    "endpoint_or_cmd": sys.executable,
                    "args_json": json.dumps(["-m", "modelforge.integrations.builtin_mcp_server"]),
                    "enabled": 1,
                    "permissions_json": json.dumps(
                        {"allow_filesystem": False, "allow_network": False, "allowed_tools": ["*"]}
                    ),
                    "assigned_agents_json": json.dumps(["*"]),
                    "discovered_tools_json": json.dumps([]),
                    "created_at": now,
                    "updated_at": now,
                },
            )

    def add_server(
        self,
        name: str,
        transport: str,  # stdio | streamable_http
        endpoint_or_cmd: str,
        args: Optional[List[str]] = None,
        allowed_tools: Optional[List[str]] = None,
        assigned_agents: Optional[List[str]] = None,
        allow_filesystem: bool = False,
        allow_network: bool = False,
    ) -> Dict[str, Any]:
        tr = transport.strip().lower()
        if tr not in ("stdio", "streamable_http"):
            raise ValueError("MCP transport must be 'stdio' or 'streamable_http'.")
        sid = f"mcp-{secrets.token_hex(4)}"
        now = time.time()
        perms = {
            "allow_filesystem": allow_filesystem,  # False by default per Section 10!
            "allow_network": allow_network,
            "allowed_tools": allowed_tools or ["*"],
        }
        self._db.upsert_row(
            "mcp_servers",
            {
                "id": sid,
                "name": name,
                "transport": tr,
                "endpoint_or_cmd": endpoint_or_cmd,
                "args_json": json.dumps(args or [], ensure_ascii=False),
                "enabled": 1,
                "permissions_json": json.dumps(perms, ensure_ascii=False),
                "assigned_agents_json": json.dumps(assigned_agents or [], ensure_ascii=False),
                "discovered_tools_json": json.dumps([], ensure_ascii=False),
                "created_at": now,
                "updated_at": now,
            },
        )
        return self.get_server(sid)

    def set_enabled(self, server_id: str, enabled: bool) -> Dict[str, Any]:
        self.get_server(server_id)
        self._db.execute(
            "UPDATE mcp_servers SET enabled = ?, updated_at = ? WHERE id = ?",
            (int(enabled), time.time(), server_id),
        )
        return self.get_server(server_id)

    def assign_to_agent(self, server_id: str, agent_id: str) -> Dict[str, Any]:
        srv = self.get_server(server_id)
        assigned = list(srv["assigned_agents"])
        if agent_id not in assigned:
            assigned.append(agent_id)
        self._db.execute(
            "UPDATE mcp_servers SET assigned_agents_json = ?, updated_at = ? WHERE id = ?",
            (json.dumps(assigned), time.time(), server_id),
        )
        return self.get_server(server_id)

    def get_server(self, server_id: str) -> Dict[str, Any]:
        row = self._db.query_one("SELECT * FROM mcp_servers WHERE id = ?", (server_id,))
        if not row:
            raise KeyError(f"MCP Server not found: {server_id}")
        return {
            "id": row["id"],
            "name": row["name"],
            "transport": row["transport"],
            "endpoint_or_cmd": row["endpoint_or_cmd"],
            "args": json.loads(row["args_json"] or "[]"),
            "enabled": bool(row["enabled"]),
            "permissions": json.loads(row["permissions_json"] or "{}"),
            "assigned_agents": json.loads(row["assigned_agents_json"] or "[]"),
            "discovered_tools": json.loads(row["discovered_tools_json"] or "[]"),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_servers(self) -> List[Dict[str, Any]]:
        rows = self._db.query_all("SELECT id FROM mcp_servers ORDER BY created_at ASC")
        return [self.get_server(r["id"]) for r in rows]

    def _stdio_rpc(self, cmd: str, args: List[str], requests: List[Dict[str, Any]], timeout: float = 12.0) -> List[Dict[str, Any]]:
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
            "PYTHONUNBUFFERED": "1",
        }
        stdin_payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in requests) + "\n"
        proc = subprocess.run(
            [cmd, *args],
            input=stdin_payload,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
        )
        if proc.returncode != 0 and not proc.stdout.strip():
            raise RuntimeError(f"MCP STDIO process failed ({proc.returncode}): {proc.stderr.strip()}")
        responses: List[Dict[str, Any]] = []
        for ln in proc.stdout.splitlines():
            ln = ln.strip()
            if ln.startswith("{"):
                responses.append(json.loads(ln))
        return responses

    def _http_rpc(self, endpoint: str, request_obj: Dict[str, Any], timeout: float = 12.0) -> Dict[str, Any]:
        if self._offline_getter():
            raise PermissionDeniedError("Streamable HTTP MCP calls are blocked in Offline Mode.")
        import httpx

        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                endpoint,
                headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
                json=request_obj,
            )
        if resp.status_code >= 400:
            raise RuntimeError(f"MCP HTTP server returned {resp.status_code}: {resp.text[:300]}")
        ctype = resp.headers.get("content-type", "")
        if "text/event-stream" in ctype:
            for line in resp.text.splitlines():
                if line.startswith("data:"):
                    return json.loads(line[5:].strip())
        return resp.json()

    def probe_and_discover_tools(self, server_id: str) -> Dict[str, Any]:
        srv = self.get_server(server_id)
        if not srv["enabled"]:
            raise PermissionDeniedError(f"MCP server '{srv['name']}' is disabled.")

        init_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "modelforge-ai", "version": "1.0.0"},
            },
        }
        notif = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        list_req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}

        if srv["transport"] == "stdio":
            resps = self._stdio_rpc(srv["endpoint_or_cmd"], srv["args"], [init_req, notif, list_req])
            tools_resp = next((r for r in resps if r.get("id") == 2), {})
        else:
            self._http_rpc(srv["endpoint_or_cmd"], init_req)
            tools_resp = self._http_rpc(srv["endpoint_or_cmd"], list_req)

        tools = (tools_resp.get("result") or {}).get("tools", [])
        self._db.execute(
            "UPDATE mcp_servers SET discovered_tools_json = ?, updated_at = ? WHERE id = ?",
            (json.dumps(tools, ensure_ascii=False), time.time(), server_id),
        )
        return {"server_id": server_id, "ok": True, "tools": tools}

    def call_tool(
        self,
        server_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        agent_id: str = "user",
    ) -> Dict[str, Any]:
        srv = self.get_server(server_id)
        if not srv["enabled"]:
            raise PermissionDeniedError(f"MCP server '{srv['name']}' is disabled.")

        assigned = srv["assigned_agents"]
        if agent_id != "user" and "*" not in assigned and agent_id not in assigned:
            raise PermissionDeniedError(
                f"Agent '{agent_id}' is not authorized to use MCP server '{srv['name']}' ({server_id})."
            )

        allowed_tools = srv["permissions"].get("allowed_tools", ["*"])
        if "*" not in allowed_tools and tool_name not in allowed_tools:
            raise PermissionDeniedError(
                f"Tool '{tool_name}' is not permitted by MCP server '{srv['name']}' policy."
            )

        t0 = time.monotonic()
        init_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "modelforge-ai", "version": "1.0.0"},
            },
        }
        call_req = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }

        status = "ok"
        try:
            if srv["transport"] == "stdio":
                resps = self._stdio_rpc(srv["endpoint_or_cmd"], srv["args"], [init_req, call_req])
                call_resp = next((r for r in resps if r.get("id") == 2), {})
            else:
                call_resp = self._http_rpc(srv["endpoint_or_cmd"], call_req)

            if "error" in call_resp:
                status = "error"
                raw_text = json.dumps(call_resp["error"], ensure_ascii=False)
            else:
                raw_text = json.dumps(call_resp.get("result", {}), ensure_ascii=False)
        except Exception as exc:
            status = "error"
            raw_text = f"{type(exc).__name__}: {exc}"

        dur_ms = round((time.monotonic() - t0) * 1000.0, 2)
        # Sanitize external MCP output so it is never treated as trusted system instructions (Section 16)
        sanitized = sanitize_untrusted_content(f"mcp://{server_id}/{tool_name}", raw_text)

        log_id = f"mcplog-{secrets.token_hex(5)}"
        self._db.upsert_row(
            "mcp_call_logs",
            {
                "id": log_id,
                "server_id": server_id,
                "agent_id": agent_id,
                "tool_name": tool_name,
                "status": status,
                "duration_ms": dur_ms,
                "request_summary": json.dumps(arguments, ensure_ascii=False)[:500],
                "response_summary": sanitized["content"][:500],
                "created_at": time.time(),
            },
        )
        self._audit.record(
            actor=agent_id,
            action=f"mcp.call:{server_id}/{tool_name}",
            status=status,
            details={"duration_ms": dur_ms},
        )
        return {
            "log_id": log_id,
            "server_id": server_id,
            "tool_name": tool_name,
            "status": status,
            "duration_ms": dur_ms,
            "result": sanitized,
        }

    def list_call_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._db.query_all(
            "SELECT * FROM mcp_call_logs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
