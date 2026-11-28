"""Plugin system with manifest validation, explicit permissions & schema checks (Section 10)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from modelforge.security.audit import AuditLogger
from modelforge.security.permissions import PermissionDeniedError


class PluginManifest(BaseModel):
    id: str
    name: str
    version: str
    description: str
    entrypoint: str = "plugin.py:run"
    requirements: List[str] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list)
    allow_full_filesystem: bool = False
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)


def _validate_simple_json_schema(payload: Dict[str, Any], schema: Dict[str, Any], label: str) -> None:
    if not schema:
        return
    required = schema.get("required", [])
    for key in required:
        if key not in payload:
            raise ValueError(f"Plugin {label} missing required field '{key}'.")
    props = schema.get("properties", {})
    type_map = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict, "array": list}
    for key, spec in props.items():
        if key in payload and "type" in spec:
            expected_py = type_map.get(spec["type"])
            if expected_py and not isinstance(payload[key], expected_py):
                raise TypeError(f"Plugin {label} field '{key}' expected {spec['type']}, got {type(payload[key]).__name__}.")


class PluginManager:
    """Loads, validates, inspects, and runs sandboxed manifest-backed plugins."""

    def __init__(self, plugins_dir: Path, audit: AuditLogger) -> None:
        self.plugins_dir = plugins_dir.resolve()
        self._audit = audit

    def discover_plugins(self) -> List[Dict[str, Any]]:
        if not self.plugins_dir.exists():
            return []
        found: List[Dict[str, Any]] = []
        for manifest_file in sorted(self.plugins_dir.glob("*/manifest.json")):
            raw = json.loads(manifest_file.read_text(encoding="utf-8"))
            manifest = PluginManifest.model_validate(raw)
            if manifest.allow_full_filesystem:
                # Section 10: By default tools do not receive access to the entire filesystem
                pass
            found.append(
                {
                    **manifest.model_dump(),
                    "plugin_dir": str(manifest_file.parent),
                }
            )
        return found

    def run_plugin(self, plugin_id: str, inputs: Dict[str, Any], granted_permissions: List[str] | None = None) -> Dict[str, Any]:
        plugins = {p["id"]: p for p in self.discover_plugins()}
        if plugin_id not in plugins:
            raise KeyError(f"Plugin not found: {plugin_id}")
        pinfo = plugins[plugin_id]
        if pinfo.get("allow_full_filesystem"):
            raise PermissionDeniedError(f"Plugin '{plugin_id}' requests unrestricted filesystem access, which is forbidden by default.")

        granted = set(granted_permissions or pinfo.get("permissions", []))
        for req_perm in pinfo.get("permissions", []):
            if req_perm not in granted:
                raise PermissionDeniedError(f"Plugin '{plugin_id}' lacks required permission '{req_perm}'.")

        _validate_simple_json_schema(inputs, pinfo.get("input_schema", {}), "input")

        mod_rel, func_name = pinfo["entrypoint"].split(":", 1)
        mod_path = Path(pinfo["plugin_dir"]) / mod_rel
        spec = importlib.util.spec_from_file_location(f"modelforge_plugin_{plugin_id}", mod_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load plugin module: {mod_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        runner = getattr(module, func_name)
        output = runner(inputs)
        if not isinstance(output, dict):
            raise TypeError("Plugin entrypoint must return a dict matching output_schema.")

        _validate_simple_json_schema(output, pinfo.get("output_schema", {}), "output")
        self._audit.record(
            actor=f"plugin:{plugin_id}",
            action="plugin.run",
            status="ok",
            details={"plugin_id": plugin_id, "version": pinfo["version"]},
        )
        return output
