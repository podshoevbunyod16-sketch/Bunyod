"""Configuration & operating mode manager (Local, Remote, Hybrid, Offline)."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field


class ExecutionMode(str, Enum):
    LOCAL = "local"
    REMOTE = "remote"
    HYBRID = "hybrid"
    OFFLINE = "offline"


class ResourceLimitsConfig(BaseModel):
    max_cpu_threads: int = Field(default=2, ge=1, le=128)
    max_ram_mb: int = Field(default=2048, ge=256)
    max_disk_mb: int = Field(default=8192, ge=128)
    max_task_seconds: int = Field(default=600, ge=5)
    max_agent_steps: int = Field(default=15, ge=1, le=200)
    max_agent_tokens: int = Field(default=16_000, ge=256)
    max_prompt_edit_retries: int = Field(default=2, ge=0, le=5)


class ModelForgeConfig(BaseModel):
    home_dir: str
    mode: ExecutionMode = ExecutionMode.LOCAL
    remote_server_url: str = ""
    remote_timeout_sec: float = 10.0
    locale: str = "ru"
    low_memory_mode: bool = False
    api_host: str = "0.0.0.0"
    api_port: int = 8765
    limits: ResourceLimitsConfig = Field(default_factory=ResourceLimitsConfig)

    @classmethod
    def load_or_create(
        cls,
        home_dir: Optional[Path] = None,
        low_memory_default: bool = False,
    ) -> "ModelForgeConfig":
        resolved_home = (
            home_dir
            or Path(os.environ.get("MODELFORGE_HOME", str(Path.home() / ".modelforge")))
        ).resolve()
        resolved_home.mkdir(parents=True, exist_ok=True)
        cfg_file = resolved_home / "config.yaml"

        env_mode = os.environ.get("MODELFORGE_MODE", "").strip().lower()
        if cfg_file.exists():
            raw = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
            raw["home_dir"] = str(resolved_home)
            cfg = cls.model_validate(raw)
        else:
            limits = ResourceLimitsConfig(
                max_ram_mb=900 if low_memory_default else 2048,
                max_cpu_threads=1 if low_memory_default else 2,
                max_task_seconds=180 if low_memory_default else 600,
            )
            cfg = cls(
                home_dir=str(resolved_home),
                low_memory_mode=low_memory_default,
                limits=limits,
            )
            cfg.save()

        if env_mode in {m.value for m in ExecutionMode}:
            cfg.mode = ExecutionMode(env_mode)
        if os.environ.get("MODELFORGE_OFFLINE", "").strip() == "1":
            cfg.mode = ExecutionMode.OFFLINE
        return cfg

    def save(self) -> Path:
        cfg_file = Path(self.home_dir) / "config.yaml"
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json")
        cfg_file.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return cfg_file

    @property
    def is_offline(self) -> bool:
        return self.mode == ExecutionMode.OFFLINE

    def to_public_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")
