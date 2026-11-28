"""Cross-platform environment, ABI, Termux/Android & Low Memory Mode detector."""

from __future__ import annotations

import os
import platform as std_platform
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class PlatformInfo:
    os_name: str  # linux | windows | macos | android
    is_termux: bool
    is_android: bool
    architecture: str  # x86_64 | arm64 | x86 | other
    raw_machine: str
    python_version: str
    cpu_count: int
    total_ram_mb: int
    available_ram_mb: int
    free_disk_mb: int
    low_memory_mode: bool
    recommended_max_params: int
    recommended_max_batch_size: int
    recommended_max_seq_len: int
    backend_delivery_mode: str  # bundled_service | termux_daemon | local_python | remote_only
    optional_packages: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResourceSnapshot:
    cpu_percent: float
    total_ram_mb: int
    used_ram_mb: int
    available_ram_mb: int
    process_rss_mb: float
    free_disk_mb: int
    low_memory_pressure: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_arch(machine: str) -> str:
    m = machine.lower().strip()
    if m in ("x86_64", "amd64"):
        return "x86_64"
    if m in ("aarch64", "arm64", "armv8l"):
        return "arm64"
    if m in ("i386", "i686", "x86"):
        return "x86"
    return m or "unknown"


def _read_ram_fallback() -> tuple[int, int]:
    """Fallback RAM reader via /proc/meminfo or conservative defaults."""
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        try:
            total_kb = 0
            avail_kb = 0
            for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    avail_kb = int(line.split()[1])
            if total_kb > 0:
                if avail_kb == 0:
                    avail_kb = total_kb // 2
                return total_kb // 1024, avail_kb // 1024
        except Exception:
            pass
    return 4096, 2048


def _check_optional_packages() -> Dict[str, bool]:
    pkgs: Dict[str, bool] = {}
    for mod_name, label in [
        ("numpy", "numpy"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("httpx", "httpx"),
        ("websockets", "websockets"),
        ("psutil", "psutil"),
        ("safetensors", "safetensors"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("litellm", "litellm"),
    ]:
        try:
            __import__(mod_name)
            pkgs[label] = True
        except Exception:
            pkgs[label] = False
    pkgs["llama_cpp_cli"] = shutil.which("llama-cli") is not None or shutil.which("llama-server") is not None
    return pkgs


def detect_platform(target_path: Optional[Path] = None, force_low_memory: Optional[bool] = None) -> PlatformInfo:
    sys_name = std_platform.system().lower()
    raw_machine = std_platform.machine()
    arch = _normalize_arch(raw_machine)

    prefix = os.environ.get("PREFIX", "")
    is_termux = "com.termux" in prefix or Path("/data/data/com.termux").exists()
    is_android = (
        is_termux
        or "ANDROID_ROOT" in os.environ
        or "ANDROID_DATA" in os.environ
        or sys_name == "android"
    )

    if is_android:
        os_name = "android"
    elif sys_name.startswith("win"):
        os_name = "windows"
    elif sys_name == "darwin":
        os_name = "macos"
    else:
        os_name = "linux"

    total_ram_mb, avail_ram_mb = _read_ram_fallback()
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        total_ram_mb = int(vm.total // (1024 * 1024))
        avail_ram_mb = int(vm.available // (1024 * 1024))
    except Exception:
        pass

    probe_dir = target_path or Path.cwd()
    try:
        disk = shutil.disk_usage(probe_dir)
        free_disk_mb = int(disk.free // (1024 * 1024))
    except Exception:
        free_disk_mb = 1024

    cpu_count = os.cpu_count() or 1

    # Specification Section 4:
    # On Android with <= 3.5 GB RAM (3 GB class devices) or low total RAM, enable Low Memory Mode by default.
    env_low_mem = os.environ.get("MODELFORGE_LOW_MEMORY", "").strip().lower()
    if force_low_memory is not None:
        low_memory_mode = force_low_memory
    elif env_low_mem in ("1", "true", "yes", "on"):
        low_memory_mode = True
    elif env_low_mem in ("0", "false", "no", "off"):
        low_memory_mode = False
    else:
        low_memory_mode = is_android or total_ram_mb <= 3600

    if low_memory_mode:
        recommended_max_params = 2_500_000
        recommended_max_batch_size = 4
        recommended_max_seq_len = 128
    elif total_ram_mb <= 8192:
        recommended_max_params = 15_000_000
        recommended_max_batch_size = 16
        recommended_max_seq_len = 256
    else:
        recommended_max_params = 60_000_000
        recommended_max_batch_size = 32
        recommended_max_seq_len = 512

    if getattr(sys, "frozen", False):
        delivery_mode = "bundled_service"
    elif is_termux:
        delivery_mode = "termux_daemon"
    elif is_android:
        delivery_mode = "remote_only"
    else:
        delivery_mode = "local_python"

    return PlatformInfo(
        os_name=os_name,
        is_termux=is_termux,
        is_android=is_android,
        architecture=arch,
        raw_machine=raw_machine,
        python_version=std_platform.python_version(),
        cpu_count=cpu_count,
        total_ram_mb=total_ram_mb,
        available_ram_mb=avail_ram_mb,
        free_disk_mb=free_disk_mb,
        low_memory_mode=low_memory_mode,
        recommended_max_params=recommended_max_params,
        recommended_max_batch_size=recommended_max_batch_size,
        recommended_max_seq_len=recommended_max_seq_len,
        backend_delivery_mode=delivery_mode,
        optional_packages=_check_optional_packages(),
    )


def get_resource_snapshot(target_path: Optional[Path] = None) -> ResourceSnapshot:
    total_ram_mb, avail_ram_mb = _read_ram_fallback()
    used_ram_mb = max(0, total_ram_mb - avail_ram_mb)
    cpu_percent = 0.0
    process_rss_mb = 0.0

    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        total_ram_mb = int(vm.total // (1024 * 1024))
        avail_ram_mb = int(vm.available // (1024 * 1024))
        used_ram_mb = int(vm.used // (1024 * 1024))
        cpu_percent = float(psutil.cpu_percent(interval=None))
        proc = psutil.Process(os.getpid())
        process_rss_mb = round(proc.memory_info().rss / (1024 * 1024), 2)
    except Exception:
        pass

    probe_dir = target_path or Path.cwd()
    try:
        free_disk_mb = int(shutil.disk_usage(probe_dir).free // (1024 * 1024))
    except Exception:
        free_disk_mb = 1024

    low_memory_pressure = avail_ram_mb < 350 or (total_ram_mb > 0 and (avail_ram_mb / total_ram_mb) < 0.12)
    return ResourceSnapshot(
        cpu_percent=cpu_percent,
        total_ram_mb=total_ram_mb,
        used_ram_mb=used_ram_mb,
        available_ram_mb=avail_ram_mb,
        process_rss_mb=process_rss_mb,
        free_disk_mb=free_disk_mb,
        low_memory_pressure=low_memory_pressure,
    )
