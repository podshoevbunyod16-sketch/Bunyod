"""Platform detection, hardware telemetry, Low Memory Mode & runtime delivery adapters."""

from modelforge.platforms.detector import (
    PlatformInfo,
    ResourceSnapshot,
    detect_platform,
    get_resource_snapshot,
)

__all__ = [
    "PlatformInfo",
    "ResourceSnapshot",
    "detect_platform",
    "get_resource_snapshot",
]
