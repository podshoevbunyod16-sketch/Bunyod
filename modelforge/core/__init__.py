"""ModelForge Core: configuration, modes, events, state, projects, tasks & central runtime."""

from modelforge.core.config import ExecutionMode, ModelForgeConfig
from modelforge.core.events import Event, EventBus
from modelforge.core.projects import Project, ProjectManager
from modelforge.core.runtime import ModelForgeCore, get_core
from modelforge.core.state import SystemStateManager
from modelforge.core.task_manager import TaskManager, TaskRecord

__all__ = [
    "Event",
    "EventBus",
    "ExecutionMode",
    "ModelForgeConfig",
    "ModelForgeCore",
    "Project",
    "ProjectManager",
    "SystemStateManager",
    "TaskManager",
    "TaskRecord",
    "get_core",
]
