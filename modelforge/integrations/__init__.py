"""External & local integrations: LLM Providers, MCP Manager (STDIO & Streamable HTTP), Plugins, GitHub & Composio."""

from modelforge.integrations.external import ComposioAdapter, GitHubIntegration
from modelforge.integrations.llm_providers import LLMProviderManager
from modelforge.integrations.mcp import MCPManager
from modelforge.integrations.plugins import PluginManager

__all__ = [
    "ComposioAdapter",
    "GitHubIntegration",
    "LLMProviderManager",
    "MCPManager",
    "PluginManager",
]
