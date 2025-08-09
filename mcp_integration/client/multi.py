import asyncio
import json
from typing import Any, Dict, List

from .base import MCPClient


class MultiMCPClient:
    """
    Manage multiple SSE MCP servers and present a unified tool surface.
    """

    def __init__(self, server_configs: List[dict]):
        """
        server_configs: List of dicts with keys:
          - id: unique server id
          - url: SSE endpoint, e.g. http://localhost:8000/sse
          - description: human description (optional)
        """
        self.server_configs = server_configs
        self.tool_map: Dict[str, Dict[str, Any]] = {}
        self.server_tools: Dict[str, List[Any]] = {}
        self.client_cache: Dict[str, MCPClient] = {}

    async def initialize(self):
        for config in self.server_configs:
            try:
                client = MCPClient(server_url=config["url"])
                self.client_cache[config["id"]] = client

                tools = await client.list_tools()

                for tool in tools:
                    self.tool_map[tool.name] = {
                        "config": config,
                        "tool": tool,
                    }
                    self.server_tools.setdefault(config["id"], []).append(tool)
            except Exception as e:
                print(f"Error initializing tool server {config.get('id')}: {e}")

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        entry = self.tool_map.get(tool_name)
        if not entry:
            raise ValueError(f"Tool '{tool_name}' not found on any server.")
        cfg = entry["config"]
        client = self.client_cache[cfg["id"]]
        return await client.call_tool(tool_name, arguments)

    def get_all_tools(self) -> List[Any]:
        return [entry["tool"] for entry in self.tool_map.values()]

    async def shutdown(self):
        for client in self.client_cache.values():
            try:
                await client.shutdown()
            except asyncio.CancelledError:
                break
            except Exception:
                pass
        self.client_cache.clear()
