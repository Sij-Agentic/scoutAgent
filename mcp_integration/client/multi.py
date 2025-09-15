import asyncio
import json
import logging
from typing import Any, Dict, List

from .base import MCPClient

logger = logging.getLogger(__name__)


class MultiMCPClient:
    """
    Manage multiple SSE MCP servers and present a unified tool surface with enhanced resilience.
    """

    def __init__(self, server_configs: List[dict], max_retries: int = 3, connection_timeout: int = 30, sse_read_timeout: int = 300):
        """
        server_configs: List of dicts with keys:
          - id: unique server id
          - url: SSE endpoint, e.g. http://localhost:8000/sse
          - description: human description (optional)
        max_retries: Number of retry attempts for connections
        connection_timeout: Timeout for individual connection attempts
        sse_read_timeout: SSE read timeout for long operations (default 300s, use 3600s for vendor research)
        """
        self.server_configs = server_configs
        self.max_retries = max_retries
        self.connection_timeout = connection_timeout
        self.sse_read_timeout = sse_read_timeout
        self.tool_map: Dict[str, Dict[str, Any]] = {}
        self.server_tools: Dict[str, List[Any]] = {}
        # Do not keep persistent clients created from a different loop.
        # We'll create short-lived clients on demand in the active loop.
        self.client_cache: Dict[str, MCPClient] = {}

    async def initialize(self):
        """Initialize connections to all MCP servers with enhanced error handling."""
        initialization_errors = []
        
        for config in self.server_configs:
            server_id = config.get('id', 'unknown')
            server_url = config.get('url', 'unknown')
            
            try:
                logger.info(f"Initializing MCP server {server_id} at {server_url}")
                
                # Use enhanced client with retry logic
                client = MCPClient(
                    server_url=server_url,
                    max_retries=self.max_retries,
                    connection_timeout=self.connection_timeout,
                    sse_read_timeout=self.sse_read_timeout
                )
                
                tools = await client.list_tools()
                await client.shutdown()

                tool_count = len(tools)
                logger.info(f"Successfully initialized server {server_id} with {tool_count} tools")
                
                for tool in tools:
                    self.tool_map[tool.name] = {
                        "config": config,
                        "tool": tool,
                    }
                    self.server_tools.setdefault(config["id"], []).append(tool)
                    
            except Exception as e:
                error_msg = f"Error initializing tool server {server_id} at {server_url}: {e}"
                logger.error(error_msg)
                initialization_errors.append(error_msg)
                # Continue with other servers instead of failing completely
                
        if initialization_errors:
            logger.warning(f"Some MCP servers failed to initialize: {len(initialization_errors)} errors")
            # Log all errors but don't fail if at least one server is working
            if not self.tool_map:
                raise ConnectionError(f"All MCP servers failed to initialize. Errors: {'; '.join(initialization_errors)}")
        
        total_tools = len(self.tool_map)
        total_servers = len([s for s in self.server_configs if s.get('id') in self.server_tools])
        logger.info(f"MultiMCP initialization complete: {total_tools} tools from {total_servers} servers")

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool with enhanced error handling and retry logic."""
        entry = self.tool_map.get(tool_name)
        if not entry:
            available_tools = list(self.tool_map.keys())
            raise ValueError(f"Tool '{tool_name}' not found on any server. Available tools: {available_tools}")
            
        cfg = entry["config"]
        server_id = cfg.get('id', 'unknown')
        server_url = cfg.get('url', 'unknown')
        
        logger.debug(f"Calling tool {tool_name} on server {server_id}")
        
        # Create enhanced client with retry logic
        client = MCPClient(
            server_url=cfg["url"],
            max_retries=self.max_retries,
            connection_timeout=self.connection_timeout,
            sse_read_timeout=self.sse_read_timeout
        )
        
        try:
            result = await client.call_tool(tool_name, arguments)
            logger.debug(f"Successfully called tool {tool_name} on server {server_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to call tool {tool_name} on server {server_id} ({server_url}): {e}")
            raise
        finally:
            await client.shutdown()

    def get_all_tools(self) -> List[Any]:
        return [entry["tool"] for entry in self.tool_map.values()]

    async def shutdown(self):
        # No persistent clients by default; best-effort cleanup if any exist.
        for client in list(self.client_cache.values()):
            try:
                await client.shutdown()
            except asyncio.CancelledError:
                break
            except Exception:
                pass
        self.client_cache.clear()
