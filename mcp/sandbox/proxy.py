"""
Tool proxy implementation for MCP.

This module provides proxy generators that allow sandboxed code to safely
access MCP tools without direct access to the client implementation.
"""

import asyncio
import functools
import inspect
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..models import Tool, ToolResponse, TextContent, ErrorContent

# Set up logging
logger = logging.getLogger("mcp.sandbox.proxy")


def create_tool_proxy(client, tool_name: str, schema: Optional[Dict[str, Any]] = None) -> Callable:
    """
    Create a synchronous proxy function for an MCP tool.
    
    Args:
        client: MultiServerClient instance
        tool_name: Name of the tool to proxy
        schema: Optional schema to validate arguments
        
    Returns:
        Callable function that proxies calls to the tool
    """
    async def _async_proxy(*args, **kwargs):
        """Async implementation of the proxy call."""
        try:
            # Call the tool via the client's function_wrapper
            result = await client.function_wrapper(tool_name, *args, **kwargs)
            return result
        except Exception as e:
            logger.exception(f"Error calling tool {tool_name}: {e}")
            return {"error": str(e)}
    
    def _sync_proxy(*args, **kwargs):
        """Synchronous wrapper around the async proxy."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_async_proxy(*args, **kwargs))
    
    # Set proxy metadata
    _sync_proxy.__name__ = tool_name
    _sync_proxy.__qualname__ = f"tool_proxy.{tool_name}"
    
    # If we have a schema, extract the docstring
    if schema:
        docstring = schema.get("description", f"Proxy for tool: {tool_name}")
        _sync_proxy.__doc__ = docstring
    
    return _sync_proxy


def create_async_tool_proxy(client, tool_name: str, schema: Optional[Dict[str, Any]] = None) -> Callable:
    """
    Create an async proxy function for an MCP tool.
    
    Args:
        client: MultiServerClient instance
        tool_name: Name of the tool to proxy
        schema: Optional schema to validate arguments
        
    Returns:
        Async callable function that proxies calls to the tool
    """
    async def _async_proxy(*args, **kwargs):
        """Async implementation of the proxy call."""
        try:
            # Call the tool via the client's function_wrapper
            result = await client.function_wrapper(tool_name, *args, **kwargs)
            return result
        except Exception as e:
            logger.exception(f"Error calling tool {tool_name}: {e}")
            return {"error": str(e)}
    
    # Set proxy metadata
    _async_proxy.__name__ = tool_name
    _async_proxy.__qualname__ = f"async_tool_proxy.{tool_name}"
    
    # If we have a schema, extract the docstring
    if schema:
        docstring = schema.get("description", f"Async proxy for tool: {tool_name}")
        _async_proxy.__doc__ = docstring
    
    return _async_proxy


class ToolProxy:
    """
    Tool proxy class that provides access to MCP tools.
    
    This class wraps a MultiServerClient and provides attribute-based access
    to tools. For example, `tools.search("query")` would call the "search" tool.
    """
    
    def __init__(self, client, async_mode: bool = False):
        """
        Initialize the tool proxy.
        
        Args:
            client: MultiServerClient instance
            async_mode: If True, create async proxies instead of sync proxies
        """
        self._client = client
        self._async_mode = async_mode
        self._tools: Dict[str, Callable] = {}
        self._initialized = False
    
    async def _initialize(self):
        """Initialize the proxy by fetching tools from the client."""
        if self._initialized:
            return
        
        # Get available tools from the client
        tools = self._client.get_available_tools()
        
        # Create proxies for each tool
        for tool_name, tool_info in tools.items():
            schema = tool_info.get("schema")
            if self._async_mode:
                self._tools[tool_name] = create_async_tool_proxy(self._client, tool_name, schema)
            else:
                self._tools[tool_name] = create_tool_proxy(self._client, tool_name, schema)
        
        self._initialized = True
    
    def __getattr__(self, name: str) -> Callable:
        """Get a tool proxy by attribute name."""
        if name.startswith('_'):
            return super().__getattr__(name)
        
        # Check if we need to initialize
        if not self._initialized:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self._initialize())
        
        # Check if the tool exists
        if name not in self._tools:
            raise AttributeError(f"Tool not found: {name}")
        
        return self._tools[name]
    
    def __dir__(self) -> List[str]:
        """Get available tool names for dir() and tab completion."""
        if not self._initialized:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self._initialize())
        
        return list(self._tools.keys()) + list(super().__dir__())


class AsyncToolProxy(ToolProxy):
    """Async version of the ToolProxy class."""
    
    def __init__(self, client):
        """Initialize with async_mode=True."""
        super().__init__(client, async_mode=True)
    
    async def __getattr__(self, name: str) -> Callable:
        """Get a tool proxy by attribute name (async version)."""
        if name.startswith('_'):
            return super().__getattr__(name)
        
        # Check if we need to initialize
        if not self._initialized:
            await self._initialize()
        
        # Check if the tool exists
        if name not in self._tools:
            raise AttributeError(f"Tool not found: {name}")
        
        return self._tools[name]


def create_tool_proxies(client) -> Tuple[ToolProxy, AsyncToolProxy]:
    """
    Create both sync and async tool proxies.
    
    Args:
        client: MultiServerClient instance
        
    Returns:
        Tuple of (sync_proxy, async_proxy)
    """
    return ToolProxy(client), AsyncToolProxy(client)
