from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Callable

from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent, Tool as MCPTypedTool

from ..config import load_server_configs
from ..client.multi import MultiMCPClient

# Unified Multi-MCP aggregator server
mcp = FastMCP("multi-tools")
_multi: MultiMCPClient | None = None


def _register_proxy_tool(tool_name: str, description: str) -> None:
    """Dynamically register a proxy tool that forwards to the underlying server tool.

    The proxy takes arbitrary keyword arguments and forwards them as-is to the
    corresponding tool on the appropriate backend server via MultiMCPClient.
    """

    async def proxy(**kwargs: Any) -> Dict[str, Any]:
        assert _multi is not None
        # Back-compat: inspector will send {"kwargs": {...}} when our function has a single
        # parameter named 'kwargs'. Unwrap if present so backends receive proper args.
        arguments = kwargs.get("kwargs", kwargs)
        # Forward the call and pass through content if available
        result = await _multi.call_tool(tool_name, arguments)
        try:
            content = getattr(result, "content", None)
            if content is not None:
                return {"content": content}
            # Fallback: wrap JSON
            return {"content": [TextContent(type="text", text=json.dumps(result))]}
        except Exception:
            return {"content": [TextContent(type="text", text=str(result))]}

    # Make tool name and doc visible to FastMCP
    proxy.__name__ = tool_name
    proxy.__doc__ = description or f"Proxy tool forwarding to '{tool_name}'"

    # Register with FastMCP
    decorator = mcp.tool()
    decorator(proxy)


def _register_typed_proxy(tool: MCPTypedTool) -> None:
    """Register a proxy with a dynamic, typed signature based on the backend tool's inputSchema.

    This allows the inspector to render separate fields instead of a single 'kwargs' box.
    """
    assert _multi is not None
    tool_name = getattr(tool, "name", None) or "tool"
    description = getattr(tool, "description", "")
    schema = getattr(tool, "inputSchema", None) or {}
    props: Dict[str, Any] = schema.get("properties", {}) if isinstance(schema, dict) else {}
    required = set(schema.get("required", []) if isinstance(schema, dict) else [])

    # Build a dynamic async function with explicit parameters so FastMCP can introspect them.
    # We don't attempt strict typing; inspector cares about names and optional vs required.
    param_list: list[str] = []
    for key in props.keys():
        if key in required:
            param_list.append(f"{key}")
        else:
            param_list.append(f"{key}=None")

    params_sig = ", ".join(param_list)

    # Function source
    fn_src = (
        "async def __generated_proxy__(" + params_sig + "):\n"
        "    _params = locals().copy()\n"
        "    # Remove unset optionals (None) to avoid sending extraneous keys\n"
        "    _args = {k: v for k, v in _params.items() if v is not None}\n"
        "    _result = await __proxy_call__(\"" + tool_name.replace("\"", "\\\"") + "\", _args)\n"
        "    return _result\n"
    )

    # Shared proxy invocation helper
    async def __proxy_call__(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        result = await _multi.call_tool(name, arguments)
        try:
            content = getattr(result, "content", None)
            if content is not None:
                return {"content": content}
            return {"content": [TextContent(type="text", text=json.dumps(result))]}
        except Exception:
            return {"content": [TextContent(type="text", text=str(result))]}

    # Create function object in an isolated namespace
    ns: Dict[str, Any] = {"__proxy_call__": __proxy_call__}
    exec(fn_src, ns, ns)
    generated = ns["__generated_proxy__"]
    generated.__name__ = tool_name
    generated.__doc__ = description or f"Proxy tool forwarding to '{tool_name}'"

    # Register with FastMCP
    mcp.tool()(generated)


def initialize_aggregator() -> None:
    global _multi
    # Load backends from YAML (or defaults)
    server_configs = load_server_configs()
    _multi = MultiMCPClient(server_configs)

    # Run async initialization to discover tools
    asyncio.run(_multi.initialize())

    # Register a proxy tool for each discovered backend tool
    for tool in _multi.get_all_tools():
        try:
            _register_typed_proxy(tool)
        except Exception:
            # Fallback to kwargs-style proxy on error
            name = getattr(tool, "name", None) or "tool"
            description = getattr(tool, "description", "")
            _register_proxy_tool(name, description)


if __name__ == "__main__":
    # Discover backend servers and register proxy tools
    initialize_aggregator()
    # Expose a single SSE endpoint that lists and calls all aggregated tools
    mcp.run(transport="sse")
