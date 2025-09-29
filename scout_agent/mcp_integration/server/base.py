from __future__ import annotations

import asyncio
from typing import Callable, Awaitable, Any

from mcp.server.fastmcp import FastMCP


class MCPServer:
    """
    Base ASGI-compatible MCP server using SSE transport.

    Usage:
        server = MCPServer(name="tool-server")

        @server.tool()
        async def echo(text: str) -> dict:
            return {"content": [{"type": "text", "text": text}]}

        # Run with Uvicorn:
        #   uvicorn mcp.server.base:server --factory
        # Or create your own app via server.asgi_app()
    """

    def __init__(self, name: str, sse_path: str = "/sse", messages_path: str = "/messages"):
        self.name = name
        # Configure FastMCP with matching SSE and message paths so our ASGI app routes line up.
        # Note: FastMCP expects message_path to typically end with a trailing slash; we pass as-is
        # because clients only use /sse in our setup. Adjust if needed.
        self._mcp = FastMCP(name, sse_path=sse_path, message_path=messages_path)
        self._sse_path = sse_path
        self._messages_path = messages_path

    # Re-export the decorator for convenience
    def tool(self):
        return self._mcp.tool()

    def asgi_app(self) -> Callable[[dict, Callable[..., Awaitable[Any]], Callable[..., Awaitable[Any]]], Awaitable[None]]:
        # Delegate to FastMCP's built-in SSE Starlette app to ensure compatibility with the installed SDK.
        starlette_app = self._mcp.sse_app(mount_path=None)

        async def app(scope, receive, send):
            await starlette_app(scope, receive, send)

        return app

    # Allow instance to be used directly as ASGI app
    async def __call__(self, scope, receive, send):
        app = self.asgi_app()
        await app(scope, receive, send)
