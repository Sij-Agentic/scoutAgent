import json
from typing import Dict, Any

from mcp.types import TextContent

from .base import MCPServer

# Create server instance
server = MCPServer(name="core-tools")


@server.tool()
async def ping() -> Dict[str, Any]:
    """Health-check tool returning a simple pong payload."""
    payload = {"ok": True, "message": "pong"}
    return {
        "content": [
            TextContent(type="text", text=json.dumps(payload))
        ]
    }


@server.tool()
async def add(a: int, b: int) -> Dict[str, Any]:
    """Add two integers and return the sum."""
    payload = {"result": a + b}
    return {
        "content": [
            TextContent(type="text", text=json.dumps(payload))
        ]
    }


# Expose ASGI app for uvicorn: `uvicorn mcp_integration.server.app:app`
app = server.asgi_app()
