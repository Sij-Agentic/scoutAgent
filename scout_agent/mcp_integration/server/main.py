import json
from typing import Dict, Any

from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

# Create FastMCP server
mcp = FastMCP("core-tools", host="127.0.0.1", port=8003)


@mcp.tool()
async def ping() -> Dict[str, Any]:
    """Health-check tool returning a simple pong payload."""
    payload = {"ok": True, "message": "pong"}
    return {"content": [TextContent(type="text", text=json.dumps(payload))]}


@mcp.tool()
async def add(a: int, b: int) -> Dict[str, Any]:
    """Add two integers and return the sum."""
    payload = {"result": a + b}
    return {"content": [TextContent(type="text", text=json.dumps(payload))]}


if __name__ == "__main__":
    # Run with SSE transport and inspector
    mcp.run(
        transport="sse",
    )
