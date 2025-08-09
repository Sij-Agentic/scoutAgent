import asyncio
from typing import Any, Dict, List, Optional

from mcp import ClientSession
from mcp.client.sse import sse_client


class MCPClient:
    """
    SSE-only MCP client wrapper.

    - Establishes an SSE connection to an MCP server.
    - Lists tools and calls tools.
    - Manages clean async shutdown.
    """

    def __init__(self, server_url: str):
        self.server_url = server_url
        self.session: Optional[ClientSession] = None
        self.session_context = None  # the aenter/aexit handle from sse_client
        self._session_stack = []  # LIFO shutdown order

    async def ensure_session(self) -> ClientSession:
        if self.session:
            return self.session

        # Create SSE client context
        self.session_context = sse_client(self.server_url)

        # Enter transport context -> returns (read, write)
        read, write = await self.session_context.__aenter__()
        self._session_stack.append(("context", self.session_context))

        # Wrap into a ClientSession and initialize
        self.session = ClientSession(read, write)
        await self.session.__aenter__()
        self._session_stack.append(("session", self.session))

        await self.session.initialize()
        return self.session

    async def list_tools(self):
        session = await self.ensure_session()
        tools_result = await session.list_tools()
        return tools_result.tools

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        session = await self.ensure_session()
        return await session.call_tool(tool_name, arguments)

    async def shutdown(self):
        # Exit in reverse (LIFO)
        for item_type, item in reversed(self._session_stack):
            try:
                await item.__aexit__(None, None, None)
            except asyncio.CancelledError:
                # Graceful on cancellation
                break
            except Exception:
                # Best-effort shutdown
                pass
        self._session_stack.clear()
        self.session = None
        self.session_context = None
