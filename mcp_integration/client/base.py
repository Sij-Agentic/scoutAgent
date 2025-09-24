import asyncio
import logging
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from mcp import ClientSession
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Enhanced SSE-only MCP client wrapper with connection resilience.

    - Establishes an SSE connection to an MCP server.
    - Lists tools and calls tools with retry logic.
    - Manages clean async shutdown.
    - Handles connection drops and automatic reconnection.
    """

    def __init__(self, server_url: str, max_retries: int = 5, connection_timeout: int = 120, sse_read_timeout: int = 1800):
        self.server_url = server_url
        self.max_retries = max_retries
        self.connection_timeout = connection_timeout
        self.sse_read_timeout = sse_read_timeout  # SSE read timeout for long operations
        self.session: Optional[ClientSession] = None
        self.session_context = None  # the aenter/aexit handle from sse_client
        self._session_stack = []  # LIFO shutdown order
        self._connection_lock = asyncio.Lock()  # Prevent concurrent connection attempts
        self._last_connection_error = None

    async def ensure_session(self) -> ClientSession:
        """Ensure we have a valid session, with connection retry logic."""
        async with self._connection_lock:
            if self.session:
                try:
                    # Test if session is still alive by attempting a quick operation
                    await asyncio.wait_for(self.session.list_tools(), timeout=5)
                    return self.session
                except Exception as e:
                    logger.warning(f"Existing session appears dead, reconnecting: {e}")
                    await self._cleanup_session()

            # Attempt to create new session with retries
            for attempt in range(self.max_retries):
                try:
                    logger.info(f"Connecting to MCP server {self.server_url} (attempt {attempt + 1}/{self.max_retries})")
                    
                    # Create SSE client context with custom timeouts
                    self.session_context = sse_client(
                        self.server_url,
                        timeout=self.connection_timeout,
                        sse_read_timeout=self.sse_read_timeout
                    )

                    # Enter transport context -> returns (read, write)
                    read, write = await asyncio.wait_for(
                        self.session_context.__aenter__(),
                        timeout=self.connection_timeout
                    )
                    self._session_stack.append(("context", self.session_context))

                    # Wrap into a ClientSession and initialize
                    self.session = ClientSession(read, write)
                    await asyncio.wait_for(
                        self.session.__aenter__(),
                        timeout=self.connection_timeout
                    )
                    self._session_stack.append(("session", self.session))

                    await asyncio.wait_for(
                        self.session.initialize(),
                        timeout=self.connection_timeout
                    )
                    
                    logger.info(f"Successfully connected to MCP server {self.server_url}")
                    self._last_connection_error = None
                    return self.session
                    
                except Exception as e:
                    self._last_connection_error = e
                    logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                    await self._cleanup_session()
                    
                    if attempt < self.max_retries - 1:
                        backoff_time = min(10, 2 ** attempt)
                        logger.info(f"Retrying connection in {backoff_time} seconds...")
                        await asyncio.sleep(backoff_time)
            
            # All attempts failed
            error_msg = f"Failed to connect to MCP server {self.server_url} after {self.max_retries} attempts"
            if self._last_connection_error:
                error_msg += f". Last error: {self._last_connection_error}"
            raise ConnectionError(error_msg)

    async def _cleanup_session(self):
        """Clean up current session without raising exceptions."""
        try:
            await self.shutdown()
        except Exception as e:
            logger.debug(f"Error during session cleanup: {e}")

    async def list_tools(self):
        """List available tools with retry logic."""
        for attempt in range(self.max_retries):
            try:
                session = await self.ensure_session()
                tools_result = await session.list_tools()
                return tools_result.tools
            except Exception as e:
                logger.warning(f"Failed to list tools (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    await self._cleanup_session()
                    await asyncio.sleep(min(5, 2 ** attempt))
                else:
                    raise

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        """Call a tool with retry logic and enhanced error handling."""
        for attempt in range(self.max_retries):
            try:
                session = await self.ensure_session()
                logger.debug(f"Calling tool {tool_name} (attempt {attempt + 1}/{self.max_retries})")
                result = await session.call_tool(tool_name, arguments)
                logger.debug(f"Successfully called tool {tool_name}")
                return result
            except Exception as e:
                error_str = str(e).lower()
                is_connection_error = any(keyword in error_str for keyword in [
                    "connection closed", "connection", "timeout", "network", 
                    "ssl", "certificate", "eof", "broken pipe"
                ])
                
                if is_connection_error and attempt < self.max_retries - 1:
                    logger.warning(f"Connection error calling tool {tool_name} (attempt {attempt + 1}/{self.max_retries}): {e}")
                    await self._cleanup_session()
                    backoff_time = min(5, 2 ** attempt)
                    logger.info(f"Retrying tool call in {backoff_time} seconds...")
                    await asyncio.sleep(backoff_time)
                else:
                    logger.error(f"Failed to call tool {tool_name} after {attempt + 1} attempts: {e}")
                    raise

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
