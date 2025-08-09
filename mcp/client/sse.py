"""
SSE Client implementation for MCP.

This module provides an SSE client for connecting to MCP servers via
Server-Sent Events.
"""

import asyncio
import json
import logging
import httpx
import uuid
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Any, Callable, Awaitable
from urllib.parse import urljoin

from .base import ClientSession

# Set up logging
logger = logging.getLogger("mcp.client.sse")


async def _sse_parser(response: httpx.Response) -> AsyncGenerator[str, None]:
    """
    Parse an SSE stream from an HTTP response.
    
    Args:
        response: The HTTP response containing the SSE stream
        
    Yields:
        Each SSE message as a string
    """
    buffer = ""
    async for line in response.aiter_lines():
        if not line.strip():
            # Empty line signifies the end of an event
            if buffer and buffer.startswith("data: "):
                # Extract message content
                message = buffer[6:].strip()
                yield message
                buffer = ""
            else:
                buffer = ""
        elif line.startswith("data: "):
            # Data line
            buffer += line
    
    if buffer and buffer.startswith("data: "):
        # Yield any remaining message
        message = buffer[6:].strip()
        yield message


async def sse_client(
    url: str,
    send_url: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 60.0,
    heartbeat_interval: float = 30.0
) -> ClientSession:
    """
    Create a client session connected to an SSE server.
    
    Args:
        url: URL of the SSE endpoint
        send_url: Optional URL for sending messages (defaults to url/messages)
        headers: Optional HTTP headers
        timeout: Request timeout in seconds
        heartbeat_interval: Interval for sending heartbeat messages
        
    Returns:
        ClientSession object connected to the server
    """
    # Default headers
    if headers is None:
        headers = {}
    
    headers.setdefault("Accept", "text/event-stream")
    headers.setdefault("Cache-Control", "no-cache")
    
    # Default send URL
    if send_url is None:
        # Extract base URL (remove /sse from the end if present)
        base_url = url
        if base_url.endswith("/sse"):
            base_url = base_url[:-4]  # Remove /sse
        send_url = urljoin(base_url, "/messages")
    
    # Client session ID
    session_id = str(uuid.uuid4())
    
    # Create HTTP client
    client = httpx.AsyncClient(
        headers=headers,
        timeout=timeout
    )
    
    # Message queue for receiving messages
    receive_queue = asyncio.Queue()
    
    # Background task for connection
    connection_task = None
    
    async def connect() -> None:
        """Connect to the SSE server and start parsing events."""
        nonlocal connection_task
        
        try:
            logger.info(f"Connecting to SSE server: {url}")
            
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                
                logger.info(f"Connected to SSE server: {url} ({response.status_code})")
                
                # Parse SSE stream
                async for message in _sse_parser(response):
                    await receive_queue.put(message)
        
        except httpx.HTTPError as e:
            logger.error(f"HTTP error connecting to SSE server: {e}")
            await receive_queue.put("")  # Signal connection end
        
        except asyncio.CancelledError:
            logger.debug("SSE connection task cancelled")
            raise
        
        except Exception as e:
            logger.exception(f"Error in SSE connection: {e}")
            await receive_queue.put("")  # Signal connection end
    
    # Start connection
    connection_task = asyncio.create_task(connect())
    
    # Create reader function
    async def read_stream() -> str:
        """Read a message from the SSE stream."""
        try:
            return await receive_queue.get()
        except Exception as e:
            logger.error(f"Error reading from SSE stream: {e}")
            return ""
    
    # Create writer function
    async def write_stream(message: str) -> None:
        """Send a message to the server via HTTP POST."""
        try:
            # Send message via POST request
            data = {
                "connection_id": session_id,
                "message": message
            }
            
            # Await the coroutine returned by client.post
            response = await client.post(send_url, json=data)
            response.raise_for_status()
        
        except httpx.HTTPError as e:
            logger.error(f"HTTP error sending message: {e}")
        
        except Exception as e:
            logger.exception(f"Error sending message: {e}")
    
    # Create heartbeat task
    async def heartbeat() -> None:
        """Send periodic heartbeat messages to keep the connection alive."""
        try:
            while True:
                await asyncio.sleep(heartbeat_interval)
                
                # Send heartbeat message via POST
                data = {
                    "connection_id": session_id,
                    "message": json.dumps({"type": "heartbeat"})
                }
                
                try:
                    # Await the coroutine returned by client.post
                    response = await client.post(send_url, json=data, timeout=10.0)
                    if response.status_code != 200:
                        logger.warning(f"Heartbeat failed: {response.status_code}")
                except Exception as e:
                    logger.warning(f"Error sending heartbeat: {e}")
        
        except asyncio.CancelledError:
            logger.debug("Heartbeat task cancelled")
            raise
        
        except Exception as e:
            logger.exception(f"Error in heartbeat task: {e}")
    
    # Start heartbeat task
    heartbeat_task = asyncio.create_task(heartbeat())
    
    # Create client session
    session = ClientSession(read_stream, write_stream, timeout)
    
    # Cleanup handler
    orig_aexit = session.__aexit__
    
    async def aexit_wrapper(exc_type, exc_val, exc_tb):
        """Wrap the original __aexit__ to clean up resources."""
        # Cancel tasks
        for task in [connection_task, heartbeat_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close HTTP client
        await client.aclose()
        
        # Call original exit
        return await orig_aexit(exc_type, exc_val, exc_tb)
    
    # Replace exit method with wrapper
    session.__aexit__ = aexit_wrapper
    
    # Initialize the session
    await session.initialize()
    
    return session
