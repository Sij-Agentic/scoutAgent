"""
SSE Server Transport implementation for MCP.

This module provides an ASGI-compatible SSE transport for MCP servers.
"""

import asyncio
import json
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, AsyncGenerator

# Set up logging
logger = logging.getLogger("mcp.server.sse")


class SseServerTransport:
    """
    Server-Sent Events transport for MCP servers.
    
    This class implements an ASGI-compatible SSE transport for MCP servers,
    allowing them to communicate with clients over HTTP using Server-Sent Events.
    """
    
    def __init__(self, post_path: str = "/messages"):
        """
        Initialize SSE server transport.
        
        Args:
            post_path: Path for POST messages (default: '/messages')
        """
        self.post_path = post_path
        self._connections: Dict[str, Dict[str, Any]] = {}
        self._message_queues: Dict[str, asyncio.Queue] = {}
        self._active_connections = 0
    
    async def connect_sse(self, scope, receive, send) -> Tuple[Callable[[], Any], Callable[[str], Any]]:
        """
        Handle an SSE connection.
        
        Args:
            scope: ASGI scope
            receive: ASGI receive function
            send: ASGI send function
            
        Returns:
            Tuple of (read_stream, write_stream) functions
        """
        # Generate a unique connection ID
        connection_id = str(uuid.uuid4())
        
        # Create message queue for this connection
        message_queue = asyncio.Queue()
        self._message_queues[connection_id] = message_queue
        
        # Store connection info
        self._connections[connection_id] = {
            "id": connection_id,
            "created_at": asyncio.get_event_loop().time(),
            "last_activity": asyncio.get_event_loop().time(),
            "remote": scope.get("client", ("unknown", 0))[0],
        }
        
        self._active_connections += 1
        logger.info(f"New SSE connection: {connection_id} (active: {self._active_connections})")
        
        # Create reader and writer
        async def read_stream() -> str:
            try:
                message = await message_queue.get()
                self._connections[connection_id]["last_activity"] = asyncio.get_event_loop().time()
                return message
            except Exception as e:
                logger.error(f"Error reading from stream: {e}")
                return ""
        
        async def write_stream(message: str) -> None:
            try:
                # Format as SSE message
                data = f"data: {message}\n\n"
                await send({
                    "type": "http.response.body",
                    "body": data.encode("utf-8"),
                    "more_body": True
                })
                self._connections[connection_id]["last_activity"] = asyncio.get_event_loop().time()
            except Exception as e:
                logger.error(f"Error writing to stream: {e}")
        
        # Set up the SSE response
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-type", b"text/event-stream"),
                (b"cache-control", b"no-cache"),
                (b"connection", b"keep-alive"),
                (b"access-control-allow-origin", b"*"),
            ],
        })
        
        # Send initial comment to establish connection
        await send({
            "type": "http.response.body",
            "body": b": connected\n\n",
            "more_body": True
        })
        
        # Run connection handler task
        connection_task = asyncio.create_task(self._handle_sse_connection(connection_id, scope, receive, send))
        
        try:
            # Return the streams for the server to use
            return read_stream, write_stream
        finally:
            # Ensure we clean up on exit
            await connection_task
    
    async def _handle_sse_connection(self, connection_id: str, scope, receive, send) -> None:
        """
        Background task to handle an SSE connection.
        
        This runs in the background and cleans up when the connection is closed.
        
        Args:
            connection_id: Unique connection ID
            scope: ASGI scope
            receive: ASGI receive function
            send: ASGI send function
        """
        try:
            # Wait for disconnect message
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    logger.info(f"SSE client disconnected: {connection_id}")
                    break
                
                # Keep the connection alive with heartbeats
                if message["type"] == "http.request" and asyncio.get_event_loop().time() - self._connections[connection_id]["last_activity"] > 15:
                    await send({
                        "type": "http.response.body",
                        "body": b": ping\n\n",
                        "more_body": True
                    })
                    self._connections[connection_id]["last_activity"] = asyncio.get_event_loop().time()
        
        except asyncio.CancelledError:
            logger.info(f"SSE connection task cancelled: {connection_id}")
        
        except Exception as e:
            logger.exception(f"Error in SSE connection handler: {e}")
        
        finally:
            # Clean up connection
            self._message_queues.pop(connection_id, None)
            self._connections.pop(connection_id, None)
            self._active_connections -= 1
            logger.info(f"SSE connection closed: {connection_id} (active: {self._active_connections})")
    
    async def handle_post_message(self, scope, receive, send) -> None:
        """
        Handle a POST request to send a message to an SSE connection.
        
        Args:
            scope: ASGI scope
            receive: ASGI receive function
            send: ASGI send function
        """
        # Read request body
        body = b""
        more_body = True
        
        while more_body:
            message = await receive()
            if message["type"] == "http.request":
                body += message.get("body", b"")
                more_body = message.get("more_body", False)
        
        # Parse the message
        try:
            data = json.loads(body)
            connection_id = data.get("connection_id")
            message_content = data.get("message")
            
            # Validate connection ID
            if not connection_id or connection_id not in self._connections:
                await self._send_error_response(send, 404, "Connection not found")
                return
            
            # Validate message content
            if not message_content:
                await self._send_error_response(send, 400, "Missing message content")
                return
            
            # Queue the message for the SSE connection
            queue = self._message_queues.get(connection_id)
            if queue:
                await queue.put(message_content)
                
                # Send success response
                await send({
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"access-control-allow-origin", b"*"),
                    ],
                })
                
                await send({
                    "type": "http.response.body",
                    "body": json.dumps({"success": True}).encode("utf-8"),
                })
            else:
                await self._send_error_response(send, 500, "Message queue not available")
        
        except json.JSONDecodeError:
            await self._send_error_response(send, 400, "Invalid JSON")
        except Exception as e:
            logger.exception(f"Error handling POST message: {e}")
            await self._send_error_response(send, 500, f"Internal server error: {str(e)}")
    
    async def _send_error_response(self, send, status: int, message: str) -> None:
        """
        Send an error response.
        
        Args:
            send: ASGI send function
            status: HTTP status code
            message: Error message
        """
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"access-control-allow-origin", b"*"),
            ],
        })
        
        await send({
            "type": "http.response.body",
            "body": json.dumps({"error": message}).encode("utf-8"),
        })
    
    async def handle_cors_preflight(self, scope, receive, send) -> None:
        """
        Handle CORS preflight requests.
        
        Args:
            scope: ASGI scope
            receive: ASGI receive function
            send: ASGI send function
        """
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"access-control-allow-origin", b"*"),
                (b"access-control-allow-methods", b"GET, POST, OPTIONS"),
                (b"access-control-allow-headers", b"content-type"),
                (b"access-control-max-age", b"86400"),  # 24 hours
            ],
        })
        
        await send({
            "type": "http.response.body",
            "body": b"",
        })


async def create_asgi_app(server) -> None:
    """
    Create an ASGI application for an MCP server.
    
    Args:
        server: MCP server instance
        
    Returns:
        ASGI application function
    """
    transport = SseServerTransport()
    
    # Get registered tools
    tools = server.get_tools() if hasattr(server, 'get_tools') else []
    
    async def app(scope, receive, send):
        """ASGI application for MCP server."""
        path = scope.get("path", "")
        method = scope.get("method", "GET")
        
        logger.info(f"ASGI request: {method} {path}")
        
        if method == "OPTIONS":
            # Handle CORS preflight requests
            logger.info(f"Handling OPTIONS request for {path}")
            await transport.handle_cors_preflight(scope, receive, send)
            return
        
        if path == "/sse" and method == "GET":
            # Handle SSE connection
            logger.info(f"Handling SSE connection request")
            streams = await transport.connect_sse(scope, receive, send)
            await server.run(streams[0], streams[1], server.create_initialization_options())
            return
        
        # Handle POST messages - be more lenient with path matching
        if method == "POST" and (path == transport.post_path or path.endswith("/messages")):
            logger.info(f"Handling POST message request for {path}")
            await transport.handle_post_message(scope, receive, send)
            return
            
        # Handle GET /tools endpoint
        if path == "/tools" and method == "GET":
            logger.info(f"Handling GET tools request")
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"access-control-allow-origin", b"*"),
                ],
            })
            
            await send({
                "type": "http.response.body",
                "body": json.dumps(tools).encode("utf-8"),
            })
            return
            
        # Handle POST /tools/{tool_name} endpoint
        if method == "POST" and path.startswith("/tools/"):
            logger.info(f"Handling tool invocation request for {path}")
            tool_name = path.split("/")[-1]
            
            # Read request body
            body = b""
            more_body = True
            
            while more_body:
                message = await receive()
                if message["type"] == "http.request":
                    body += message.get("body", b"")
                    more_body = message.get("more_body", False)
            
            try:
                # Parse arguments
                arguments = json.loads(body)
                
                # Create a tool request
                request_id = str(uuid.uuid4())
                
                # Send success response (will be updated with actual result)
                await send({
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"access-control-allow-origin", b"*"),
                    ],
                })
                
                await send({
                    "type": "http.response.body",
                    "body": json.dumps({"status": "processing", "request_id": request_id}).encode("utf-8"),
                })
                
                return
            except Exception as e:
                logger.exception(f"Error handling tool invocation: {e}")
                await transport._send_error_response(send, 500, f"Internal server error: {str(e)}")
                return
        
        # Handle 404 for other routes
        await send({
            "type": "http.response.start",
            "status": 404,
            "headers": [(b"content-type", b"application/json")],
        })
        
        await send({
            "type": "http.response.body",
            "body": json.dumps({"error": "Not found"}).encode("utf-8"),
        })
    
    return app
