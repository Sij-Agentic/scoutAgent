"""
Base client implementation for MCP.

This module provides the core ClientSession class for connecting to MCP servers
and calling tools.
"""

import asyncio
import json
import logging
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from ..models import Content, ServerInfo, Tool, ToolResponse, ToolsResponse

# Set up logging
logger = logging.getLogger("mcp.client")


class ClientSession:
    """
    MCP Client Session for interacting with MCP servers.
    
    This class manages a session with an MCP server, including initialization,
    tool listing, and tool calling.
    """
    
    def __init__(self, 
                 read_stream: Callable[[], Awaitable[str]], 
                 write_stream: Callable[[str], Awaitable[None]],
                 timeout: float = 60.0):
        """
        Initialize a new client session.
        
        Args:
            read_stream: Async callable that returns messages from the server
            write_stream: Async callable that sends messages to the server
            timeout: Default timeout for requests in seconds
        """
        self.read_stream = read_stream
        self.write_stream = write_stream
        self.timeout = timeout
        self.server_info: Optional[ServerInfo] = None
        self._initialized = False
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._read_task: Optional[asyncio.Task] = None
        self._session_id = str(uuid.uuid4())
        self._available_tools: Dict[str, Tool] = {}
    
    async def __aenter__(self):
        """Enter async context manager."""
        # Start background task to read responses
        self._read_task = asyncio.create_task(self._read_responses())
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager."""
        # Cancel read task and clean up
        if self._read_task and not self._read_task.done():
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
        
        # Cancel any pending requests
        for request_id, future in list(self._pending_requests.items()):
            if not future.done():
                future.set_exception(asyncio.CancelledError("Session closed"))
        
        self._pending_requests.clear()
        self._initialized = False
    
    async def _read_responses(self):
        """Background task to read and process responses from the server."""
        try:
            while True:
                # Read a message from the server
                message = await self.read_stream()
                if not message:
                    # Empty message might indicate end of stream
                    logger.warning("Received empty message from server, stream may be closed")
                    continue
                
                # Parse the message
                try:
                    response = json.loads(message)
                    await self._process_response(response)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON response: {message}")
                except Exception as e:
                    logger.exception(f"Error processing response: {e}")
        
        except asyncio.CancelledError:
            logger.debug("Read response task cancelled")
            raise
        
        except Exception as e:
            logger.exception(f"Fatal error in read_responses: {e}")
            # Fail any pending requests
            for request_id, future in list(self._pending_requests.items()):
                if not future.done():
                    future.set_exception(RuntimeError(f"Connection error: {e}"))
    
    async def _process_response(self, response: Dict[str, Any]):
        """
        Process a response message from the server.
        
        Args:
            response: Response message from the server
        """
        # Check message type
        if "type" not in response:
            logger.error("Received response without 'type' field")
            return
        
        response_type = response["type"]
        
        if response_type == "initialize_result":
            # Handle initialization response
            self.server_info = ServerInfo(**response.get("server_info", {}))
            self._initialized = True
            logger.info(f"Connected to server: {self.server_info.name} {self.server_info.version}")
            
            # Resolve initialization future if exists
            if "init" in self._pending_requests:
                future = self._pending_requests.pop("init")
                if not future.done():
                    future.set_result(self.server_info)
        
        elif response_type == "list_tools_result":
            # Handle tools list response
            tools_response = ToolsResponse(**response)
            
            # Update available tools
            for tool in tools_response.tools:
                self._available_tools[tool.name] = tool
            
            # Resolve list_tools future if exists
            if "list_tools" in self._pending_requests:
                future = self._pending_requests.pop("list_tools")
                if not future.done():
                    future.set_result(tools_response)
        
        elif response_type == "call_tool_result":
            # Handle tool call response
            tool_response = ToolResponse(**response.get("response", {}))
            
            # Check if we have a pending request for this response
            request_id = response.get("request_id", "")
            if request_id and request_id in self._pending_requests:
                future = self._pending_requests.pop(request_id)
                if not future.done():
                    future.set_result(tool_response)
            else:
                logger.warning(f"Received response for unknown request ID: {request_id}")
        
        elif response_type == "error":
            # Handle error response
            error_message = response.get("message", "Unknown error")
            
            # Check if this is a response to a pending request
            request_id = response.get("request_id", "")
            if request_id and request_id in self._pending_requests:
                future = self._pending_requests.pop(request_id)
                if not future.done():
                    future.set_exception(RuntimeError(error_message))
            else:
                logger.error(f"Received error from server: {error_message}")
        
        else:
            logger.warning(f"Received unknown response type: {response_type}")
    
    async def _send_request(self, request_type: str, request_id: Optional[str] = None, 
                           **kwargs) -> asyncio.Future:
        """
        Send a request to the server and register a future for the response.
        
        Args:
            request_type: Type of request
            request_id: Optional request ID
            **kwargs: Additional request parameters
            
        Returns:
            Future that will be resolved with the response
        """
        # Generate a request ID if not provided
        request_id = request_id or str(uuid.uuid4())
        
        # Create the request message
        request = {
            "type": request_type,
            "request_id": request_id,
            **kwargs
        }
        
        # Create a future for the response
        future = asyncio.Future()
        self._pending_requests[request_id] = future
        
        # Send the request
        try:
            await self.write_stream(json.dumps(request))
        except Exception as e:
            # Clean up future if send fails
            self._pending_requests.pop(request_id, None)
            raise RuntimeError(f"Error sending request: {e}")
        
        return future
    
    async def initialize(self) -> ServerInfo:
        """
        Initialize the client session with the server.
        
        Returns:
            ServerInfo object with server information
        """
        if self._initialized:
            return self.server_info
        
        # Send initialization request
        future = await self._send_request("initialize", request_id="init")
        
        try:
            # Wait for the response with timeout
            self.server_info = await asyncio.wait_for(future, self.timeout)
            self._initialized = True
            return self.server_info
        except asyncio.TimeoutError:
            raise RuntimeError("Initialization timed out")
    
    async def list_tools(self) -> ToolsResponse:
        """
        Get a list of available tools from the server.
        
        Returns:
            ToolsResponse with the list of available tools
        """
        # Ensure we're initialized
        if not self._initialized:
            await self.initialize()
        
        # Send list_tools request
        future = await self._send_request("list_tools", request_id="list_tools")
        
        try:
            # Wait for the response with timeout
            tools_response = await asyncio.wait_for(future, self.timeout)
            return tools_response
        except asyncio.TimeoutError:
            raise RuntimeError("list_tools request timed out")
    
    async def call_tool(self, name: str, arguments: Dict[str, Any], 
                        request_id: Optional[str] = None) -> ToolResponse:
        """
        Call a tool on the server.
        
        Args:
            name: Tool name
            arguments: Tool arguments
            request_id: Optional request ID
            
        Returns:
            ToolResponse with the tool execution result
        """
        # Ensure we're initialized
        if not self._initialized:
            await self.initialize()
        
        # Check if tool is available
        if not self._available_tools and "list_tools" in self.server_info.capabilities:
            await self.list_tools()
        
        # Send call_tool request
        request_id = request_id or str(uuid.uuid4())
        future = await self._send_request(
            "call_tool",
            request_id=request_id,
            request={"name": name, "arguments": arguments, "request_id": request_id}
        )
        
        try:
            # Wait for the response with timeout
            tool_response = await asyncio.wait_for(future, self.timeout)
            return tool_response
        except asyncio.TimeoutError:
            raise RuntimeError(f"Tool call '{name}' timed out after {self.timeout} seconds")
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """
        Get a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool object if found, None otherwise
        """
        return self._available_tools.get(name)
