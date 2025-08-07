"""
Multi-server client implementation for MCP.

This module provides a client that can connect to multiple MCP servers
and manage tools across them.
"""

import asyncio
import json
import logging
import yaml
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

from ..models import Tool, ToolResponse, ServerConfig, TextContent, ErrorContent
from .base import ClientSession
from .sse import sse_client

# Set up logging
logger = logging.getLogger("mcp.client.multi")


class MultiServerClient:
    """
    Client for managing multiple MCP servers.
    
    This class provides functionality for connecting to multiple MCP servers,
    discovering and calling tools across them, and handling server failures.
    """
    
    def __init__(self):
        """Initialize a new multi-server client."""
        self.servers: Dict[str, Dict[str, Any]] = {}
        self.sessions: Dict[str, ClientSession] = {}
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.tools_by_server: Dict[str, Set[str]] = {}
        self._initialized = False
    
    @classmethod
    async def create(cls, config_file: Optional[str] = None, 
                   server_configs: Optional[List[ServerConfig]] = None) -> 'MultiServerClient':
        """
        Create and initialize a new multi-server client.
        
        Args:
            config_file: Optional path to a YAML config file
            server_configs: Optional list of server configurations
            
        Returns:
            Initialized multi-server client
        """
        client = cls()
        
        # Load configurations
        if config_file:
            await client.load_config(config_file)
        
        if server_configs:
            for config in server_configs:
                client.add_server(config)
        
        # Initialize the client
        await client.initialize()
        
        return client
    
    async def load_config(self, config_file: str) -> None:
        """
        Load server configurations from a YAML file.
        
        Args:
            config_file: Path to the YAML config file
        """
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            if not isinstance(config, dict):
                raise ValueError(f"Invalid config format: expected dict, got {type(config)}")
            
            # Process servers section
            servers = config.get('servers', [])
            for server_config in servers:
                # Convert to ServerConfig
                config_obj = ServerConfig(**server_config)
                self.add_server(config_obj)
        
        except Exception as e:
            logger.exception(f"Error loading config from {config_file}: {e}")
            raise
    
    def add_server(self, config: ServerConfig) -> str:
        """
        Add a server configuration.
        
        Args:
            config: Server configuration
            
        Returns:
            Server ID
        """
        # Generate a server ID if not provided
        server_id = config.id or f"{config.name}_{len(self.servers)}"
        
        # Store server configuration
        self.servers[server_id] = {
            "id": server_id,
            "config": config,
            "status": "disconnected"
        }
        
        # Initialize empty tool set for this server
        self.tools_by_server[server_id] = set()
        
        logger.info(f"Added server: {server_id} ({config.name} @ {config.url})")
        
        return server_id
    
    async def initialize(self) -> None:
        """
        Initialize connections to all servers.
        
        This connects to all configured servers and discovers available tools.
        """
        if self._initialized:
            return
        
        # Connect to all servers
        connect_tasks = []
        for server_id, server in self.servers.items():
            connect_tasks.append(self._connect_to_server(server_id))
        
        # Wait for all connections to complete
        if connect_tasks:
            await asyncio.gather(*connect_tasks, return_exceptions=True)
        
        self._initialized = True
    
    async def _connect_to_server(self, server_id: str) -> None:
        """
        Connect to a server and discover its tools.
        
        Args:
            server_id: Server ID
        """
        server = self.servers.get(server_id)
        if not server:
            logger.error(f"Server not found: {server_id}")
            return
        
        config = server["config"]
        
        try:
            logger.info(f"Connecting to server: {server_id} ({config.url})")
            
            # Create SSE client session
            session = await sse_client(
                url=f"{config.url}/sse",
                send_url=f"{config.url}/messages",
                headers=config.headers,
                timeout=config.timeout
            )
            
            # Store the session
            self.sessions[server_id] = session
            server["status"] = "connected"
            
            # Discover tools
            await self._discover_tools(server_id)
            
            logger.info(f"Successfully connected to server: {server_id}")
        
        except Exception as e:
            logger.exception(f"Error connecting to server {server_id}: {e}")
            server["status"] = "error"
            server["error"] = str(e)
    
    async def _discover_tools(self, server_id: str) -> None:
        """
        Discover available tools on a server.
        
        Args:
            server_id: Server ID
        """
        session = self.sessions.get(server_id)
        if not session:
            logger.warning(f"Cannot discover tools: no session for server {server_id}")
            return
        
        try:
            # List tools
            tools_response = await session.list_tools()
            
            # Store tools
            for tool in tools_response.tools:
                # Create tool record with server information
                tool_key = f"{server_id}.{tool.name}" if self.servers[server_id]["config"].prefix_tools else tool.name
                
                self.tools[tool_key] = {
                    "name": tool.name,
                    "server_id": server_id,
                    "description": tool.description,
                    "schema": tool.inputSchema,
                    "tool_obj": tool
                }
                
                # Add to server's tool set
                self.tools_by_server[server_id].add(tool_key)
            
            logger.info(f"Discovered {len(tools_response.tools)} tools on server {server_id}")
        
        except Exception as e:
            logger.exception(f"Error discovering tools on server {server_id}: {e}")
    
    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all available tools across all servers.
        
        Returns:
            Dictionary of tool information
        """
        return self.tools
    
    def get_tool_list(self) -> List[Tool]:
        """
        Get a list of all available tools.
        
        Returns:
            List of Tool objects
        """
        return [tool["tool_obj"] for tool in self.tools.values()]
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any],
                       request_id: Optional[str] = None) -> ToolResponse:
        """
        Call a tool by name.
        
        Args:
            tool_name: Tool name (with optional server prefix)
            arguments: Tool arguments
            request_id: Optional request ID
            
        Returns:
            Tool response
        """
        # Ensure we're initialized
        if not self._initialized:
            await self.initialize()
        
        # Find the tool
        tool_info = self.tools.get(tool_name)
        
        # If not found with the exact name, try to find it without server prefix
        if not tool_info:
            # Check if any tool matches the name without prefix
            matches = [info for name, info in self.tools.items() if info["name"] == tool_name]
            
            if len(matches) == 1:
                # Single match found
                tool_info = matches[0]
                tool_name = next(name for name, info in self.tools.items() if info == tool_info)
            elif len(matches) > 1:
                # Multiple matches, ambiguous
                servers = [info["server_id"] for info in matches]
                return ToolResponse(content=[
                    ErrorContent(
                        type="error",
                        message=f"Ambiguous tool name '{tool_name}', available on multiple servers: {', '.join(servers)}",
                        code="ambiguous_tool"
                    )
                ])
        
        if not tool_info:
            return ToolResponse(content=[
                ErrorContent(
                    type="error",
                    message=f"Tool not found: {tool_name}",
                    code="tool_not_found"
                )
            ])
        
        # Get server and session
        server_id = tool_info["server_id"]
        session = self.sessions.get(server_id)
        
        if not session:
            return ToolResponse(content=[
                ErrorContent(
                    type="error",
                    message=f"No active session for server: {server_id}",
                    code="server_disconnected"
                )
            ])
        
        try:
            # Call the tool
            original_name = tool_info["name"]
            response = await session.call_tool(original_name, arguments, request_id)
            return response
        
        except Exception as e:
            logger.exception(f"Error calling tool {tool_name} on server {server_id}: {e}")
            return ToolResponse(content=[
                ErrorContent(
                    type="error",
                    message=f"Tool execution failed: {str(e)}",
                    code="execution_error"
                )
            ])
    
    async def function_wrapper(self, tool_name: str, *args, **kwargs) -> Any:
        """
        Wrapper to call tools with positional and keyword arguments.
        
        This provides a more convenient interface for calling tools.
        
        Args:
            tool_name: Tool name
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Tool execution result (parsed from JSON if possible)
        """
        # Get tool schema to map positional args to parameters
        tool_info = self.tools.get(tool_name)
        
        # If not found with the exact name, try to find it without server prefix
        if not tool_info:
            # Check if any tool matches the name without prefix
            matches = [info for name, info in self.tools.items() if info["name"] == tool_name]
            
            if len(matches) == 1:
                # Single match found
                tool_info = matches[0]
                tool_name = next(name for name, info in self.tools.items() if info == tool_info)
            elif len(matches) > 1:
                # Multiple matches, ambiguous
                servers = [info["server_id"] for info in matches]
                raise ValueError(f"Ambiguous tool name '{tool_name}', available on multiple servers: {', '.join(servers)}")
        
        if not tool_info:
            raise ValueError(f"Tool not found: {tool_name}")
        
        # Get schema
        schema = tool_info["schema"]
        required_params = schema.get("required", [])
        properties = schema.get("properties", {})
        param_names = list(properties.keys())
        
        # Combine args and kwargs
        arguments = {}
        
        # Map positional args to parameter names
        for i, arg in enumerate(args):
            if i < len(param_names):
                arguments[param_names[i]] = arg
            else:
                raise ValueError(f"Too many positional arguments for tool: {tool_name}")
        
        # Add keyword args
        arguments.update(kwargs)
        
        # Check for required parameters
        missing_params = [param for param in required_params if param not in arguments]
        if missing_params:
            raise ValueError(f"Missing required parameters for tool {tool_name}: {', '.join(missing_params)}")
        
        # Call the tool
        response = await self.call_tool(tool_name, arguments)
        
        # Process response
        if not response.content:
            return None
        
        # Try to parse JSON from text content
        if isinstance(response.content[0], TextContent):
            text = response.content[0].text
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
        
        # Return the raw content if parsing fails
        return response.content
    
    async def close(self) -> None:
        """Close all server connections."""
        close_tasks = []
        
        for server_id, session in self.sessions.items():
            close_tasks.append(self._close_session(server_id))
        
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        
        self._initialized = False
        logger.info("Closed all server connections")
    
    async def _close_session(self, server_id: str) -> None:
        """
        Close a specific server session.
        
        Args:
            server_id: Server ID
        """
        session = self.sessions.pop(server_id, None)
        if session:
            try:
                await session.__aexit__(None, None, None)
                self.servers[server_id]["status"] = "disconnected"
                logger.info(f"Closed connection to server: {server_id}")
            except Exception as e:
                logger.error(f"Error closing session for server {server_id}: {e}")
    
    async def __aenter__(self) -> 'MultiServerClient':
        """Enter async context manager."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager."""
        await self.close()
