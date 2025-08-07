"""
MCP Service implementation for ScoutAgent.

This module provides a service implementation for MCP tool access
that integrates with the ScoutAgent service registry.
"""

import asyncio
import logging
import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

# Use conditionally imported dependencies
try:
    from service_registry import ServiceBase, service, requires, inject
    from custom_logging import get_logger
except ImportError:
    # Mock implementations for testing
    def service(*args, **kwargs):
        def decorator(cls):
            return cls
        return decorator
    
    def requires(*args, **kwargs):
        def decorator(cls):
            return cls
        return decorator
    
    def inject(*args, **kwargs):
        def decorator(cls):
            return cls
        return decorator
    
    class ServiceBase:
        def __init__(self, name, version="1.0.0"):
            self.name = name
            self.version = version
    
    def get_logger(name):
        return logging.getLogger(name)

from .client.multi import MultiServerClient
from .models import ServerConfig, Tool
from .sandbox.executor import execute_code_with_tools, SandboxedExecution


@service(name="mcp_tools", singleton=True)
@requires("config", optional=True)
@requires("logging", optional=True)
class MCPToolService(ServiceBase):
    """
    Service for accessing MCP tools.
    
    This service provides access to tools across multiple MCP servers
    and integrates with the ScoutAgent service registry.
    """
    
    def __init__(self):
        """Initialize the MCP tool service."""
        super().__init__(name="mcp_tools", version="1.0.0")
        self.logger = get_logger("service.mcp_tools")
        self.config = None
        self.client = None
        self.server_configs = []
        self.default_config_paths = [
            "./config/mcp_servers.yaml",
            "./mcp_servers.yaml",
            os.path.expanduser("~/.config/scoutagent/mcp_servers.yaml")
        ]
    
    def setup_direct(self, config=None, logger=None):
        """
        Setup the service directly without using the service registry.
        
        Args:
            config: Configuration service or dict
            logger: Logger instance
        """
        if logger:
            self.logger = logger
        
        if config:
            self.config = config
    
    async def _initialize(self, registry):
        """
        Initialize the MCP tool service.
        
        Args:
            registry: Service registry for accessing dependencies
            
        Returns:
            True if initialization was successful
        """
        try:
            self.logger.info("Initializing MCP tool service")
            
            if registry:
                # Get config service from registry if available
                try:
                    self.config = registry.get_service("config")
                    self.logger.info("Using config service from registry")
                except:
                    self.logger.info("Config service not available in registry, using direct initialization")
            
            if not self.config:
                # Create a minimal config if not available
                class MinimalConfig:
                    def get_config_value(self, key, default=None):
                        return default
                
                self.config = MinimalConfig()
                self.logger.info("Using minimal config")
            
            # Load server configurations
            await self._load_server_configs()
            
            self.logger.info("MCP tool service initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP tool service: {e}")
            return False
    
    async def _load_server_configs(self):
        """Load server configurations from config files."""
        # Check if we have server configs in the main config
        mcp_config = self.config.get_config_value("mcp", {})
        servers_config = mcp_config.get("servers", [])
        
        # Convert dict configs to ServerConfig objects
        if servers_config:
            for server in servers_config:
                self.server_configs.append(ServerConfig(**server))
        
        # Try to load from config files
        for config_path in self.default_config_paths:
            path = Path(config_path)
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    if isinstance(config, dict) and "servers" in config:
                        for server in config["servers"]:
                            self.server_configs.append(ServerConfig(**server))
                    
                    self.logger.info(f"Loaded server configs from {path}")
                    break
                except Exception as e:
                    self.logger.warning(f"Error loading config from {path}: {e}")
    
    async def _start(self):
        """
        Start the MCP tool service.
        
        Returns:
            True if startup was successful
        """
        try:
            self.logger.info("Starting MCP tool service")
            
            # Create and initialize the multi-server client
            self.client = await MultiServerClient.create(server_configs=self.server_configs)
            
            # Log available tools
            tools = self.client.get_available_tools()
            self.logger.info(f"Available tools: {len(tools)}")
            for tool_name in tools.keys():
                self.logger.debug(f"Tool available: {tool_name}")
            
            self.logger.info("MCP tool service started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start MCP tool service: {e}")
            return False
    
    async def _stop(self):
        """
        Stop the MCP tool service.
        
        Returns:
            True if shutdown was successful
        """
        try:
            self.logger.info("Stopping MCP tool service")
            
            # Close client connections
            if self.client:
                await self.client.close()
                self.client = None
            
            self.logger.info("MCP tool service stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop MCP tool service: {e}")
            return False
    
    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all available tools across all servers.
        
        Returns:
            Dictionary of tool information
        """
        if not self.client:
            return {}
        
        return self.client.get_available_tools()
    
    def get_tool_list(self) -> List[Tool]:
        """
        Get a list of all available tools.
        
        Returns:
            List of Tool objects
        """
        if not self.client:
            return []
        
        return self.client.get_tool_list()
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any],
                       request_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Call a tool by name.
        
        Args:
            tool_name: Tool name (with optional server prefix)
            arguments: Tool arguments
            request_id: Optional request ID
            
        Returns:
            Tool response as a dictionary
        """
        if not self.client:
            return {"error": "MCP tool service not started"}
        
        response = await self.client.call_tool(tool_name, arguments, request_id)
        return response.dict()
    
    async def execute_code_with_tools(self, code: str, language: str = "python",
                                    timeout: int = 30, async_mode: bool = False) -> Dict[str, Any]:
        """
        Execute code with access to MCP tools.
        
        Args:
            code: Code to execute
            language: Programming language
            timeout: Execution timeout in seconds
            async_mode: Whether to use async tool proxies
            
        Returns:
            Execution result as a dictionary
        """
        if not self.client:
            return {
                "success": False,
                "error": "MCP tool service not started",
                "exec_id": ""
            }
        
        result = await execute_code_with_tools(
            code=code,
            client=self.client,
            language=language,
            timeout=timeout,
            async_mode=async_mode
        )
        
        return result.to_dict()
    
    async def get_sandboxed_execution(self, async_mode: bool = False) -> SandboxedExecution:
        """
        Get a sandboxed execution context with tool access.
        
        Args:
            async_mode: Whether to use async tool proxies
            
        Returns:
            SandboxedExecution context manager
        """
        if not self.client:
            raise RuntimeError("MCP tool service not started")
        
        return SandboxedExecution(self.client, async_mode=async_mode)


# Global service instance for convenience
_mcp_tool_service_instance: Optional[MCPToolService] = None


def get_mcp_tool_service() -> MCPToolService:
    """
    Get the global MCP tool service instance.
    
    Returns:
        MCPToolService instance
    """
    global _mcp_tool_service_instance
    
    if _mcp_tool_service_instance is None:
        try:
            from service_registry import get_registry
            
            # Check if registered in service registry
            registry = get_registry()
            
            if registry.has_service_instance("mcp_tools"):
                _mcp_tool_service_instance = registry.get_service("mcp_tools")
            else:
                # Create and register a new instance
                _mcp_tool_service_instance = MCPToolService()
                # Register in the registry
                registry.register_instance(_mcp_tool_service_instance)
        except Exception as e:
            # If registry access fails, create a standalone instance
            logger = get_logger("mcp_tool_service")
            logger.warning(f"Could not access service registry: {e}. Creating standalone instance.")
            _mcp_tool_service_instance = MCPToolService()
            
    return _mcp_tool_service_instance
