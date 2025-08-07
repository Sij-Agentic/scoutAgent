"""
Base Server implementation for MCP.

This module provides the core Server class that manages tool registration,
listing, and execution, along with the Context class for managing tool context.
"""

import asyncio
import inspect
import json
import logging
import time
import traceback
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional, Type, Union, get_type_hints

from ..models import ServerInfo, Tool, ToolRequest, ToolResponse, ToolSchema, TextContent, ErrorContent

# Set up logging
logger = logging.getLogger("mcp.server")


class Context:
    """
    Context object passed to tool implementations.
    
    Provides access to server information and helper methods for tools.
    """
    
    def __init__(self, server: 'Server', request_id: str):
        """
        Initialize a new context for tool execution.
        
        Args:
            server: The server instance this context belongs to
            request_id: Unique ID for the current request
        """
        self.server = server
        self.request_id = request_id
        self._data: Dict[str, Any] = {}
        self._start_time = time.time()
    
    @property
    def elapsed_time(self) -> float:
        """Return the elapsed time since context creation."""
        return time.time() - self._start_time
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the context data."""
        return self._data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in the context data."""
        self._data[key] = value
    
    def log(self, level: str, message: str) -> None:
        """Log a message with the specified level."""
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logger.log(numeric_level, f"[{self.request_id}] {message}")


class Server:
    """
    MCP Server implementation that manages tools and handles requests.
    
    This class provides the core functionality for an MCP server, including
    tool registration, listing, and execution.
    """
    
    def __init__(self, name: str, version: str = "1.0.0", description: Optional[str] = None, timeout: float = 60.0):
        """
        Initialize a new MCP server.
        
        Args:
            name: Server name
            version: Server version
            description: Optional server description
            timeout: Default timeout for tool execution in seconds
        """
        self.name = name
        self.version = version
        self.description = description or f"MCP Server: {name}"
        self.timeout = timeout
        self.tools: Dict[str, Tool] = {}
        self.tool_handlers: Dict[str, Callable] = {}
        self.info = ServerInfo(
            name=name,
            version=version,
            description=description,
            capabilities=["list_tools", "call_tool"]
        )
        self._server_initialized = False
        self._initialize_handlers()
    
    def _initialize_handlers(self) -> None:
        """Initialize built-in request handlers."""
        # Register core handlers
        self.list_tools_handler = self.list_tools()
        self.call_tool_handler = self.call_tool()
    
    def register_tool(self, name: str, description: str, handler: Callable, input_schema: Optional[Union[Dict, ToolSchema]] = None) -> None:
        """
        Register a new tool with the server.
        
        Args:
            name: Tool name
            description: Tool description
            handler: Callable that implements the tool
            input_schema: JSON Schema for tool inputs
        """
        if name in self.tools:
            raise ValueError(f"Tool '{name}' is already registered")
        
        # Create schema from function signature if not provided
        if input_schema is None:
            input_schema = self._create_schema_from_signature(handler)
        
        # Convert dict schema to ToolSchema
        if isinstance(input_schema, dict):
            input_schema = ToolSchema(**input_schema)
        
        tool = Tool(
            name=name,
            description=description,
            inputSchema=input_schema
        )
        
        self.tools[name] = tool
        self.tool_handlers[name] = handler
        logger.info(f"Registered tool: {name}")
    
    def _create_schema_from_signature(self, handler: Callable) -> ToolSchema:
        """
        Create a JSON schema from a function signature.
        
        Args:
            handler: The function to create a schema for
            
        Returns:
            A JSON schema describing the function parameters
        """
        sig = inspect.signature(handler)
        type_hints = get_type_hints(handler)
        
        properties = {}
        required = []
        
        # Skip first parameter if it's context
        params = list(sig.parameters.items())
        if params and params[0][0] == "ctx" and len(params) > 1:
            params = params[1:]  # Skip context parameter
        
        # Handle pydantic model parameter
        if len(params) == 1 and hasattr(type_hints.get(params[0][0], None), "__fields__"):
            model_class = type_hints[params[0][0]]
            # Extract schema from Pydantic model
            schema = model_class.schema()
            return ToolSchema(**schema)
        
        # Process parameters
        for name, param in params:
            param_type = type_hints.get(name, Any)
            properties[name] = {"type": self._type_to_json_type(param_type)}
            
            if param.default is param.empty:
                required.append(name)
        
        return ToolSchema(
            type="object",
            properties=properties,
            required=required
        )
    
    def _type_to_json_type(self, typ: Type) -> str:
        """Convert a Python type to a JSON Schema type."""
        if typ in (str, bytes):
            return "string"
        elif typ in (int, float):
            return "number"
        elif typ is bool:
            return "boolean"
        elif typ in (list, tuple):
            return "array"
        elif typ in (dict, Dict):
            return "object"
        else:
            return "string"  # Default to string for unknown types
    
    def list_tools(self):
        """Decorator for the list_tools handler."""
        def decorator(func):
            async def wrapper():
                return {"tools": list(self.tools.values())}
            
            return wrapper
        
        # Return a default implementation if called without a function
        if callable(self):
            return decorator(self)
        return decorator
    
    def call_tool(self):
        """Decorator for the call_tool handler."""
        def decorator(func):
            async def wrapper(name: str, arguments: Dict[str, Any], request_id: Optional[str] = None) -> ToolResponse:
                if name not in self.tools:
                    return ToolResponse(content=[
                        ErrorContent(
                            type="error",
                            message=f"Tool not found: {name}",
                            code="tool_not_found"
                        )
                    ])
                
                # Generate request ID if not provided
                if request_id is None:
                    request_id = str(uuid.uuid4())
                
                try:
                    # Create execution context
                    ctx = Context(self, request_id)
                    
                    # Call the original handler if provided
                    if callable(func) and func is not wrapper:
                        return await func(name, arguments, ctx)
                    
                    # Get tool handler
                    handler = self.tool_handlers[name]
                    
                    # Execute tool with timeout
                    try:
                        # Check if handler expects a context parameter
                        sig = inspect.signature(handler)
                        param_names = list(sig.parameters.keys())
                        
                        # Determine if we should pass a context object
                        if param_names and param_names[0] == "ctx":
                            # Pass context as first arg
                            result = await asyncio.wait_for(
                                handler(ctx, **arguments),
                                timeout=self.timeout
                            )
                        else:
                            # Pass only the arguments
                            result = await asyncio.wait_for(
                                handler(**arguments),
                                timeout=self.timeout
                            )
                        
                        # Handle different return types
                        if isinstance(result, ToolResponse):
                            return result
                        elif isinstance(result, dict) and "content" in result:
                            return ToolResponse(**result)
                        elif isinstance(result, (str, int, float, bool, list, dict)):
                            # Convert simple types to text content
                            return ToolResponse(content=[
                                TextContent(text=json.dumps(result))
                            ])
                        else:
                            return ToolResponse(content=[
                                TextContent(text=str(result))
                            ])
                        
                    except asyncio.TimeoutError:
                        logger.error(f"Tool execution timed out: {name}")
                        return ToolResponse(content=[
                            ErrorContent(
                                type="error",
                                message=f"Tool execution timed out after {self.timeout} seconds",
                                code="timeout"
                            )
                        ])
                
                except Exception as e:
                    logger.exception(f"Error executing tool {name}: {e}")
                    return ToolResponse(content=[
                        ErrorContent(
                            type="error",
                            message=f"Tool execution failed: {str(e)}",
                            code="execution_error"
                        )
                    ])
            
            return wrapper
        
        # Return a default implementation if called without a function
        if callable(self):
            return decorator(self)
        return decorator
    
    def tool(self, name: Optional[str] = None, description: Optional[str] = None, 
            input_schema: Optional[Dict] = None):
        """
        Decorator to register a function as a tool.
        
        Args:
            name: Optional name for the tool (defaults to function name)
            description: Optional description (defaults to docstring)
            input_schema: Optional JSON schema for inputs
        
        Returns:
            Decorated function
        """
        def decorator(func):
            nonlocal name, description
            
            # Use function name if not provided
            if name is None:
                name = func.__name__
            
            # Use docstring if description not provided
            if description is None:
                description = inspect.getdoc(func) or f"Tool: {name}"
            
            # Register the tool
            self.register_tool(
                name=name,
                description=description,
                handler=func,
                input_schema=input_schema
            )
            
            return func
        
        return decorator
    
    def create_initialization_options(self) -> Dict[str, Any]:
        """Create initialization options for the server."""
        return {
            "server_info": self.info.dict()
        }
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a request message from a client.
        
        Args:
            request: Request message
            
        Returns:
            Response message
        """
        if "type" not in request:
            return {"error": "Invalid request format: missing 'type' field"}
        
        req_type = request["type"]
        
        if req_type == "initialize":
            return {"type": "initialize_result", "server_info": self.info.dict()}
        
        elif req_type == "list_tools":
            tools_list = await self.list_tools_handler()
            return {"type": "list_tools_result", **tools_list}
        
        elif req_type == "call_tool":
            try:
                tool_request = ToolRequest(**request.get("request", {}))
                response = await self.call_tool_handler(
                    tool_request.name,
                    tool_request.arguments,
                    tool_request.request_id
                )
                return {
                    "type": "call_tool_result",
                    "response": response.dict()
                }
            except Exception as e:
                logger.exception(f"Error processing call_tool request: {e}")
                return {
                    "type": "error",
                    "message": f"Error processing request: {str(e)}"
                }
        
        else:
            return {"type": "error", "message": f"Unknown request type: {req_type}"}
    
    async def run(self, read_stream: Callable[[], Awaitable[str]], 
                write_stream: Callable[[str], Awaitable[None]],
                init_options: Optional[Dict[str, Any]] = None):
        """
        Run the server with the given streams for reading and writing.
        
        Args:
            read_stream: Async callable that returns messages from the client
            write_stream: Async callable that sends messages to the client
            init_options: Optional initialization options
        """
        self._server_initialized = True
        
        try:
            # Send initialization message with server info
            await write_stream(json.dumps({
                "type": "initialize_result",
                "server_info": self.info.dict(),
                **(init_options or {})
            }))
            
            # Process messages in a loop
            while True:
                try:
                    message = await read_stream()
                    if not message:
                        # Empty message might indicate end of stream
                        logger.info("Received empty message, exiting")
                        break
                    
                    # Parse and handle the request
                    request = json.loads(message)
                    response = await self.handle_request(request)
                    
                    # Send the response
                    await write_stream(json.dumps(response))
                
                except json.JSONDecodeError:
                    logger.error("Invalid JSON message received")
                    await write_stream(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON message"
                    }))
                
                except Exception as e:
                    logger.exception(f"Error handling message: {e}")
                    await write_stream(json.dumps({
                        "type": "error",
                        "message": f"Internal server error: {str(e)}"
                    }))
        
        except Exception as e:
            logger.exception(f"Fatal server error: {e}")
            try:
                await write_stream(json.dumps({
                    "type": "error",
                    "message": f"Fatal server error: {str(e)}"
                }))
            except:
                pass
        
        finally:
            self._server_initialized = False
            logger.info("Server stopped")
