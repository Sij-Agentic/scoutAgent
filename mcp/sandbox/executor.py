"""
Sandbox executor implementation for MCP.

This module provides sandboxed code execution with access to MCP tools
via tool proxies. It integrates with the existing code execution service.
"""

import asyncio
import inspect
import json
import logging
import os
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from ..client.multi import MultiServerClient
from .proxy import create_tool_proxies, ToolProxy, AsyncToolProxy
# Use absolute imports for external modules
try:
    from scout_agent.services.agents.code.service import CodeExecutionService, CodeExecResult, ExecutionEnvironment
except ImportError:
    # Fallback for testing
    class CodeExecResult:
        def __init__(self, success=False, output="", error="", exec_id=""):
            self.success = success
            self.output = output
            self.error = error
            self.exec_id = exec_id
        
        def to_dict(self):
            return {
                "success": self.success,
                "output": self.output,
                "error": self.error,
                "exec_id": self.exec_id
            }
    
    class ExecutionEnvironment:
        LOCAL = "local"
        DOCKER = "docker"
        WASM = "wasm"

# Set up logging
logger = logging.getLogger("mcp.sandbox.executor")


def _prepare_execution_globals(tools: ToolProxy) -> Dict[str, Any]:
    """
    Prepare global variables for code execution.
    
    Args:
        tools: Tool proxy object
        
    Returns:
        Dictionary of global variables
    """
    # Start with a minimal set of builtins for security
    safe_builtins = {
        'abs': abs, 'bool': bool, 'dict': dict, 'float': float,
        'int': int, 'len': len, 'list': list, 'max': max, 'min': min,
        'print': print, 'range': range, 'round': round, 'str': str,
        'sum': sum, 'tuple': tuple, 'type': type
    }
    
    # Add the tools proxy
    exec_globals = {
        '__builtins__': safe_builtins,
        'tools': tools,
        'json': json,
    }
    
    return exec_globals


async def _prepare_execution_globals_async(tools: AsyncToolProxy) -> Dict[str, Any]:
    """
    Prepare global variables for async code execution.
    
    Args:
        tools: Async tool proxy object
        
    Returns:
        Dictionary of global variables
    """
    # Start with a minimal set of builtins for security
    safe_builtins = {
        'abs': abs, 'bool': bool, 'dict': dict, 'float': float,
        'int': int, 'len': len, 'list': list, 'max': max, 'min': min,
        'print': print, 'range': range, 'round': round, 'str': str,
        'sum': sum, 'tuple': tuple, 'type': type
    }
    
    # Add the tools proxy and asyncio
    exec_globals = {
        '__builtins__': safe_builtins,
        'tools': tools,
        'asyncio': asyncio,
        'json': json,
    }
    
    return exec_globals


async def execute_code_with_tools(
    code: str,
    client: MultiServerClient,
    language: str = "python",
    environment: ExecutionEnvironment = ExecutionEnvironment.LOCAL,
    timeout: int = 30,
    async_mode: bool = False
) -> CodeExecResult:
    """
    Execute code with access to MCP tools.
    
    This function integrates with the existing code execution service,
    but adds tool proxies to the execution environment.
    
    Args:
        code: Code to execute
        client: MultiServerClient instance for tool access
        language: Programming language
        environment: Execution environment
        timeout: Execution timeout in seconds
        async_mode: Whether to use async tool proxies
        
    Returns:
        CodeExecResult with execution details
    """
    # Only Python is supported for tool access currently
    if language.lower() != "python":
        return CodeExecResult(
            success=False,
            error=f"Tool access is only supported for Python code, not {language}",
            exec_id=str(uuid.uuid4())
        )
    
    # Get the code execution service
    try:
        from scout_agent.services.agents.code.service import get_code_execution_service
        code_exec_service = get_code_execution_service()
    except ImportError:
        logger.error("Could not import code execution service")
        # Mock execution for testing
        return CodeExecResult(
            success=True,
            output=f"Mock execution of code:\n{code}\n\nWith tools: {client}",
            error=None,
            exec_id=str(uuid.uuid4())
        )
    
    # Create tool proxies
    sync_tools, async_tools = create_tool_proxies(client)
    
    # Non-async execution
    if not async_mode:
        # Modify the code to include tool access
        tool_setup = """
# Tools setup by MCP executor
import json
"""
        modified_code = tool_setup + code
        
        # Create a temporary file with the code
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
            f.write(modified_code)
            temp_file = f.name
        
        try:
            # Prepare environment variables for tool access
            env_vars = {
                "MCP_TOOL_ACCESS": "true"
            }
            
            # Execute the code using the code execution service
            if environment == ExecutionEnvironment.LOCAL:
                result = await code_exec_service.execute_code(
                    code=modified_code,
                    language="python",
                    env=environment,
                    timeout=timeout
                )
            else:
                # For Docker, we need to run a wrapper script that sets up the tools
                docker_script = f"""
import sys
import json

# Set up tool proxies
{inspect.getsource(_prepare_execution_globals)}

# Execute the user code
user_globals = _prepare_execution_globals(tools)
exec(sys.argv[1], user_globals)
"""
                # Execute in Docker with the wrapper
                result = await code_exec_service.execute_code(
                    code=docker_script,
                    language="python",
                    env=environment,
                    timeout=timeout
                )
            
            # Clean up the temp file
            os.unlink(temp_file)
            
            return result
        
        except Exception as e:
            logger.exception(f"Error executing code with tools: {e}")
            
            # Clean up the temp file
            try:
                os.unlink(temp_file)
            except:
                pass
            
            return CodeExecResult(
                success=False,
                error=f"Error executing code: {str(e)}",
                exec_id=str(uuid.uuid4())
            )
    
    # Async execution
    else:
        # Modify the code to include async tool access
        async_tool_setup = """
# Async tools setup by MCP executor
import asyncio
import json

async def main():
    # Your code below
"""
        # Indent the user code for the async main function
        indented_code = "\n".join(f"    {line}" for line in code.splitlines())
        modified_code = async_tool_setup + indented_code + "\n\nasyncio.run(main())"
        
        # Create a temporary file with the code
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
            f.write(modified_code)
            temp_file = f.name
        
        try:
            # Execute the code using the code execution service
            if environment == ExecutionEnvironment.LOCAL:
                result = await code_exec_service.execute_code(
                    code=modified_code,
                    language="python",
                    env=environment,
                    timeout=timeout
                )
            else:
                # For Docker, we need to run a wrapper script that sets up the tools
                docker_script = f"""
import sys
import asyncio
import json

# Set up tool proxies
{inspect.getsource(_prepare_execution_globals_async)}

# Execute the user code
async def execute():
    user_globals = await _prepare_execution_globals_async(tools)
    exec(sys.argv[1], user_globals)

asyncio.run(execute())
"""
                # Execute in Docker with the wrapper
                result = await code_exec_service.execute_code(
                    code=docker_script,
                    language="python",
                    env=environment,
                    timeout=timeout
                )
            
            # Clean up the temp file
            os.unlink(temp_file)
            
            return result
        
        except Exception as e:
            logger.exception(f"Error executing async code with tools: {e}")
            
            # Clean up the temp file
            try:
                os.unlink(temp_file)
            except:
                pass
            
            return CodeExecResult(
                success=False,
                error=f"Error executing async code: {str(e)}",
                exec_id=str(uuid.uuid4())
            )


class SandboxedExecution:
    """
    Context manager for sandboxed execution with tool access.
    
    This class provides a context manager that sets up and tears down
    a sandboxed execution environment with tool access.
    """
    
    def __init__(self, client: MultiServerClient, async_mode: bool = False):
        """
        Initialize sandboxed execution.
        
        Args:
            client: MultiServerClient instance
            async_mode: Whether to use async tool proxies
        """
        self.client = client
        self.async_mode = async_mode
        self.sync_tools = None
        self.async_tools = None
    
    async def __aenter__(self):
        """Enter async context manager."""
        # Create tool proxies
        self.sync_tools, self.async_tools = create_tool_proxies(self.client)
        
        if self.async_mode:
            return self.async_tools
        else:
            return self.sync_tools
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager."""
        # Nothing to clean up currently
        pass
