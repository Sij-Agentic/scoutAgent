"""
Test script for MCP integration.

This script tests the integration of MCP components:
- Server with tool registration
- SSE transport
- Client connectivity
- Tool discovery and calling
- Sandboxed execution with tool access
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Configure environment for testing
os.environ["TESTING"] = "1"

# Add parent directory to path to allow importing from mcp package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.models import ServerInfo, Tool, ToolSchema
from mcp.server.base import Server, Context
from mcp.client.multi import MultiServerClient
from mcp.models import ServerConfig
from mcp.service import MCPToolService
from mcp.sandbox.executor import execute_code_with_tools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("mcp_test")


# Define some example tools
async def add_numbers(a: int, b: int, ctx: Context = None) -> int:
    """Add two numbers together."""
    if ctx:
        ctx.log("INFO", f"Adding {a} + {b}")
    return a + b


async def echo_text(text: str, ctx: Context = None) -> str:
    """Echo the input text."""
    if ctx:
        ctx.log("INFO", f"Echoing: {text}")
    return text


async def run_sse_server():
    """Run an SSE server with example tools."""
    try:
        from uvicorn import Config, Server as UvicornServer
        from mcp.server.sse import create_asgi_app
        
        # Create MCP server
        server = Server(name="Test MCP Server", description="Test server with example tools")
        
        # Register tools
        server.tool(name="add", description="Add two numbers")(add_numbers)
        server.tool(name="echo", description="Echo text")(echo_text)
        
        # Create ASGI app
        app = await create_asgi_app(server)
        
        # Run with uvicorn
        config = Config(app=app, host="127.0.0.1", port=8000, log_level="info")
        uvicorn_server = UvicornServer(config=config)
        
        logger.info("Starting SSE server on http://127.0.0.1:8000")
        await uvicorn_server.serve()
    except ImportError as e:
        logger.error(f"Could not start SSE server due to missing dependencies: {e}")
        logger.info("Running in mock mode for testing API interfaces only")
        # Just sleep to simulate server running
        await asyncio.sleep(60)


async def test_client_direct():
    """Test direct client connectivity."""
    # Configure client to connect to local server
    server_config = ServerConfig(
        id="test_server",
        name="Test Server",
        url="http://127.0.0.1:8000",
        timeout=30.0
    )
    
    # Create client
    logger.info("Creating multi-server client")
    client = await MultiServerClient.create(server_configs=[server_config])
    
    try:
        # Get available tools
        tools = client.get_available_tools()
        logger.info(f"Available tools: {list(tools.keys())}")
        
        # Call add tool
        logger.info("Calling add tool with 5 + 7")
        result = await client.call_tool("add", {"a": 5, "b": 7})
        logger.info(f"Result: {result}")
        
        # Call echo tool
        logger.info("Calling echo tool with 'Hello, MCP!'")
        result = await client.call_tool("echo", {"text": "Hello, MCP!"})
        logger.info(f"Result: {result}")
        
        # Test function wrapper
        logger.info("Testing function wrapper")
        result = await client.function_wrapper("add", 10, 20)
        logger.info(f"Function wrapper result: {result}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error in client test: {e}")
        return False
    
    finally:
        # Close client
        await client.close()


async def test_service_integration():
    """Test service integration."""
    # Create service
    service = MCPToolService()
    
    try:
        # Setup directly
        service.setup_direct()
        
        # Initialize and start
        await service._initialize(None)
        await service._start()
        
        # Get available tools
        tools = service.get_available_tools()
        logger.info(f"Service available tools: {list(tools.keys())}")
        
        # Call add tool
        logger.info("Service calling add tool with 12 + 34")
        result = await service.call_tool("add", {"a": 12, "b": 34})
        logger.info(f"Service result: {result}")
        
        # Test sandboxed execution
        logger.info("Testing sandboxed execution")
        code = """
print("Sandboxed execution with tool access")
result = tools.add(40, 2)
print(f"40 + 2 = {result}")
message = tools.echo("Message from sandbox")
print(f"Echo response: {message}")
"""
        exec_result = await service.execute_code_with_tools(code)
        logger.info(f"Execution success: {exec_result['success']}")
        logger.info(f"Execution output:\n{exec_result['output']}")
        
        if exec_result.get("error"):
            logger.error(f"Execution error: {exec_result['error']}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error in service test: {e}")
        return False
    
    finally:
        # Stop service
        await service._stop()


async def run_tests():
    """Run all tests."""
    # Start server in background
    server_task = asyncio.create_task(run_sse_server())
    
    # Wait for server to start
    await asyncio.sleep(2)
    
    try:
        # Run client test
        logger.info("=== Testing Client ===")
        client_success = await test_client_direct()
        logger.info(f"Client test {'passed' if client_success else 'failed'}")
        
        # Run service test
        logger.info("=== Testing Service Integration ===")
        service_success = await test_service_integration()
        logger.info(f"Service test {'passed' if service_success else 'failed'}")
        
        # Overall result
        if client_success and service_success:
            logger.info("✅ All tests passed!")
            return True
        else:
            logger.error("❌ Some tests failed")
            return False
    
    finally:
        # Cancel server
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


def main():
    """Run tests."""
    logger.info("Starting MCP integration tests")
    
    try:
        result = asyncio.run(run_tests())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("Tests interrupted")
        sys.exit(1)


if __name__ == "__main__":
    main()
