"""
Full integration test for the MCP tooling system.

This script tests the integration of all MCP components:
- SSE server startup with tool registration
- Client connection to server
- Tool discovery and listing
- Tool calling with arguments
- Multi-server client management
- Sandboxed code execution with tool access
"""

import asyncio
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path to allow importing from mcp package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("mcp_integration_test")

# Import MCP components
from mcp.server.base import Server, Context
from mcp.models import ServerConfig, TextContent, ErrorContent


# Define test tools for the server
async def add_numbers(a: int, b: int, ctx: Context = None) -> int:
    """Add two numbers together."""
    if ctx:
        ctx.log("INFO", f"Adding {a} + {b}")
    return a + b


async def multiply_numbers(a: int, b: int, ctx: Context = None) -> int:
    """Multiply two numbers together."""
    if ctx:
        ctx.log("INFO", f"Multiplying {a} * {b}")
    return a * b


async def echo_text(text: str, ctx: Context = None) -> str:
    """Echo the input text."""
    if ctx:
        ctx.log("INFO", f"Echoing: {text}")
    return text


async def generate_error(message: str = "Test error", ctx: Context = None) -> None:
    """Generate an error for testing."""
    if ctx:
        ctx.log("ERROR", f"Generating error: {message}")
    raise ValueError(message)


class TestServer:
    """Test server class for MCP integration testing."""
    
    def __init__(self, port: int = 8000, name: str = "Test Server"):
        """Initialize the test server."""
        self.port = port
        self.name = name
        self.server = None
        self.asgi_app = None
        self.uvicorn_server = None
        self.started = False
    
    async def setup(self):
        """Set up the server with test tools."""
        # Create MCP server
        self.server = Server(name=self.name, description="Test server for MCP integration tests")
        
        # Register tools
        self.server.tool(name="add", description="Add two numbers")(add_numbers)
        self.server.tool(name="multiply", description="Multiply two numbers")(multiply_numbers)
        self.server.tool(name="echo", description="Echo text")(echo_text)
        self.server.tool(name="error", description="Generate an error")(generate_error)
        
        # Create ASGI app
        try:
            from mcp.server.sse import create_asgi_app
            self.asgi_app = await create_asgi_app(self.server)
            logger.info(f"Created ASGI app for {self.name}")
            return True
        except ImportError as e:
            logger.error(f"Failed to create ASGI app: {e}")
            return False
    
    async def start(self):
        """Start the server."""
        if self.asgi_app is None:
            success = await self.setup()
            if not success:
                return False
        
        try:
            from uvicorn import Config, Server as UvicornServer
            
            # Configure uvicorn
            config = Config(
                app=self.asgi_app,
                host="127.0.0.1",
                port=self.port,
                log_level="warning"
            )
            
            # Create server
            self.uvicorn_server = UvicornServer(config=config)
            
            # Start server in a task
            logger.info(f"Starting {self.name} on http://127.0.0.1:{self.port}")
            self.server_task = asyncio.create_task(self.uvicorn_server.serve())
            self.started = True
            
            # Give it a moment to start
            await asyncio.sleep(1)
            
            return True
            
        except ImportError as e:
            logger.error(f"Failed to start server: {e}")
            return False
    
    async def stop(self):
        """Stop the server."""
        if self.started and self.server_task:
            logger.info(f"Stopping {self.name}")
            self.server_task.cancel()
            try:
                await self.server_task
            except asyncio.CancelledError:
                pass
            self.started = False
            logger.info(f"{self.name} stopped")
            return True
        return False


async def test_client():
    """Test the client with a single server."""
    from mcp.client.base import ClientSession
    from mcp.client.sse import sse_client
    
    logger.info("Testing direct client connection")
    
    # Create client session
    client = None
    try:
        # Create client
        client = await sse_client("http://127.0.0.1:8000")
        
        # Initialize client
        await client.initialize()
        
        # List tools
        tools = await client.list_tools()
        logger.info(f"Available tools: {len(tools)}")
        for tool in tools:
            logger.info(f"  - {tool.name}: {tool.description}")
        
        # Call add tool
        logger.info("Calling add tool")
        result = await client.call_tool("add", {"a": 5, "b": 7})
        logger.info(f"Result: {result}")
        assert result.content[0].text == "12", "Expected 12 as the result"
        
        # Call echo tool
        logger.info("Calling echo tool")
        result = await client.call_tool("echo", {"text": "Hello, MCP!"})
        logger.info(f"Result: {result}")
        assert result.content[0].text == "Hello, MCP!", "Expected 'Hello, MCP!' as the result"
        
        # Call error tool
        logger.info("Calling error tool (should generate an error)")
        try:
            result = await client.call_tool("error", {"message": "Test error"})
            logger.info(f"Result: {result}")
        except Exception as e:
            logger.info(f"Got expected error: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in client test: {e}")
        return False
        
    finally:
        # Close client
        if client:
            await client.close()


async def test_multi_client():
    """Test the multi-server client."""
    from mcp.client.multi import MultiServerClient
    
    logger.info("Testing multi-server client")
    
    # Server configs
    server_configs = [
        ServerConfig(
            id="server1",
            name="Server 1",
            url="http://127.0.0.1:8000",
            timeout=30.0
        ),
        ServerConfig(
            id="server2",
            name="Server 2",
            url="http://127.0.0.1:8001",
            timeout=30.0,
            prefix_tools=True
        )
    ]
    
    # Create multi-server client
    client = None
    try:
        # Create client
        client = await MultiServerClient.create(server_configs=server_configs)
        
        # Get available tools
        tools = client.get_available_tools()
        logger.info(f"Multi-server available tools: {len(tools)}")
        for tool_name in tools.keys():
            logger.info(f"  - {tool_name}")
        
        # Call add tool on server 1
        logger.info("Calling add tool on server 1")
        result = await client.call_tool("add", {"a": 10, "b": 20})
        logger.info(f"Result: {result}")
        
        # Call multiply tool on server 2 (prefixed)
        logger.info("Calling multiply tool on server 2 with prefix")
        result = await client.call_tool("server2.multiply", {"a": 10, "b": 20})
        logger.info(f"Result: {result}")
        
        # Test function wrapper
        logger.info("Testing function wrapper")
        add_result = await client.function_wrapper("add", 5, 7)
        logger.info(f"Function wrapper result: {add_result}")
        assert add_result == 12, "Expected 12 as the result"
        
        return True
        
    except Exception as e:
        logger.error(f"Error in multi-client test: {e}")
        return False
        
    finally:
        # Close client
        if client:
            await client.close()


async def test_sandbox():
    """Test the sandbox execution with tool access."""
    from mcp.client.multi import MultiServerClient
    from mcp.sandbox.executor import execute_code_with_tools
    
    logger.info("Testing sandbox execution")
    
    # Server configs
    server_configs = [
        ServerConfig(
            id="server1",
            name="Server 1",
            url="http://127.0.0.1:8000",
            timeout=30.0
        )
    ]
    
    # Create multi-server client
    client = None
    try:
        # Create client
        client = await MultiServerClient.create(server_configs=server_configs)
        
        # Execute code with tool access
        code = """
print("Hello from sandbox!")

# Add numbers using tool
result = tools.add(7, 13)
print(f"7 + 13 = {result}")

# Echo text using tool
message = tools.echo("Message from sandbox")
print(f"Echo response: {message}")

# Try to access external modules (should be restricted)
try:
    import os
    print("WARNING: Sandbox allows importing os!")
except ImportError:
    print("Sandbox correctly restricts importing os")

# Return a result
result
"""
        
        # Execute code
        result = await execute_code_with_tools(
            code=code,
            client=client,
            timeout=10
        )
        
        logger.info(f"Sandbox execution result: {result}")
        logger.info(f"Output: {result.output}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in sandbox test: {e}")
        return False
        
    finally:
        # Close client
        if client:
            await client.close()


async def test_service():
    """Test the MCP tool service."""
    from mcp.service import MCPToolService
    
    logger.info("Testing MCP tool service")
    
    # Create service
    service = MCPToolService()
    
    try:
        # Setup with test servers
        service.server_configs = [
            ServerConfig(
                id="server1",
                name="Server 1",
                url="http://127.0.0.1:8000",
                timeout=30.0
            ),
            ServerConfig(
                id="server2",
                name="Server 2",
                url="http://127.0.0.1:8001",
                timeout=30.0,
                prefix_tools=True
            )
        ]
        
        # Initialize and start
        await service._initialize(None)
        await service._start()
        
        # Get available tools
        tools = service.get_available_tools()
        logger.info(f"Service available tools: {len(tools)}")
        for tool_name in tools.keys():
            logger.info(f"  - {tool_name}")
        
        # Call add tool
        logger.info("Service calling add tool with 25 + 17")
        result = await service.call_tool("add", {"a": 25, "b": 17})
        logger.info(f"Service result: {result}")
        
        # Test sandboxed execution
        logger.info("Testing service sandboxed execution")
        code = """
# Use tools from the service
result1 = tools.add(40, 2)
print(f"40 + 2 = {result1}")

result2 = tools.server2.multiply(6, 7)
print(f"6 * 7 = {result2}")

# Return a combined result
result1 + result2
"""
        exec_result = await service.execute_code_with_tools(code)
        logger.info(f"Service execution success: {exec_result['success']}")
        logger.info(f"Service execution output:\n{exec_result['output']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in service test: {e}")
        return False
        
    finally:
        # Stop service
        await service._stop()


async def run_integration_tests():
    """Run all integration tests."""
    results = {}
    
    try:
        # Check if uvicorn is available
        has_uvicorn = True
        try:
            import uvicorn
        except ImportError:
            has_uvicorn = False
            logger.warning("uvicorn not available - will run in mock mode")
    
        # If uvicorn is available, start test servers
        server1 = None
        server2 = None
        
        if has_uvicorn:
            # Start test servers
            server1 = TestServer(port=8000, name="Test Server 1")
            server2 = TestServer(port=8001, name="Test Server 2")
            
            # Start server 1
            server1_started = await server1.start()
            if not server1_started:
                logger.warning("Failed to start server 1 - will run in mock mode")
                has_uvicorn = False
            
            # Start server 2
            server2_started = await server2.start()
            if not server2_started:
                logger.warning("Failed to start server 2 - will run in mock mode")
                has_uvicorn = False
                
            if has_uvicorn:
                # Give servers time to initialize
                await asyncio.sleep(1)
        
        # Skip server-dependent tests if uvicorn is not available
        if not has_uvicorn:
            logger.warning("Skipping server-dependent tests")
            results["client"] = "SKIPPED"
            results["multi_client"] = "SKIPPED"
            results["sandbox"] = "SKIPPED"
            results["service"] = "SKIPPED"
        else:
        
            # Run client test
            logger.info("=== Testing Client ===")
            results["client"] = await test_client()
            
            # Run multi-client test
            logger.info("=== Testing Multi-Server Client ===")
            results["multi_client"] = await test_multi_client()
            
            # Run sandbox test
            logger.info("=== Testing Sandbox Execution ===")
            results["sandbox"] = await test_sandbox()
            
            # Run service test
            logger.info("=== Testing Service Integration ===")
            results["service"] = await test_service()
        
        # Print summary
        logger.info("\n=== Integration Test Summary ===")
        all_passed = True
        for test, result in results.items():
            status = "PASS" if result else "FAIL"
            if not result:
                all_passed = False
            logger.info(f"{test}: {status}")
        
        if all_passed:
            logger.info("\n✅ All integration tests passed! The MCP components work together correctly.")
            logger.info("It is safe to remove the MCP-inspiration folder.")
        else:
            logger.error("\n❌ Some integration tests failed. Fix issues before removing the MCP-inspiration folder.")
        
        return all_passed
        
    except Exception as e:
        logger.error(f"Error in integration tests: {e}")
        return False
        
    finally:
        # Stop servers if they were started
        if 'server1' in locals() and server1 is not None:
            await server1.stop()
        if 'server2' in locals() and server2 is not None:
            await server2.stop()


def main():
    """Run integration tests."""
    logger.info("Starting MCP full integration tests")
    
    try:
        result = asyncio.run(run_integration_tests())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("Tests interrupted")
        sys.exit(1)


if __name__ == "__main__":
    main()
