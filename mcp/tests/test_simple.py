"""
Simple test script for MCP components.

This script tests basic functionality of MCP components without requiring
external dependencies like uvicorn or the full service infrastructure.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path to allow importing from mcp package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("mcp_simple_test")

# Import the models module
from mcp.models import (
    ServerConfig,
    Tool, 
    ToolSchema,
    ToolRequest,
    ToolResponse,
    TextContent,
    ErrorContent
)


def test_models():
    """Test basic model functionality."""
    logger.info("Testing MCP models...")
    
    # Create a server config
    config = ServerConfig(
        id="test",
        name="Test Server",
        url="http://localhost:8000",
        headers={"Authorization": "Bearer test"},
        prefix_tools=True
    )
    
    logger.info(f"Server config: {config}")
    
    # Create a tool schema
    schema = ToolSchema(
        name="test_tool",
        description="A test tool",
        input_schema={
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "A message to echo"
                }
            },
            "required": ["message"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "result": {
                    "type": "string"
                }
            }
        }
    )
    
    logger.info(f"Tool schema: {schema}")
    
    # Create a tool
    tool = Tool(
        name="test_tool",
        description="A test tool",
        inputSchema=schema,  # This needs to be inputSchema, not schema
        server_id="test"     # This is an extra field not in the model
    )
    
    logger.info(f"Tool: {tool}")
    
    # Create a tool request
    request = ToolRequest(
        request_id="123",
        name="test_tool",  # Changed from tool_name to name to match the model
        arguments={
            "message": "Hello, world!"
        }
    )
    
    logger.info(f"Tool request: {request}")
    
    # Create a successful tool response
    success_response = ToolResponse(
        # Note: ToolResponse doesn't have request_id or tool_name fields
        # It only takes a list of Content objects in the content field
        content=[TextContent(text="Hello, world!")]
    )
    
    logger.info(f"Success response: {success_response}")
    
    # Create an error tool response
    error_response = ToolResponse(
        content=[ErrorContent(message="Test error", code="500")]  # Fixed: error→message, code as string
    )
    
    logger.info(f"Error response: {error_response}")
    
    return True


def test_client_imports():
    """Test that client modules can be imported."""
    logger.info("Testing client imports...")
    
    try:
        from mcp.client.base import ClientSession
        from mcp.client.sse import sse_client
        from mcp.client.multi import MultiServerClient
        
        logger.info("Successfully imported client modules")
        return True
    except ImportError as e:
        logger.error(f"Failed to import client modules: {e}")
        return False


def test_server_imports():
    """Test that server modules can be imported."""
    logger.info("Testing server imports...")
    
    try:
        from mcp.server.base import Server, Context
        logger.info("Successfully imported server modules")
        return True
    except ImportError as e:
        logger.error(f"Failed to import server modules: {e}")
        return False


def test_sandbox_imports():
    """Test that sandbox modules can be imported."""
    logger.info("Testing sandbox imports...")
    
    try:
        from mcp.sandbox.proxy import create_tool_proxy, create_async_tool_proxy
        from mcp.sandbox.executor import execute_code_with_tools, SandboxedExecution
        
        logger.info("Successfully imported sandbox modules")
        return True
    except ImportError as e:
        logger.error(f"Failed to import sandbox modules: {e}")
        return False


def test_service_imports():
    """Test that service modules can be imported."""
    logger.info("Testing service imports...")
    
    try:
        from mcp.service import MCPToolService, get_mcp_tool_service
        
        logger.info("Successfully imported service modules")
        return True
    except ImportError as e:
        logger.error(f"Failed to import service modules: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    results = {}
    
    results["models"] = test_models()
    results["client_imports"] = test_client_imports()
    results["server_imports"] = test_server_imports()
    results["sandbox_imports"] = test_sandbox_imports()
    results["service_imports"] = test_service_imports()
    
    # Print summary
    logger.info("\n=== Test Summary ===")
    all_passed = True
    for test, result in results.items():
        status = "PASS" if result else "FAIL"
        if not result:
            all_passed = False
        logger.info(f"{test}: {status}")
    
    if all_passed:
        logger.info("\n✅ All tests passed! The MCP components can be imported correctly.")
    else:
        logger.error("\n❌ Some tests failed")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
