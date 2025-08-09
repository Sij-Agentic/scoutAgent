#!/usr/bin/env python3
"""
MCP Direct HTTP Client Demo

This script demonstrates how to connect to a running MCP server using direct HTTP requests,
query available tools, and call tools without using the SSE client implementation.

Usage:
    1. First start an MCP server using the run_servers.py script:
       python -m mcp.run_servers --server web_search
       
    2. Then run this script in another terminal:
       python -m mcp.tests.test_direct_client
"""

import os
import sys
import asyncio
import json
import logging
import httpx
from typing import Dict, Any, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("mcp_direct_client")

# Add parent directory to Python path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import MCP models
from mcp.models import Tool, ToolResponse, Content, TextContent


async def query_tools(server_url: str) -> List[Dict[str, Any]]:
    """
    Query available tools from the server using direct HTTP requests.
    
    Args:
        server_url: Base URL of the MCP server
        
    Returns:
        List of tool definitions
    """
    logger.info(f"Querying tools from {server_url}")
    
    # Create a request to list tools
    async with httpx.AsyncClient() as client:
        try:
            # Send a GET request to the tools endpoint
            response = await client.get(f"{server_url}/tools")
            response.raise_for_status()
            
            # Parse the response
            tools_data = response.json()
            logger.info(f"Found {len(tools_data)} tools")
            
            return tools_data
        
        except httpx.HTTPError as e:
            logger.error(f"HTTP error querying tools: {e}")
            
            # Try alternative approach - use SSE endpoint with a special message
            try:
                logger.info("Trying alternative approach via SSE endpoint...")
                
                # Connect to SSE endpoint
                sse_response = await client.get(f"{server_url}/sse", timeout=5.0)
                sse_response.raise_for_status()
                
                # Send a message to request tools
                message = {
                    "type": "list_tools"
                }
                
                message_response = await client.post(
                    f"{server_url}/messages",
                    json={"message": json.dumps(message)},
                    timeout=5.0
                )
                
                if message_response.status_code == 200:
                    logger.info("Successfully sent tool request via messages endpoint")
                else:
                    logger.error(f"Failed to send tool request: {message_response.status_code}")
                
                # We would need to parse SSE events to get the response
                # This is simplified for demonstration
                return []
            
            except Exception as e2:
                logger.error(f"Alternative approach failed: {e2}")
                return []


async def call_tool(server_url: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call a tool on the server using direct HTTP requests.
    
    Args:
        server_url: Base URL of the MCP server
        tool_name: Name of the tool to call
        arguments: Tool arguments
        
    Returns:
        Tool response
    """
    logger.info(f"Calling tool {tool_name} on {server_url}")
    
    # Create a request to call the tool
    async with httpx.AsyncClient() as client:
        try:
            # Send a POST request to the tools endpoint
            response = await client.post(
                f"{server_url}/tools/{tool_name}",
                json=arguments
            )
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            logger.info(f"Tool response: {result}")
            
            return result
        
        except httpx.HTTPError as e:
            logger.error(f"HTTP error calling tool: {e}")
            
            # Try alternative approach - use SSE endpoint with a special message
            try:
                logger.info("Trying alternative approach via SSE endpoint...")
                
                # Connect to SSE endpoint
                sse_response = await client.get(f"{server_url}/sse", timeout=5.0)
                sse_response.raise_for_status()
                
                # Send a message to call the tool
                message = {
                    "type": "tool_request",
                    "request_id": "direct-client-request",
                    "name": tool_name,
                    "arguments": arguments
                }
                
                message_response = await client.post(
                    f"{server_url}/messages",
                    json={"message": json.dumps(message)},
                    timeout=5.0
                )
                
                if message_response.status_code == 200:
                    logger.info("Successfully sent tool request via messages endpoint")
                else:
                    logger.error(f"Failed to send tool request: {message_response.status_code}")
                
                # We would need to parse SSE events to get the response
                # This is simplified for demonstration
                return {"status": "pending", "message": "Request sent via SSE"}
            
            except Exception as e2:
                logger.error(f"Alternative approach failed: {e2}")
                return {"error": str(e)}


async def demo_web_search_server() -> None:
    """
    Demonstrate connecting to the web search server and using its tools.
    """
    server_url = "http://127.0.0.1:8001"
    logger.info(f"Connecting to MCP server at {server_url}")
    
    try:
        # Query available tools
        tools = await query_tools(server_url)
        
        if not tools:
            logger.warning("No tools available on this server")
            return
        
        # Print tool details
        for tool in tools:
            print(f"\n{'=' * 40}")
            print(f"Tool: {tool.get('name')}")
            print(f"Description: {tool.get('description')}")
            print(f"Schema: {json.dumps(tool.get('schema', {}), indent=2)}")
            print(f"{'=' * 40}\n")
        
        # Call the search_web tool if available
        search_tool = next((t for t in tools if t.get("name") == "search_web"), None)
        if search_tool:
            logger.info("Calling search_web tool...")
            
            # Call the tool with a test query
            response = await call_tool(
                server_url,
                "search_web",
                {"query": "What is MCP protocol"}
            )
            
            # Print the response
            print(f"\n{'=' * 40}")
            print(f"Tool Response:")
            print(json.dumps(response, indent=2))
            print(f"{'=' * 40}\n")
        else:
            logger.warning("search_web tool not available")
        
        # Call the reddit_thread_fetcher tool if available
        reddit_tool = next((t for t in tools if t.get("name") == "reddit_thread_fetcher"), None)
        if reddit_tool:
            logger.info("Calling reddit_thread_fetcher tool...")
            
            # Call the tool with a test thread ID
            response = await call_tool(
                server_url,
                "reddit_thread_fetcher",
                {"thread_id": "t3_15jxbqo"}  # Example thread ID
            )
            
            # Print the response
            print(f"\n{'=' * 40}")
            print(f"Tool Response:")
            print(json.dumps(response, indent=2))
            print(f"{'=' * 40}\n")
        else:
            logger.warning("reddit_thread_fetcher tool not available")
    
    except Exception as e:
        logger.error(f"Error connecting to server: {e}")


async def demo_text_processing_server() -> None:
    """
    Demonstrate connecting to the text processing server and using its tools.
    """
    server_url = "http://127.0.0.1:8002"
    logger.info(f"Connecting to MCP server at {server_url}")
    
    try:
        # Query available tools
        tools = await query_tools(server_url)
        
        if not tools:
            logger.warning("No tools available on this server")
            return
        
        # Print tool details
        for tool in tools:
            print(f"\n{'=' * 40}")
            print(f"Tool: {tool.get('name')}")
            print(f"Description: {tool.get('description')}")
            print(f"Schema: {json.dumps(tool.get('schema', {}), indent=2)}")
            print(f"{'=' * 40}\n")
        
        # Call the extract_text_from_html tool if available
        extract_tool = next((t for t in tools if t.get("name") == "extract_text_from_html"), None)
        if extract_tool:
            logger.info("Calling extract_text_from_html tool...")
            
            # Call the tool with a test HTML
            response = await call_tool(
                server_url,
                "extract_text_from_html",
                {"html": "<html><body><h1>Hello World</h1><p>This is a test.</p></body></html>"}
            )
            
            # Print the response
            print(f"\n{'=' * 40}")
            print(f"Tool Response:")
            print(json.dumps(response, indent=2))
            print(f"{'=' * 40}\n")
        else:
            logger.warning("extract_text_from_html tool not available")
        
        # Call the summarize_text tool if available
        summarize_tool = next((t for t in tools if t.get("name") == "summarize_text"), None)
        if summarize_tool:
            logger.info("Calling summarize_text tool...")
            
            # Call the tool with a test text
            response = await call_tool(
                server_url,
                "summarize_text",
                {"text": "The Model Context Protocol (MCP) is a protocol for communication between language models and tools. It enables models to discover, understand, and use tools via a standardized interface. MCP helps create a consistent experience across different models and tools, making it easier to build and maintain AI applications."}
            )
            
            # Print the response
            print(f"\n{'=' * 40}")
            print(f"Tool Response:")
            print(json.dumps(response, indent=2))
            print(f"{'=' * 40}\n")
        else:
            logger.warning("summarize_text tool not available")
    
    except Exception as e:
        logger.error(f"Error connecting to server: {e}")


async def main() -> None:
    """Main entry point for the script."""
    print("\n" + "=" * 80)
    print("MCP DIRECT HTTP CLIENT DEMONSTRATION".center(80))
    print("=" * 80 + "\n")
    
    print("This script demonstrates connecting to running MCP servers,")
    print("querying available tools, and calling tools using direct HTTP requests.\n")
    
    # Try to connect to the web search server
    print("\n" + "-" * 80)
    print("DEMO 1: WEB SEARCH SERVER".center(80))
    print("-" * 80 + "\n")
    await demo_web_search_server()
    
    # Try to connect to the text processing server
    print("\n" + "-" * 80)
    print("DEMO 2: TEXT PROCESSING SERVER".center(80))
    print("-" * 80 + "\n")
    await demo_text_processing_server()
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE".center(80))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        logger.exception(f"Unhandled error: {e}")
