#!/usr/bin/env python3
"""
MCP Client Demo

This script demonstrates how to connect to a running MCP server,
query available tools, and call tools.

Usage:
    1. First start an MCP server using the run_servers.py script:
       python -m mcp.run_servers --server web_search
       
    2. Then run this script in another terminal:
       python -m mcp.tests.test_client_demo
"""

import os
import sys
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("mcp_client_demo")

# Add parent directory to Python path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import MCP components
from mcp.client.sse import sse_client
from mcp.models import Tool, ToolResponse, Content, TextContent


async def print_tool_details(tool: Tool) -> None:
    """
    Print details about a tool.
    
    Args:
        tool: Tool object
    """
    print(f"\n{'=' * 40}")
    print(f"Tool: {tool.name}")
    print(f"Description: {tool.description}")
    print(f"Schema: {json.dumps(tool.schema.dict(), indent=2)}")
    print(f"{'=' * 40}\n")


async def format_tool_response(response: ToolResponse) -> None:
    """
    Format and print a tool response.
    
    Args:
        response: Tool response object
    """
    print(f"\n{'=' * 40}")
    print(f"Tool Response (status: {response.status}):")
    
    for content_item in response.content:
        if isinstance(content_item, TextContent):
            print(f"Text: {content_item.text}")
        else:
            print(f"Content: {content_item.dict()}")
    
    if response.error:
        print(f"Error: {response.error}")
    
    print(f"{'=' * 40}\n")


async def demo_web_search_server() -> None:
    """
    Demonstrate connecting to the web search server and using its tools.
    """
    server_url = "http://127.0.0.1:8001"
    logger.info(f"Connecting to MCP server at {server_url}")
    
    try:
        # Connect to the SSE server
        async with await sse_client(f"{server_url}/sse") as session:
            # Initialize the session
            server_info = await session.initialize()
            logger.info(f"Connected to server: {server_info.name} (v{server_info.version})")
            logger.info(f"Server description: {server_info.description}")
            logger.info(f"Server capabilities: {', '.join(server_info.capabilities)}")
            
            # List available tools
            logger.info("Querying available tools...")
            tools_response = await session.list_tools()
            
            if not tools_response.tools:
                logger.warning("No tools available on this server")
                return
            
            logger.info(f"Found {len(tools_response.tools)} available tools")
            
            # Print details for each tool
            for tool in tools_response.tools:
                await print_tool_details(tool)
            
            # Demo calling the search_web tool if available
            search_tool = session.get_tool("search_web")
            if search_tool:
                logger.info("Calling search_web tool...")
                
                # Call the tool with a test query
                response = await session.call_tool(
                    "search_web", 
                    {"query": "What is MCP protocol"}
                )
                
                # Print the response
                await format_tool_response(response)
            else:
                logger.warning("search_web tool not available")
            
            # Demo calling the reddit_thread_fetcher tool if available
            reddit_tool = session.get_tool("reddit_thread_fetcher")
            if reddit_tool:
                logger.info("Calling reddit_thread_fetcher tool...")
                
                # Call the tool with a test thread ID
                response = await session.call_tool(
                    "reddit_thread_fetcher", 
                    {"thread_id": "t3_15jxbqo"}  # Example thread ID
                )
                
                # Print the response
                await format_tool_response(response)
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
        # Connect to the SSE server
        async with await sse_client(f"{server_url}/sse") as session:
            # Initialize the session
            server_info = await session.initialize()
            logger.info(f"Connected to server: {server_info.name} (v{server_info.version})")
            
            # List available tools
            logger.info("Querying available tools...")
            tools_response = await session.list_tools()
            
            if not tools_response.tools:
                logger.warning("No tools available on this server")
                return
            
            logger.info(f"Found {len(tools_response.tools)} available tools")
            
            # Print details for each tool
            for tool in tools_response.tools:
                await print_tool_details(tool)
            
            # Demo calling the extract_text_from_html tool if available
            extract_tool = session.get_tool("extract_text_from_html")
            if extract_tool:
                logger.info("Calling extract_text_from_html tool...")
                
                # Call the tool with a test HTML
                response = await session.call_tool(
                    "extract_text_from_html", 
                    {"html": "<html><body><h1>Hello World</h1><p>This is a test.</p></body></html>"}
                )
                
                # Print the response
                await format_tool_response(response)
            else:
                logger.warning("extract_text_from_html tool not available")
            
            # Demo calling the summarize_text tool if available
            summarize_tool = session.get_tool("summarize_text")
            if summarize_tool:
                logger.info("Calling summarize_text tool...")
                
                # Call the tool with a test text
                response = await session.call_tool(
                    "summarize_text", 
                    {"text": "The Model Context Protocol (MCP) is a protocol for communication between language models and tools. It enables models to discover, understand, and use tools via a standardized interface. MCP helps create a consistent experience across different models and tools, making it easier to build and maintain AI applications."}
                )
                
                # Print the response
                await format_tool_response(response)
            else:
                logger.warning("summarize_text tool not available")
    
    except Exception as e:
        logger.error(f"Error connecting to server: {e}")


async def main() -> None:
    """Main entry point for the script."""
    print("\n" + "=" * 80)
    print("MCP CLIENT DEMONSTRATION".center(80))
    print("=" * 80 + "\n")
    
    print("This script demonstrates connecting to running MCP servers,")
    print("querying available tools, and calling tools.\n")
    
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
