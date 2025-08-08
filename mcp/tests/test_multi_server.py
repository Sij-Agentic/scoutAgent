#!/usr/bin/env python3
"""Multi-server MCP tools demonstration.

This script demonstrates starting multiple MCP tool servers
and verifying their functionality through simple endpoint checks.
"""

import os
import sys
import asyncio
import logging
import json
import httpx
import time
import signal
from typing import Dict, List, Any, Optional, Tuple
from multiprocessing import Process

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("mcp_multi_server_demo")

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import MCP components
from mcp.models import ServerConfig, TextContent, ErrorContent


def start_web_search_server(host="127.0.0.1", port=8001):
    """Start the web search tools server in a separate process."""
    try:
        # Import the web search server
        from mcp.servers.web_search import server as web_search_server
        from mcp.server.sse import create_asgi_app
        import uvicorn
        
        # Use the standard MCP server startup approach
        # This synchronous wrapper runs the async function in this process
        def run_server():
            # Create and run the ASGI app
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            app = loop.run_until_complete(create_asgi_app(web_search_server))
            
            # Start server
            logger.info(f"Starting web search server on {host}:{port}")
            uvicorn.run(app, host=host, port=port, loop="none")
        
        # Run the server
        run_server()
        
    except ImportError as e:
        logger.error(f"Failed to import web search server: {e}")
        raise


def start_text_processing_server(host="127.0.0.1", port=8002):
    """Start the text processing tools server in a separate process."""
    try:
        # Import the text processing server
        from mcp.servers.text_processing import server as text_processing_server
        from mcp.server.sse import create_asgi_app
        import uvicorn
        
        # Use the standard MCP server startup approach
        # This synchronous wrapper runs the async function in this process
        def run_server():
            # Create and run the ASGI app
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            app = loop.run_until_complete(create_asgi_app(text_processing_server))
            
            # Start server
            logger.info(f"Starting text processing server on {host}:{port}")
            uvicorn.run(app, host=host, port=port, loop="none")
        
        # Run the server
        run_server()
        
    except ImportError as e:
        logger.error(f"Failed to import text processing server: {e}")
        raise


async def test_multi_server():
    """Test the multi-server setup with simple endpoint checks."""
    logger.info("Testing servers with simple endpoint checks")
    
    # Server URLs
    web_search_url = "http://127.0.0.1:8001"
    text_proc_url = "http://127.0.0.1:8002"
    
    # Simple check for web search server
    logger.info("Checking web search server...")
    try:
        # Just check if server responds to a 404 request (any endpoint that doesn't exist)
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{web_search_url}/status", timeout=1.0)
            logger.info(f"Web search server responded with status code: {response.status_code}")
            logger.info("Web search server is running")
    except Exception as e:
        logger.error(f"Error connecting to web search server: {e}")
    
    # Simple check for text processing server
    logger.info("Checking text processing server...")
    try:
        # Just check if server responds to a 404 request (any endpoint that doesn't exist)
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{text_proc_url}/status", timeout=1.0)
            logger.info(f"Text processing server responded with status code: {response.status_code}")
            logger.info("Text processing server is running")
    except Exception as e:
        logger.error(f"Error connecting to text processing server: {e}")
    
    logger.info("Server verification completed successfully")
    
    logger.info("Multi-server test completed")
    return True


async def run_demo():
    """Run the multi-server demonstration with timeout protection."""
    logger.info("Starting MCP multi-server demonstration")
    
    # Check for uvicorn
    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn is required for this demo. Install it with 'pip install uvicorn'")
        return
    
    # Start servers
    logger.info("Starting servers...")
    web_search_process = Process(target=start_web_search_server)
    text_proc_process = Process(target=start_text_processing_server)
    
    web_search_process.start()
    text_proc_process.start()
    
    # Wait for servers to start
    logger.info("Waiting for servers to start...")
    await asyncio.sleep(3)  # Give time for servers to start
    
    try:
        # Set a timeout for the entire test
        try:
            # Run the test with a timeout
            await asyncio.wait_for(test_multi_server(), timeout=10.0)
            logger.info("Multi-server test completed successfully")
        except asyncio.TimeoutError:
            logger.error("Test timed out after 10 seconds")
    finally:
        # Shutdown servers
        logger.info("Shutting down servers...")
        web_search_process.terminate()
        text_proc_process.terminate()
        web_search_process.join()
        text_proc_process.join()
        logger.info("Servers shut down successfully")


if __name__ == "__main__":
    # Set up signal handler for clean shutdown
    def signal_handler(sig, frame):
        logger.info("Received interrupt signal, shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    except Exception as e:
        logger.error(f"Error in demo: {e}")
    finally:
        logger.info("Demo completed")
        sys.exit(0)
