#!/usr/bin/env python3
"""
MCP Server Runner

This script allows running individual MCP servers or all servers at once.
It supports running the web search server, text processing server, or both.
"""

import os
import sys
import asyncio
import argparse
import logging
import signal
import uvicorn
from multiprocessing import Process
from typing import List, Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("mcp_server_runner")

# Add parent directory to Python path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import MCP components
from mcp.server.sse import create_asgi_app


def run_web_search_server(port: int = 8001) -> None:
    """
    Run the web search server.
    
    Args:
        port: Port to run the server on
    """
    from mcp.servers.web_search import server
    
    logger.info(f"Starting web search server on 127.0.0.1:{port}")
    app = asyncio.run(create_asgi_app(server))
    uvicorn.run(app, host="127.0.0.1", port=port)


def run_text_processing_server(port: int = 8002) -> None:
    """
    Run the text processing server.
    
    Args:
        port: Port to run the server on
    """
    from mcp.servers.text_processing import server
    
    logger.info(f"Starting text processing server on 127.0.0.1:{port}")
    app = asyncio.run(create_asgi_app(server))
    uvicorn.run(app, host="127.0.0.1", port=port)


def run_all_servers(web_search_port: int = 8001, text_proc_port: int = 8002) -> None:
    """
    Run all servers in separate processes.
    
    Args:
        web_search_port: Port for the web search server
        text_proc_port: Port for the text processing server
    """
    # Start servers in separate processes
    web_search_process = Process(target=run_web_search_server, args=(web_search_port,))
    text_proc_process = Process(target=run_text_processing_server, args=(text_proc_port,))
    
    processes = [web_search_process, text_proc_process]
    
    # Set up signal handler for clean shutdown
    def signal_handler(sig, frame):
        logger.info("Received interrupt signal, shutting down servers...")
        for process in processes:
            if process.is_alive():
                process.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        logger.info("Starting all MCP servers...")
        for process in processes:
            process.start()
        
        # Keep the main process running
        logger.info("All servers started. Press Ctrl+C to stop.")
        for process in processes:
            process.join()
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down servers...")
    
    finally:
        # Ensure all processes are terminated
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
        
        logger.info("All servers shut down.")


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run MCP servers")
    
    parser.add_argument(
        "--server", 
        choices=["web_search", "text_processing", "all"], 
        default="all",
        help="Server to run (default: all)"
    )
    
    parser.add_argument(
        "--web-search-port", 
        type=int, 
        default=8001, 
        help="Port for web search server (default: 8001)"
    )
    
    parser.add_argument(
        "--text-proc-port", 
        type=int, 
        default=8002, 
        help="Port for text processing server (default: 8002)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.server == "all":
            run_all_servers(args.web_search_port, args.text_proc_port)
        elif args.server == "web_search":
            run_web_search_server(args.web_search_port)
        elif args.server == "text_processing":
            run_text_processing_server(args.text_proc_port)
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    
    except Exception as e:
        logger.error(f"Error running server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
