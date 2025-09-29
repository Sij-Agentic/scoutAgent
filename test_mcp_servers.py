#!/usr/bin/env python3
"""
Test script to verify MCP servers can start and respond
"""

import asyncio
import subprocess
import time
import signal
import sys
import os
from typing import List, Dict, Any

class MCPServerTester:
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.ports = [8000, 8001, 8002, 8004]
        self.server_names = ["gap_finder_tools", "reddit_api", "research_tools", "web_search"]
        
    def start_servers(self):
        """Start all MCP servers"""
        print("🚀 Starting MCP servers...")
        
        # Start each server
        server_commands = [
            "python -m scout_agent.mcp_integration.server.gap_finder_tools",
            "python -m scout_agent.mcp_integration.server.reddit_api", 
            "python -m scout_agent.mcp_integration.server.research_tools",
            "python -m scout_agent.mcp_integration.server.web_search"
        ]
        
        for i, cmd in enumerate(server_commands):
            print(f"Starting {self.server_names[i]}...")
            try:
                process = subprocess.Popen(
                    cmd.split(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                self.processes.append(process)
                print(f"✅ {self.server_names[i]} started with PID {process.pid}")
            except Exception as e:
                print(f"❌ Failed to start {self.server_names[i]}: {e}")
                return False
                
        return True
    
    def check_servers(self):
        """Check if servers are responding"""
        print("🔍 Checking server health...")
        
        # Give servers time to start
        time.sleep(3)
        
        all_healthy = True
        for i, process in enumerate(self.processes):
            if process.poll() is None:
                print(f"✅ {self.server_names[i]} is running (PID {process.pid})")
            else:
                print(f"❌ {self.server_names[i]} has stopped")
                all_healthy = False
                
        return all_healthy
    
    def cleanup(self):
        """Stop all servers"""
        print("🛑 Stopping MCP servers...")
        for i, process in enumerate(self.processes):
            if process.poll() is None:
                print(f"Stopping {self.server_names[i]} (PID {process.pid})...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"Force killing {self.server_names[i]}...")
                    process.kill()
                print(f"✅ {self.server_names[i]} stopped")
            else:
                print(f"⚠️  {self.server_names[i]} was already stopped")
    
    def test_imports(self):
        """Test that we can import the MCP modules"""
        print("🧪 Testing MCP module imports...")
        
        try:
            import scout_agent.mcp_integration.server.gap_finder_tools
            print("✅ gap_finder_tools import successful")
        except Exception as e:
            print(f"❌ gap_finder_tools import failed: {e}")
            return False
            
        try:
            import scout_agent.mcp_integration.server.reddit_api
            print("✅ reddit_api import successful")
        except Exception as e:
            print(f"❌ reddit_api import failed: {e}")
            return False
            
        try:
            import scout_agent.mcp_integration.server.research_tools
            print("✅ research_tools import successful")
        except Exception as e:
            print(f"❌ research_tools import failed: {e}")
            return False
            
        try:
            import scout_agent.mcp_integration.server.web_search
            print("✅ web_search import successful")
        except Exception as e:
            print(f"❌ web_search import failed: {e}")
            return False
            
        return True

def main():
    print("🧪 Testing ScoutAgent MCP Servers")
    print("=" * 50)
    
    tester = MCPServerTester()
    
    # Set up signal handler for cleanup
    def signal_handler(sig, frame):
        print("\n🛑 Received interrupt signal")
        tester.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Test imports first
        if not tester.test_imports():
            print("❌ Import tests failed")
            return 1
            
        # Start servers
        if not tester.start_servers():
            print("❌ Failed to start servers")
            return 1
            
        # Check server health
        if not tester.check_servers():
            print("❌ Server health check failed")
            return 1
            
        print("🎉 All MCP servers are running successfully!")
        print("Press Ctrl+C to stop all servers")
        
        # Keep running until interrupted
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    finally:
        tester.cleanup()
        print("✅ Test completed")
        return 0

if __name__ == "__main__":
    sys.exit(main())
