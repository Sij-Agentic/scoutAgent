#!/bin/bash

# Interactive ScoutAgent Container Runner
# This script helps you run ScoutAgent in interactive mode with separate terminals

set -e

CONTAINER_NAME="scout-agent-interactive"
IMAGE_NAME="scout-agent:latest"

echo "🚀 ScoutAgent Interactive Mode Setup"
echo "=================================="

# Check if container is already running
if sudo docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
    echo "✅ Container $CONTAINER_NAME is already running"
else
    echo "🔧 Starting interactive container..."
    sudo docker run -d --name $CONTAINER_NAME $IMAGE_NAME sleep infinity
    echo "✅ Container started successfully"
fi

echo ""
echo "📋 Available Commands:"
echo "====================="
echo ""
echo "1. Start MCP Servers (Terminal 1):"
echo "   sudo docker exec -it $CONTAINER_NAME /bin/bash"
echo "   # Then inside container:"
echo "   python -m scout_agent.mcp_integration.server.gap_finder_tools &"
echo "   python -m scout_agent.mcp_integration.server.reddit_api &"
echo "   python -m scout_agent.mcp_integration.server.research_tools &"
echo "   python -m scout_agent.mcp_integration.server.web_search &"
echo ""
echo "2. Run Main Workflow (Terminal 2):"
echo "   sudo docker exec -it $CONTAINER_NAME /bin/bash"
echo "   # Then inside container:"
echo "   python -m scout_agent.main --target-market \"Knowledge management tools\" --keywords \"bidirectional links,markdown sync\" --subreddits \"PKMS\" --per-query-limit 1"
echo ""
echo "3. Check Server Status:"
echo "   sudo docker exec -it $CONTAINER_NAME /bin/bash"
echo "   # Then inside container:"
echo "   netstat -tlnp | grep -E ':(8000|8001|8002|8004)'"
echo ""
echo "4. Stop Container:"
echo "   sudo docker stop $CONTAINER_NAME && sudo docker rm $CONTAINER_NAME"
echo ""
echo "💡 Tip: Open multiple terminal windows and use the commands above!"
