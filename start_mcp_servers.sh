#!/bin/bash

# Start all MCP servers in the interactive container
# Run this in Terminal 1

CONTAINER_NAME="scout-agent-interactive"

echo "🔧 Starting MCP Servers in container: $CONTAINER_NAME"
echo "=================================================="

# Check if container is running
if ! sudo docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
    echo "❌ Container $CONTAINER_NAME is not running!"
    echo "Please run: ./run_interactive.sh first"
    exit 1
fi

echo "Starting MCP servers..."

# Start each server in the background with environment variables loaded
sudo docker exec -d $CONTAINER_NAME bash -c "if [ -f .env ]; then set -a; . ./.env; set +a; while IFS= read -r line; do if [[ \$line =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then key=\$(echo \"\$line\" | cut -d'=' -f1); value=\$(echo \"\$line\" | cut -d'=' -f2-); value=\$(echo \"\$value\" | sed \"s/^['\\\"]//; s/['\\\"]$//\"); export \"\$key=\$value\"; fi; done < .env; fi; python -m scout_agent.mcp_integration.server.gap_finder_tools"
echo "✅ Started gap_finder_tools on port 8000"

sudo docker exec -d $CONTAINER_NAME bash -c "if [ -f .env ]; then set -a; . ./.env; set +a; while IFS= read -r line; do if [[ \$line =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then key=\$(echo \"\$line\" | cut -d'=' -f1); value=\$(echo \"\$line\" | cut -d'=' -f2-); value=\$(echo \"\$value\" | sed \"s/^['\\\"]//; s/['\\\"]$//\"); export \"\$key=\$value\"; fi; done < .env; fi; python -m scout_agent.mcp_integration.server.reddit_api"
echo "✅ Started reddit_api on port 8001"

sudo docker exec -d $CONTAINER_NAME bash -c "if [ -f .env ]; then set -a; . ./.env; set +a; while IFS= read -r line; do if [[ \$line =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then key=\$(echo \"\$line\" | cut -d'=' -f1); value=\$(echo \"\$line\" | cut -d'=' -f2-); value=\$(echo \"\$value\" | sed \"s/^['\\\"]//; s/['\\\"]$//\"); export \"\$key=\$value\"; fi; done < .env; fi; python -m scout_agent.mcp_integration.server.research_tools"
echo "✅ Started research_tools on port 8002"

sudo docker exec -d $CONTAINER_NAME bash -c "if [ -f .env ]; then set -a; . ./.env; set +a; while IFS= read -r line; do if [[ \$line =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then key=\$(echo \"\$line\" | cut -d'=' -f1); value=\$(echo \"\$line\" | cut -d'=' -f2-); value=\$(echo \"\$value\" | sed \"s/^['\\\"]//; s/['\\\"]$//\"); export \"\$key=\$value\"; fi; done < .env; fi; python -m scout_agent.mcp_integration.server.web_search"
echo "✅ Started web_search on port 8004"

echo ""
echo "⏳ Waiting for servers to start..."
sleep 5

echo "🔍 Checking server status..."
sudo docker exec $CONTAINER_NAME python -c "
import socket
ports = [8000, 8001, 8002, 8004]
for port in ports:
    try:
        s = socket.socket()
        s.connect(('127.0.0.1', port))
        s.close()
        print(f'✅ Port {port} is responding')
    except:
        print(f'❌ Port {port} is not responding')
"

echo ""
echo "🎯 MCP servers are ready!"
echo "Now you can run the main workflow in another terminal."
