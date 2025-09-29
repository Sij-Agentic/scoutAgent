#!/bin/bash

# ScoutAgent Main Workflow Script
# This script runs the main ScoutAgent workflow
# Note: MCP servers should be started separately using start_mcp_servers.sh

set -e

# NOTE: Environment variables will be injected by Fargate task definition / Secrets Manager.
# The following .env autoload was used for local runs and is intentionally disabled for AWS runtime.
# If you need local testing, uncomment this block.
# if [ -f .env ]; then
#     echo "📋 Loading environment variables from .env file..."
#     set -a
#     . ./.env
#     set +a
#     while IFS= read -r line; do
#         if [[ $line =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
#             key=$(echo "$line" | cut -d'=' -f1)
#             value=$(echo "$line" | cut -d'=' -f2-)
#             value=$(echo "$value" | sed "s/^['\"]//; s/['\"]$//")
#             export "$key=$value"
#         fi
#     done < .env
#     echo "✅ Environment variables loaded and exported"
# fi

echo "🎯 Starting ScoutAgent Main Workflow..."

# Check if MCP servers are running
echo "🔍 Checking MCP server connectivity..."
for port in 8000 8001 8002 8004; do
    if python -c "import socket; socket.socket().connect_ex(('127.0.0.1', $port))" 2>/dev/null; then
        echo "✅ Port $port is responding"
    else
        echo "❌ Port $port is not responding"
        echo "⚠️  Please start MCP servers first using: ./start_mcp_servers.sh"
        exit 1
    fi
done

echo "✅ All MCP servers are ready!"

# Execute the main ScoutAgent command
echo "🚀 Starting main ScoutAgent workflow..."
python -m scout_agent.main "$@"